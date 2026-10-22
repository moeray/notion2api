from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from urllib.parse import urlsplit

# PATCHED-BY-OPS: httpx's TLS fingerprint is rejected by Notion's edge for the
# agent endpoints (createAgentThread answers 400 UserValidationError while the
# low-risk getAvailableModels passes). curl_cffi impersonates Chrome's TLS/JA3
# and the same request returns 200. Keep the dependency optional at import time
# so the module still loads where curl_cffi is absent.
try:
    from curl_cffi.requests import AsyncSession
    from curl_cffi.requests.errors import RequestsError
    _HAS_CURL_CFFI = True
except Exception:  # pragma: no cover - fallback keeps the old transport usable
    AsyncSession = None
    RequestsError = Exception
    _HAS_CURL_CFFI = False

import httpx

from .media import FilePrompt, IMAGE_EXTENSIONS

MODEL = "albuquerque-quinn"
PUBLIC_MODEL = "claude-opus-5-5"
SESSION_HEADERS = {"user-agent", "notion-client-version", "notion-audit-log-platform",
                   "x-notion-cell-hint", "x-notion-cell-hint-source"}


# Body/stop-reason phrases meaning the workspace is out of AI quota (inspected, never relayed).
QUOTA_MARKERS = ("rate limit", "ratelimit", "rate_limit", "quota", "credit", "usage limit", "exhausted")


class UpstreamError(Exception):
    def __init__(self, message: str, status: int = 502, rejected: bool = False, exhausted: bool = False,
                 retry_after: float | None = None):
        super().__init__(message)
        self.status = status
        self.rejected = rejected
        self.diagnostics = {}
        # Workspace quota/rate limit: the gateway may continue the session in another workspace.
        self.exhausted = exhausted
        self.retry_after = retry_after


def quota_signal(response):
    if response.status_code == 429:
        return True
    if not 400 <= response.status_code < 500 or response.status_code in {401, 403, 404, 408}:
        return False
    return any(marker in response.text[:4000].lower() for marker in QUOTA_MARKERS)


def retry_after_seconds(response):
    try:
        value = float(response.headers.get("retry-after", ""))
    except ValueError:
        return None
    return value if 0 < value <= 86400 else None


class Transcript:
    """Apply snapshots by entity ID; never mistake a partial answer for completion."""

    def __init__(self, after_sequence=-1, previous_turn_id=None):
        self.entities = {}
        self.cursor = None
        self.session = {}
        self.after_sequence = after_sequence
        self.sequence = after_sequence
        self.previous_turn_id = previous_turn_id
        self.turn_id = None
        self.user_sequence = after_sequence
        # Any Notion-side entity after our message (thinking, output, completion) proves the
        # workspace is serving this turn. Our own message and quota banners do not count.
        self.active = False

    def update(self, page: dict):
        self.cursor = page.get("forward_cursor", self.cursor)
        self.session = page.get("session") or self.session
        self.sequence = max(self.sequence, page.get("forward_after_sequence", -1))
        for patch in page.get("patches", []):
            if patch.get("op") == "put":
                entity = patch.get("entity", {})
                self.sequence = max(self.sequence, entity.get("sequence", -1))
                if entity.get("kind") == "user_message":
                    self.user_sequence = max(self.user_sequence, entity.get("sequence", -1))
                if entity.get("sequence", -1) > self.after_sequence and entity.get("kind") != "user_message" \
                        and not str(entity.get("id", "")).startswith("rate-limit-status"):
                    self.active = True
                # Thinking and Notion's internal execution details are not relayed.
                if entity.get("kind") in {"assistant_message", "turn_completed", "error"}:
                    self.entities[entity["id"]] = entity
            elif patch.get("op") == "remove":
                self.entities.pop(patch.get("id"), None)
            elif patch.get("op") == "session":
                self.session = patch.get("session", self.session)

    def partial_text(self):
        """Text generated so far in the current turn (may be incomplete JSON)."""
        messages = sorted(
            (e for e in self.entities.values() if e.get("kind") == "assistant_message"
             and e.get("sequence", 0) > self.after_sequence
             and e.get("sequence", 0) > self.user_sequence
             and (self.user_sequence > self.after_sequence or not self.previous_turn_id
                  or e.get("turn_id") != self.previous_turn_id)),
            key=lambda e: e.get("sequence", 0),
        )
        return "\n".join(block.get("text", "") for message in messages
                         for block in message.get("content", []) if block.get("type") == "text")

    def result(self):
        completed = [e for e in self.entities.values() if e.get("kind") == "turn_completed"
                     and e.get("sequence", 0) > self.after_sequence
                     and e.get("sequence", 0) > self.user_sequence
                     and (self.user_sequence > self.after_sequence or not self.previous_turn_id
                          or e.get("turn_id") != self.previous_turn_id)]
        if not completed:
            return None
        done = max(completed, key=lambda e: e.get("sequence", 0))
        self.turn_id = done.get("turn_id")
        if done.get("stop_reason") != "end_turn":
            reason = str(done.get("stop_reason"))
            raise UpstreamError("Notion turn did not finish normally: " + reason,
                                exhausted=any(marker in reason.lower() for marker in QUOTA_MARKERS))
        messages = sorted(
            (e for e in self.entities.values() if e.get("kind") == "assistant_message"
             and e.get("turn_id") == done.get("turn_id")
             and self.user_sequence < e.get("sequence", 0) <= done.get("sequence", 0)),
            key=lambda e: e.get("sequence", 0),
        )
        text = "\n".join(
            block.get("text", "") for message in messages
            for block in message.get("content", []) if block.get("type") == "text"
        ).strip()
        if not text:
            raise UpstreamError("Notion completed without an assistant text response")
        return text


class NotionAgent:
    def __init__(self, session_file: Path, timeout: float | None = None):
        config = json.loads(session_file.read_text())
        # Credential destination is intentionally fixed, not user-overridable.
        if config.get("origin") != "https://app.notion.com":
            raise ValueError("Only https://app.notion.com is supported")
        self.space_id = config["space_id"]
        self.account_scope = hashlib.sha256((config["user_id"] + ":" + self.space_id).encode()).hexdigest()
        self.timeout = timeout or float(os.getenv("NOTION_GATEWAY_TIMEOUT", "300"))
        # Fail a turn whose transcript stops advancing instead of waiting out the full timeout.
        self.idle_timeout = float(os.getenv("NOTION_GATEWAY_IDLE_TIMEOUT", "90"))
        self.image_files = OrderedDict()
        headers = {k: v for k, v in config.get("headers", {}).items() if k.lower() in SESSION_HEADERS}
        headers.update({"origin": config["origin"], "referer": config["origin"] + "/chat",
                        "x-notion-space-id": self.space_id,
                        "x-notion-active-user-header": config["user_id"]})
        self.base_url = config["origin"]
        if _HAS_CURL_CFFI:
            # Chrome TLS/JA3 fingerprint; httpx gets a 400 from Notion's edge.
            self.client = AsyncSession(
                impersonate="chrome", headers=headers,
                cookies={c["name"]: c["value"] for c in config["cookies"]},
                timeout=40, allow_redirects=False,
            )
        else:
            self.client = httpx.AsyncClient(
                base_url=config["origin"], headers=headers,
                cookies={c["name"]: c["value"] for c in config["cookies"]},
                timeout=httpx.Timeout(40, connect=15), follow_redirects=False,
            )

    async def close(self):
        if _HAS_CURL_CFFI:
            await self.client.close()
        else:
            await self.client.aclose()

    async def post(self, endpoint: str, body: dict):
        url = self.base_url + "/api/v3/" + endpoint
        read_only = endpoint in {"getThreadTranscript", "getAvailableModels", "getCreditRateLimitStatus"}
        # Read-only requests are safe to retry. Writes require session-level
        # reconciliation; never replay a generation POST because its response broke.
        # PATCHED-BY-OPS: a 429 is a rate limit, not a broken response - the turn
        # was never accepted, so a short backoff-and-retry is safe even for writes.
        attempts = 4 if read_only else 2
        for attempt in range(attempts):
            try:
                if _HAS_CURL_CFFI:
                    response = await self.client.post(url, json=body)
                else:
                    response = await self.client.post("/api/v3/" + endpoint, json=body)
            except (RequestsError, httpx.HTTPError) as exc:
                if read_only and attempt + 1 < attempts:
                    await asyncio.sleep(0.5 * (2 ** attempt))
                    continue
                error = UpstreamError("Notion transport failed (" + type(exc).__name__ + ")")
                error.diagnostics = {"upstream_endpoint": endpoint,
                                     "transport_error": type(exc).__name__,
                                     "read_attempts": attempt + 1}
                raise error from None
            # PATCHED-BY-OPS: Notion's short-window throttle answers 429 with a
            # billing-period retryAfter that is useless for a live turn. Back
            # off briefly (5s, 10s) and retry instead of failing the request -
            # this is what keeps a large multi-chunk turn alive.
            if response.status_code == 429 and attempt + 1 < attempts:
                wait = 5 * (2 ** attempt)
                print("[rate-limit] %s 429, retry %d/%d in %ds"
                      % (endpoint, attempt + 1, attempts - 1, wait), flush=True)
                await asyncio.sleep(wait)
                continue
            if response.status_code in {408, 500, 502, 503, 504} and read_only and attempt + 1 < attempts:
                await asyncio.sleep(0.5 * (2 ** attempt))
                continue
            break
        if response.status_code != 200:
            status = response.status_code if response.status_code in {401, 403, 429} else 502
            # Never include arbitrary upstream response bodies, credentials or trace URLs.
            error = UpstreamError(f"Notion {endpoint} returned HTTP {response.status_code}", status,
                                  rejected=400 <= response.status_code < 500 and response.status_code != 408,
                                  exhausted=quota_signal(response), retry_after=retry_after_seconds(response))
            try:
                detail = response.json()
            except ValueError:
                detail = {}
            if not isinstance(detail, dict):
                detail = {}
            debug = detail.get("debugMessage", "")
            # DEBUG-BY-OPS: keep the rejected upstream body on disk so a 400 can
            # be diagnosed locally (the shipped code deliberately hides it).
            try:
                with open(os.path.join(os.path.dirname(__file__), "..", "..", ".local",
                                       "upstream-errors.log"), "a", encoding="utf-8") as _fh:
                    _fh.write("%s %s %s\n" % (endpoint, response.status_code, response.text[:1500]))
            except Exception:
                pass
            # A larger UTF-8 prompt may pass character limits but overflow Notion's
            # internal duplicated DynamoDB write. Do not expose arbitrary error text.
            if response.status_code == 400 and isinstance(debug, str) and re.fullmatch(
                    r"create_session DynamoDB transaction action is \d+ bytes; max is \d+", debug):
                error.status = 413
                error.args = ("Notion rejected an oversized UTF-8 context; compact the conversation or split the input",)
            error.diagnostics = {"upstream_endpoint": endpoint, "upstream_status": response.status_code,
                                 "request_bytes": len(json.dumps(body).encode("utf-8")),
                                 "reasoning_effort": body.get("reasoningEffort")}
            for field in ("name", "errorId"):
                value = detail.get(field)
                if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,80}", value):
                    error.diagnostics["upstream_" + field] = value
            raise error
        try:
            return response.json()
        except ValueError:
            raise UpstreamError("Notion returned a non-JSON response") from None

    async def verify_model(self):
        directory = await self.post("getAvailableModels", {"spaceId": self.space_id})
        model = next((m for m in directory.get("models", []) if m.get("model") == MODEL), None)
        if not model or model.get("isDisabled") or model.get("restrictedForPersonalAgent"):
            raise UpstreamError("Opus 5.5 is not enabled for this Notion workspace", 403)
        if model.get("modelMessage") != "Opus 5.5":
            raise UpstreamError("Notion model mapping changed; refusing an unverified model")
        return {"id": PUBLIC_MODEL, "upstream_id": MODEL, "name": model["modelMessage"]}

    async def start_turn(self, prompt, effort, thread_id, event_id, continuation=False, model=None):
        # PATCHED-BY-OPS: `model` carries the upstream id resolved from the
        # client's slug (see model_map.py). None keeps the historical behaviour
        # of always using the built-in default.
        # AUTO-BY-OPS: `model == "auto"` means "let Notion pick" - the thread is
        # created without model/reasoningEffort and follow-up turns omit
        # modelSelection entirely. Used by the `notion-auto` logical model, which
        # keeps working when the explicit paid models are out of AI credits.
        auto = str(model or "").lower() == "auto"
        content = [{"type": "text", "text": [[str(prompt)]]}]
        content.extend({"type": "file", "file_id": file_id} for file_id in getattr(prompt, "file_ids", []))
        if continuation:
            # PATCHED-BY-OPS: a web capture of an AUTO follow-up (2026-10-21)
            # shows the real client sends NO modelSelection at all:
            #   {spaceId, threadId, event, surface, clientEventId}
            # Our earlier {"type":"auto"} guess is invalid and earned
            # 400 ValidationError "Invalid input." on every second turn.
            # An explicit model still pins the turn (verified capture).
            reply = {
                "spaceId": self.space_id, "threadId": thread_id,
                "event": {"type": "user.message", "content": content},
                "surface": "full_page_chat", "clientEventId": event_id,
            }
            if not auto:
                reply["modelSelection"] = {"type": "explicit", "model": model or MODEL,
                                           "reasoningEffort": effort}
            await self.post("sendEventToAgentThread", reply)
        else:
            # PATCHED-BY-OPS: a real web capture (2026-10-20, max effort) uses
            # TOP-LEVEL model/reasoningEffort here; with the curl_cffi transport
            # it sticks and agent_provider comes back albuquerque-quinn / max.
            # The 400 that pushed the author to `modelSelection` was the httpx
            # TLS fingerprint, not the field names.
            body = {
                "type": "personal_agent", "spaceId": self.space_id, "threadId": thread_id,
                "enableSuggestedEditsTools": True,
                "createdSource": "full_page_chat", "content": content,
                "policies": {"approval_mode": "review_for_me"},
                "agentMemorySettings": {"useMemories": True, "excludeChatFromMemories": False},
                "browserEnabled": False, "clientMessageId": event_id,
            }
            if not auto:
                body["model"] = model or MODEL
                body["reasoningEffort"] = effort
            reply = await self.post("createAgentThread", body)
            if auto:
                # The response names the model Notion actually picked; without
                # this line an auto thread gives no visibility into what ran.
                provider = ((reply or {}).get("thread") or {}).get("data", {}).get("agent_provider") or {}
                if provider.get("model"):
                    print("[notion-auto] upstream picked model=%s effort=%s"
                          % (provider.get("model"), provider.get("reasoningEffort")),
                          flush=True)

    async def prepare_images(self, prompt, thread_id, continuation):
        """Upload before committing a pending generation; an upload cannot strand a turn."""
        target = {"type": "thread", "threadId": thread_id} if continuation else {"type": "user"}
        attached = []
        for image_id, image in prompt.images.items():
            key = (thread_id, image_id)
            file_id = self.image_files.get(key)
            if file_id is None:
                file_id = await self.upload_image(image_id, image, target)
                self.image_files[key] = file_id
                while len(self.image_files) > 128:
                    self.image_files.popitem(last=False)
            self.image_files.move_to_end(key)
            attached.append({"image_id": image_id, "file_id": file_id})
        # One text block must precede the file blocks. The mapping identifies images
        # inside tool results or multiple historical messages without exposing bytes.
        note = ("Attached files are the actual images referenced by image_id in the client messages. "
                "Inspect the images directly and associate each with its original message/tool result. "
                "Attachment map: " + json.dumps(attached, separators=(",", ":")) + "\n")
        return FilePrompt(note + prompt, [item["file_id"] for item in attached])

    async def upload_image(self, image_id, image, target):
        descriptor = await self.post("createAgentServiceFileUploadURL", {
            "spaceId": self.space_id, "target": target,
            "filename": "image-" + image_id[:16] + "." + IMAGE_EXTENSIONS[image.media_type],
            "mediaType": image.media_type, "sizeBytes": len(image.data),
        })
        try:
            file_id, upload = descriptor["file"]["id"], descriptor["upload"]
            if not isinstance(file_id, str) or not file_id:
                raise ValueError("file id")
            parts = None
            # Storage must never receive Notion cookies or incoming client credentials.
            async with httpx.AsyncClient(timeout=httpx.Timeout(90, connect=15), follow_redirects=False) as storage:
                if upload["type"] == "single_part":
                    await self.upload_part(storage, upload, image.data)
                elif upload["type"] == "multi_part":
                    part_size = upload["part_size_bytes"]
                    if type(part_size) is not int or part_size <= 0:
                        raise ValueError("part size")
                    expected = (len(image.data) + part_size - 1) // part_size
                    if len(upload["parts"]) != expected:
                        raise ValueError("part count")
                    parts = []
                    for number, part in enumerate(upload["parts"], 1):
                        if part["part_number"] != number:
                            raise ValueError("part number")
                        response = await self.upload_part(storage, part,
                            image.data[(number - 1) * part_size:number * part_size])
                        etag = response.headers.get("etag")
                        if not etag or not etag.strip():
                            raise ValueError("missing ETag")
                        parts.append({"partNumber": number, "etag": etag})
                else:
                    raise ValueError("upload type")
        except (KeyError, TypeError, ValueError):
            raise UpstreamError("Notion returned an unsupported image upload descriptor", rejected=True) from None
        result = await self.post("completeAgentServiceFileUpload", {
            "spaceId": self.space_id, "target": target, "fileId": file_id,
            **({"parts": parts} if parts is not None else {}),
        })
        if not isinstance(result, dict) or result.get("id") != file_id:
            raise UpstreamError("Notion did not confirm the uploaded image", rejected=True)
        return file_id

    @staticmethod
    async def upload_part(storage, part, data):
        url = urlsplit(part["url"])
        if (url.scheme != "https" or url.username or url.password or url.port not in {None, 443}
                or not re.fullmatch(r"(?:[a-z0-9-]+\.)?s3(?:[.-][a-z0-9-]+)?\.amazonaws\.com", url.hostname or "")
                or part["method"] != "PUT"):
            raise ValueError("upload destination")
        headers = part.get("headers", {})
        if not isinstance(headers, dict) or any(
                not isinstance(k, str) or not isinstance(v, str)
                or k.lower() in {"cookie", "authorization", "proxy-authorization", "host"}
                for k, v in headers.items()):
            raise ValueError("upload headers")
        try:
            response = await storage.put(part["url"], headers=headers, content=data)
        except httpx.HTTPError as exc:
            raise UpstreamError("Image storage transport failed (" + type(exc).__name__ + ")",
                                rejected=True) from None
        if not 200 <= response.status_code < 300:
            raise UpstreamError(f"Image storage upload returned HTTP {response.status_code}", rejected=True)
        return response

    async def turn_is_absent(self, thread_id, cursor, after_sequence, previous_turn_id, submitted_at=None):
        """Reconcile an unacknowledged continuation against a stable, idle transcript.

        A timeout alone is never evidence that a write failed. Require the remote
        session AND transcript to remain at the last completed checkpoint, with
        no queued input. Missing fields, new events and read errors fail closed.
        Initial thread creation has no checkpoint and cannot use this recovery.
        """
        if not cursor or after_sequence < 0 or not previous_turn_id:
            return False
        if submitted_at is not None:
            await asyncio.sleep(max(0, min(5, 5 - (time.time() - submitted_at))))
        body = {"spaceId": self.space_id, "threadId": thread_id, "direction": "forward",
                "limit": 100, "cursor": cursor}
        for attempt in range(3):
            page = await self.post("getThreadTranscript", body)
            session = page.get("session") or {}
            if (session.get("status") != "idle" or session.get("sequence") != after_sequence
                    or page.get("session_pending") is not False
                    or page.get("forward_after_sequence") != after_sequence
                    or page.get("has_more_forward") is not False
                    or page.get("patches") != []):
                return False
            if attempt < 2:
                await asyncio.sleep(1)
        return True

    async def wait_for_turn(self, thread_id, cursor=None, after_sequence=-1, previous_turn_id=None, on_text=None):
        try:
            return await self._wait_for_turn(thread_id, cursor, after_sequence, previous_turn_id, on_text)
        except TimeoutError:
            error = UpstreamError(f"Notion turn did not complete within {self.timeout:g}s; "
                                  "the original turn is preserved for retry", 504)
            error.diagnostics = {"error_code": "upstream_timeout", "stage": "poll", "timeout_s": self.timeout}
            raise error from None

    async def _wait_for_turn(self, thread_id, cursor=None, after_sequence=-1, previous_turn_id=None, on_text=None):
        async with asyncio.timeout(self.timeout):
            loop = asyncio.get_running_loop()
            progressed = loop.time()
            state = Transcript(after_sequence, previous_turn_id)
            state.cursor = cursor
            seen = ""
            marker = None
            while True:
                body = {"spaceId": self.space_id, "threadId": thread_id, "direction": "forward", "limit": 100}
                if state.cursor:
                    body["cursor"] = state.cursor
                page = await self.post("getThreadTranscript", body)
                state.update(page)
                # Drain all pages before resolving turn completion.
                if page.get("has_more_forward"):
                    continue
                if on_text:
                    partial = state.partial_text()
                    if partial != seen:
                        seen = partial
                        on_text(partial)
                try:
                    result = state.result()
                except UpstreamError as exc:
                    if state.turn_id:
                        # A failed but completed turn is safe to advance past on
                        # a changed request. An unfinished turn is not.
                        exc.completed_turn = {"cursor": state.cursor, "sequence": state.sequence,
                                              "turn_id": state.turn_id}
                    raise
                if result is not None:
                    return result, state.cursor, state.sequence, state.turn_id
                if state.session.get("pending_input"):
                    raise UpstreamError("Notion requested an internal action; local tool bridging cannot approve it")
                current = (state.sequence, state.active, len(state.partial_text()))
                if current != marker:
                    marker, progressed = current, loop.time()
                elif state.active and self.idle_timeout > 0 and loop.time() - progressed > self.idle_timeout:
                    error = UpstreamError(f"Notion turn made no progress for {self.idle_timeout:g}s; "
                                          "the original turn is preserved for retry", 504)
                    error.diagnostics = {"error_code": "upstream_stalled", "stage": "poll",
                                         "sequence": state.sequence, "session_status": state.session.get("status")}
                    raise error
                await asyncio.sleep(0.75 if on_text else 1.5)

    async def complete(self, prompt: str, effort: str = "medium"):
        """Standalone diagnostic call; client endpoints use the persistent session store."""
        thread_id = str(uuid.uuid4())
        await self.start_turn(prompt, effort, thread_id, str(uuid.uuid4()))
        text, *_ = await self.wait_for_turn(thread_id)
        return text, thread_id
