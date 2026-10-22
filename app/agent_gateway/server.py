from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac
import json
import os
import secrets
import time
import uuid
import weakref
import httpx
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .protocol import (ProtocolError, prepare_request, parse_answer, estimate_tokens, sse, content_events,
                       answer_options, finalize_answer, auto_mode_verdict)
from .upstream import NotionAgent, UpstreamError, PUBLIC_MODEL
from .responses import PICKER_MODEL, convert_request, convert_answer, response_object, output_events, response_usage
from .model_map import resolve as resolve_model, entries as model_entries, known as model_known
from .routing import passthrough, HOP_HEADERS
from .sessions import SessionStore, client_session, is_away_summary
from .config import ROOT, LOCAL, PROFILE, PROFILE_LOCAL, PROFILES, CATALOG, profile_path
from .pool import WorkspacePool
from .usage import UsageReporter, context_usage, desktop_footer_enabled

# Raw body bound; images travel as native attachments outside the text payload.
MAX_REQUEST_BYTES = int(os.getenv("NOTION_GATEWAY_MAX_REQUEST_BYTES", "128000000"))


def error_type(status):
    return {401: "authentication_error", 403: "permission_error", 429: "rate_limit_error"}.get(status, "api_error" if status >= 500 else "invalid_request_error")


def session_lock(key):
    lock = app.state.locks.get(key)
    if lock is None:
        lock = app.state.locks[key] = asyncio.Lock()
    return lock


def error_response(message, status=400):
    return JSONResponse({"type": "error", "error": {"type": error_type(status),
                                                    "message": message}}, status_code=status)


def audit(record):
    path = LOCAL / "gateway-audit.jsonl"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(fd, "a") as handle:
        handle.write(json.dumps({"time": time.time(), **record}) + "\n")


@asynccontextmanager
async def lifespan(app):
    LOCAL.mkdir(mode=0o700, exist_ok=True)
    key_path = LOCAL / "gateway-key"
    if not key_path.exists():
        fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as handle:
            handle.write(secrets.token_urlsafe(32))
    app.state.key = key_path.read_text().strip()
    if len(app.state.key) < 24:
        raise RuntimeError("Gateway key must contain at least 24 characters")
    members, skipped, scopes = [], [], set()
    for name in PROFILES:
        session_file = profile_path(name) / "notion-session.json"
        if not session_file.exists():
            if name == PROFILE:
                raise RuntimeError("Notion profile is not configured. Run scripts/setup_gateway.py --help first.")
            skipped.append({"profile": name, "error": "not configured"})
            continue
        agent = NotionAgent(session_file)
        # An expired Notion session must not prevent existing provider routes from working.
        try:
            model = await agent.verify_model()
        except (UpstreamError, httpx.HTTPError) as exc:
            model = {"id": PUBLIC_MODEL, "available": False, "error": str(exc)}
        if name == PROFILE:
            app.state.upstream, app.state.model = agent, model
        elif model.get("available") is False or agent.account_scope in scopes:
            # Never fail over into a workspace without verified Opus 5.5 access.
            skipped.append({"profile": name, "error": model.get("error", "duplicate workspace")})
            await agent.close()
            continue
        scopes.add(agent.account_scope)
        members.append((name, agent))
    app.state.pool = WorkspacePool(members, skipped)
    app.state.usage_reporter = UsageReporter()
    app.state.routes = json.loads((LOCAL / "routes.json").read_text()) if (LOCAL / "routes.json").exists() else {}
    app.state.proxy_http = httpx.AsyncClient(timeout=httpx.Timeout(300, connect=20), follow_redirects=False)
    # One turn at a time per client session; separate sessions and side requests run concurrently.
    app.state.locks = weakref.WeakValueDictionary()
    app.state.sessions = SessionStore(PROFILE_LOCAL / "gateway-sessions.sqlite3",
                                     namespace=app.state.upstream.account_scope, adopt_legacy=PROFILE == "default")
    try:
        yield
    finally:
        for _, agent in app.state.pool.members:
            await agent.close()
        await app.state.proxy_http.aclose()
        app.state.sessions.close()


app = FastAPI(title="Notion local coding gateway (experimental)", lifespan=lifespan, docs_url=None, redoc_url=None)


@app.middleware("http")
async def local_auth(request, call_next):
    host = urlsplit("http://" + request.headers.get("host", "")).hostname
    if host not in {"127.0.0.1", "localhost", "::1", "testserver"}:
        return error_response("Loopback host required", 403)
    if request.client and request.client.host not in {"127.0.0.1", "::1", "testclient"}:
        return error_response("Loopback clients only", 403)
    if request.headers.get("origin"):
        return error_response("Browser-origin requests are not supported", 403)
    if request.url.path.startswith(("/v1/", "/codex/", "/claude/")):
        supplied = (request.headers.get("x-notion-gateway-key") if request.url.path.startswith("/codex/")
                    else request.headers.get("x-api-key") or request.headers.get("authorization", "").removeprefix("Bearer ")) or ""
        if not hmac.compare_digest(supplied, request.app.state.key):
            return error_response("Invalid local gateway key", 401)
    return await call_next(request)


@app.get("/health")
async def health():
    return {"status": "ok", "model": app.state.model, "tool_mode": "text-json-bridge", "usage": "estimated",
            "session_mode": "persistent-client-thread", "profile": PROFILE, "workspaces": app.state.pool.status()}


@app.head("/api/hello")
async def hello():
    return Response(status_code=200)


@app.get("/v1/models")
@app.get("/claude/v1/models")
async def models():
    # PATCHED-BY-OPS: expose every model the workspace offers (model_map.json)
    # instead of the single hard-coded Opus 5.5 entry.
    items = model_entries() or [(PICKER_MODEL, PICKER_MODEL)]
    data = [{"type": "model", "id": slug, "display_name": slug,
             "created_at": "2026-10-18T00:00:00Z"} for slug, _ in items]
    return {"data": data, "has_more": False,
            "first_id": data[0]["id"] if data else None,
            "last_id": data[-1]["id"] if data else None}


@app.get("/v1/notion/usage")
@app.get("/claude/v1/notion/usage")
async def notion_usage(request: Request):
    key, source = client_session(request.headers, {}, "messages")
    record = app.state.sessions.get(key) if source != "unidentified" else {}
    scope = record.get("scope") or app.state.sessions.namespace
    member = next((m for m in app.state.pool.members if m[1].account_scope == scope), None)
    tokens = record.get("result", {}).get("usage", {}).get("input_tokens")
    if member is None and record.get("scope"):
        return JSONResponse({"source": "notion", "status": "unavailable", "profile": None,
                             "rolling": None, "billing_period": None, "observed_at": None,
                             "reason": "session_workspace_unavailable", "context": context_usage(tokens)},
                            headers={"cache-control": "no-store"})
    member = member or app.state.pool.members[0]
    name, upstream = member
    quota = await app.state.usage_reporter.read(upstream)
    return JSONResponse({**quota, "profile": name, "context": context_usage(tokens)},
                        headers={"cache-control": "no-store"})


@app.post("/v1/messages")
@app.post("/claude/v1/messages")
async def messages(request: Request):
    raw = bytearray()
    async for chunk in request.stream():
        raw.extend(chunk)
        if len(raw) > MAX_REQUEST_BYTES:
            return error_response("Request exceeds local gateway size limit", 413)
    try:
        body = json.loads(raw)
        if request.url.path.startswith("/claude/") and body.get("model") != PICKER_MODEL:
            return await passthrough(request, bytes(raw), "claude", "v1/messages")
        prompt, tools, choice, effort = prepare_request(body)
        options = answer_options(body)
    except ProtocolError as exc:
        return error_response(str(exc))
    except (ValueError, TypeError, KeyError, AttributeError):
        return error_response("Invalid or unsupported Messages request; supported: text, images and custom tools")
    message_id = "msg_" + uuid.uuid4().hex
    usage = {"input_tokens": estimate_tokens(prompt), "output_tokens": 0}
    message = {"id": message_id, "type": "message", "role": "assistant", "model": body["model"],
               "content": [], "stop_reason": None, "stop_sequence": None, "usage": usage}
    verdict = auto_mode_verdict(body, options)
    if verdict:
        text, stop, stop_sequence = verdict
        result = {**message, "content": [{"type": "text", "text": text}], "stop_reason": stop,
                  "stop_sequence": stop_sequence, "usage": {**usage, "output_tokens": 1}}
        audit({"id": message_id, "event": "auto_mode_allowed", "protocol": "messages", "input_chars": len(prompt)})
        headers = {"x-notion-auto-mode": "local-allow"}
        if not body.get("stream", False):
            return JSONResponse(result, headers=headers)

        def local_stream():
            yield sse("message_start", message=message)
            yield from content_events(result["content"])
            yield sse("message_delta", delta={"stop_reason": stop, "stop_sequence": stop_sequence},
                      usage=result["usage"])
            yield sse("message_stop")

        return StreamingResponse(local_stream(), media_type="text/event-stream",
                                 headers={**headers, "cache-control": "no-cache"})
    session_key, session_source = client_session(request.headers, body, "messages")
    preserve_history = session_source in {"claude-header", "claude-metadata"} and is_away_summary(body)
    if options["stop_sequences"] or options["schema"] is not None:
        # Side requests (e.g. auto-mode safety classifier) reuse the main session header;
        # keep them out of the main Notion thread so its history is never resynced.
        session_key = hashlib.sha256((session_key + json.dumps(options, sort_keys=True)).encode()).hexdigest()
        session_source = "side-request"
    decorate = (app.state.usage_reporter.decorate
                if desktop_footer_enabled(request.headers, body, options, preserve_history) else None)

    def render(text):
        content, stop = parse_answer(text, tools, choice)
        content, stop, stop_sequence = finalize_answer(content, stop, options)
        return {**message, "content": content, "stop_reason": stop, "stop_sequence": stop_sequence,
                "usage": {**usage, "output_tokens": estimate_tokens(content)}}

    async def execute(progress=None):
        started = time.monotonic()
        audit({"id": message_id, "event": "started", "protocol": "messages", "session_hash": session_key,
               "session_source": session_source})
        try:
            async with session_lock(session_key):
                result, info = await app.state.pool.run(app.state.sessions, session_key, prompt, effort, render,
                                                        progress, stateless=session_source == "side-request",
                                                          preserve_history=preserve_history,
                                                          decorate=decorate,
                                                          model=resolve_model(body.get("model")))
            result = {**result, "id": message_id}
            audit({"id": message_id, "event": "completed", "model": PUBLIC_MODEL,
                   **info, "session_source": session_source, "duration_s": round(time.monotonic() - started, 2),
                   "request_sha256": hashlib.sha256(raw).hexdigest(), "input_chars": len(prompt),
                   "tools": [c["name"] for c in result["content"] if c["type"] == "tool_use"],
                   "stop_reason": result["stop_reason"]})
            return result
        except BaseException as exc:
            audit({"id": message_id, "event": "failed", "error_class": type(exc).__name__,
                   "session_hash": session_key, "duration_s": round(time.monotonic() - started, 2),
                   "error_status": getattr(exc, "status", 502), **getattr(exc, "diagnostics", {})})
            raise

    headers = {"x-notion-usage-estimated": "true", "x-notion-tool-mode": "text-json-bridge"}
    if not body.get("stream", False):
        try:
            return JSONResponse(await execute(), headers=headers)
        except (UpstreamError, ProtocolError, TimeoutError) as exc:
            return error_response(str(exc) or "Notion turn timed out", getattr(exc, "status", 502))

    async def stream():
        # The JSON bridge can need protocol correction. Keep partial prose private
        # until the complete answer (including every tool argument) validates.
        task = asyncio.create_task(execute())
        try:
            yield sse("message_start", message=message)
            while not task.done():
                finished, _ = await asyncio.wait({task}, timeout=5)
                if not finished:
                    yield sse("ping")
            result = task.result()
            for event in content_events(result["content"]):
                yield event
            yield sse("message_delta", delta={"stop_reason": result["stop_reason"],
                                              "stop_sequence": result.get("stop_sequence")},
                      usage=result["usage"])
            yield sse("message_stop")
        except (UpstreamError, ProtocolError, TimeoutError) as exc:
            yield sse("error", error={"type": error_type(getattr(exc, "status", 502)),
                                      "message": str(exc) or "Notion turn timed out"})
        finally:
            if not task.done():
                task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={**headers, "cache-control": "no-cache", "x-accel-buffering": "no"})


@app.head("/claude/api/hello")
async def claude_hello():
    return Response(status_code=200)


@app.post("/claude/v1/messages/count_tokens")
async def count_tokens(request: Request):
    raw = await request.body()
    body = json.loads(raw)
    if body.get("model") == PICKER_MODEL:
        return Response(status_code=404)
    return await passthrough(request, raw, "claude", "v1/messages/count_tokens")


@app.get("/codex/models")
async def codex_models(request: Request):
    # PATCHED-BY-OPS: emit one catalog entry per model in model_map.json, reusing
    # the shipped Opus 5.5 entry as the template (context window, tool support,
    # reasoning levels). Previously this returned a single hard-coded model.
    base = json.loads(CATALOG.read_text())["models"][0]
    items = model_entries() or [(PICKER_MODEL, PICKER_MODEL)]
    models = []
    for slug, _upstream in items:
        entry = dict(base)
        entry["slug"] = slug
        entry["display_name"] = slug
        entry["description"] = "Notion AI model " + slug
        models.append(entry)
    return JSONResponse({"models": models}, headers={"x-notion-catalog-source": "model-map"})


@app.post("/codex/responses/compact")
async def compact(request: Request):
    raw = await request.body()
    if json.loads(raw).get("model") == PICKER_MODEL:
        return error_response("Use local compaction for Notion", 400)
    return await passthrough(request, raw, "codex", "responses/compact")


@app.post("/v1/responses")
@app.post("/codex/responses")
async def responses(request: Request):
    raw = bytearray()
    async for chunk in request.stream():
        raw.extend(chunk)
        if len(raw) > MAX_REQUEST_BYTES:
            return error_response("Request exceeds local gateway size limit", 413)
    try:
        body = json.loads(raw)
        if request.url.path.startswith("/codex/") and body.get("model") != PICKER_MODEL:
            return await passthrough(request, bytes(raw), "codex", "responses")
        if body.get("model") not in {PICKER_MODEL, PUBLIC_MODEL} and not model_known(body.get("model")):
            raise ProtocolError("Unsupported Notion model")
        prompt, tools, choice, effort, mapping = convert_request(body)
    except ProtocolError as exc:
        return error_response(str(exc))
    except (ValueError, TypeError, KeyError, AttributeError):
        return error_response("Invalid or unsupported Responses request")
    response_id = "resp_" + uuid.uuid4().hex
    if os.getenv("NOTION_GATEWAY_DEBUG_REQUESTS") == "1":
        # Opt-in synthetic-client diagnostics only; not enabled by the installed command.
        fd = os.open(LOCAL / "debug-responses-request.json", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as handle:
            handle.write(json.dumps(body))
    audit({"id": response_id, "event": "started", "protocol": "responses", "input_chars": len(prompt),
           "input_bytes": len(prompt.encode()), "request_bytes": len(raw), "tool_names": list(mapping)})
    initial = response_object(response_id, body["model"])
    session_key, session_source = client_session(request.headers, body, "responses")

    def render(text):
        output = convert_answer(text, tools, choice, mapping)
        return response_object(response_id, body["model"], output, response_usage(prompt, output), "completed")

    async def execute():
        started = time.monotonic()
        try:
            async with session_lock(session_key):
                result, info = await app.state.pool.run(app.state.sessions, session_key, prompt, effort, render,
                                                  model=resolve_model(body.get("model")))
            result = {**result, "id": response_id, "created_at": initial["created_at"]}
            audit({"id": response_id, "event": "completed", "protocol": "responses", "model": body["model"],
                   **info, "session_source": session_source, "duration_s": round(time.monotonic() - started, 2),
                   "input_chars": len(prompt), "tools": [x["name"] for x in result["output"] if x["type"] != "message"]})
            return result
        except BaseException as exc:
            audit({"id": response_id, "event": "failed", "error_class": type(exc).__name__, "protocol": "responses",
                   "session_hash": session_key, "duration_s": round(time.monotonic() - started, 2),
                   "error_status": getattr(exc, "status", 502), **getattr(exc, "diagnostics", {})})
            raise

    if not body.get("stream", False):
        try:
            return JSONResponse(await execute(), headers={"x-notion-usage-estimated": "true"})
        except (UpstreamError, ProtocolError, TimeoutError) as exc:
            return error_response(str(exc) or "Notion timed out", getattr(exc, "status", 502))

    async def stream():
        sequence = 0

        def emit(event):
            nonlocal sequence
            event = {**event, "sequence_number": sequence}
            sequence += 1
            return "event: " + event["type"] + "\ndata: " + json.dumps(event, ensure_ascii=False) + "\n\n"

        task = asyncio.create_task(execute())
        try:
            yield emit({"type": "response.created", "response": initial})
            yield emit({"type": "response.in_progress", "response": initial})
            while not task.done():
                finished, _ = await asyncio.wait({task}, timeout=5)
                if not finished:
                    yield ": ping\n\n"
            result = task.result()
            for event in output_events(result["output"]):
                yield emit(event)
            yield emit({"type": "response.completed", "response": result})
        except (UpstreamError, ProtocolError, TimeoutError) as exc:
            yield emit({"type": "response.failed", "response": {**initial, "status": "failed",
                        "error": {"code": "server_error", "message": str(exc) or "Notion timed out"}}})
        finally:
            if not task.done():
                task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"cache-control": "no-cache",
                             "x-accel-buffering": "no", "x-notion-usage-estimated": "true"})
