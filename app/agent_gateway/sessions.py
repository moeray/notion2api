"""Durable client-session bindings and incremental Notion turns.

Only hashes of client history are retained. The last validated response is cached
so retries preserve tool IDs. Never re-submit an inference with an unknown outcome.
Call run() under the server's inference lock, including response validation/commit.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import uuid
from pathlib import Path

from .upstream import UpstreamError

MARKER = "CLIENT_CONTEXT_JSON:\n"
# Bound on what is actually sent per turn (append mode sends only new messages).
MAX_SEND_CHARS = int(os.getenv("NOTION_GATEWAY_MAX_SEND_CHARS", "180000"))
MAX_TOOL_RESULT_CHARS = int(os.getenv("NOTION_GATEWAY_MAX_TOOL_RESULT_CHARS", "16000"))
# Notion re-reads the whole thread every turn, so cost grows with thread length.
# Optional (0 = off; Notion manages its own thread context): past this many accumulated
# chars, continue on a fresh thread seeded with recent history.
THREAD_MAX_CHARS = int(os.getenv("NOTION_GATEWAY_THREAD_MAX_CHARS", "0"))
ROLLOVER_SEED_CHARS = int(os.getenv("NOTION_GATEWAY_ROLLOVER_SEED_CHARS", "80000"))
# Optional tighter bound for stateless side requests (0 = same as MAX_SEND_CHARS).
SIDE_MAX_SEND_CHARS = int(os.getenv("NOTION_GATEWAY_SIDE_MAX_SEND_CHARS", "0"))


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def client_session(headers, body, protocol):
    headers = {k.lower(): v for k, v in headers.items()}
    identity = headers.get("x-notion-session-id")
    source = "explicit"
    agent = request_class = ""
    if not identity and protocol == "messages":
        identity = headers.get("x-claude-code-session-id")
        source = "claude-header"
        agent = headers.get("x-claude-code-agent-id", "")
        request_class = headers.get("x-claude-code-request-class", "")
        if not identity:
            metadata = body.get("metadata")
            uid = metadata.get("user_id", "") if isinstance(metadata, dict) else ""
            try:
                meta = json.loads(uid)
                identity = meta.get("session_id") if isinstance(meta, dict) else None
            except (ValueError, TypeError):
                match = re.search(r"_session_([0-9a-fA-F-]{36})$", uid) if isinstance(uid, str) else None
                identity = match.group(1) if match else None
            source = "claude-metadata"
    if not identity and protocol == "responses":
        meta = body.get("client_metadata")
        meta = meta if isinstance(meta, dict) else {}
        identity = (meta.get("thread_id") or headers.get("x-codex-thread-id")
                    or meta.get("session_id") or headers.get("session_id")
                    or body.get("prompt_cache_key"))
        source = "codex-thread"
    if not isinstance(identity, str) or not identity:
        # Never guess identity from identical prompts; unrelated clients must not mix.
        identity, source = str(uuid.uuid4()), "unidentified"
    return digest([protocol, identity, agent, request_class]), source


def stable_directive(block):
    if block.get("type") != "text":
        return block
    text = block["text"]
    text = re.sub(r"^x-anthropic-billing-header:[^\n]*(?:\n|$)", "", text)
    text = re.sub(r"<total_tokens>\s*\d+ tokens left\s*</total_tokens>", "", text).strip()
    return {**block, "text": text} if text else None


def context_parts(prompt):
    preamble, separator, raw = prompt.partition(MARKER)
    if not separator:
        raise ValueError("Missing bridge context")
    context = json.loads(raw)
    context["system"] = [v for b in context["system"] if (v := stable_directive(b)) is not None]
    atoms, directives = [], []
    for message in context.pop("messages"):
        if message["role"] == "system":
            blocks = [v for b in message["content"] if (v := stable_directive(b)) is not None]
            if blocks:
                directives.append({"role": "system", "content": blocks})
        else:
            atoms.extend({"role": message["role"], "content": block} for block in message["content"])
    context["message_system"] = directives
    return preamble, context, atoms


def delivered_atoms(result):
    """Canonical client-visible output, including IDs assigned by this bridge."""
    if "output" not in result:
        return [{"role": "assistant", "content": block} for block in result.get("content", [])]
    atoms = []
    for item in result["output"]:
        kind = item["type"]
        if kind == "message":
            atoms.extend({"role": "assistant", "content": {"type": "text", "text": b["text"]}}
                         for b in item["content"])
        elif kind in {"function_call", "custom_tool_call"}:
            name = (item.get("namespace", "") + "." if item.get("namespace") else "") + item["name"]
            args = json.loads(item["arguments"]) if kind == "function_call" else {"input": item["input"]}
            atoms.append({"role": "assistant", "content": {
                "type": "tool_use", "id": item["call_id"], "name": name, "input": args}})
    return atoms


def group(atoms):
    messages = []
    for atom in atoms:
        if messages and messages[-1]["role"] == atom["role"]:
            messages[-1]["content"].append(atom["content"])
        else:
            messages.append({"role": atom["role"], "content": [atom["content"]]})
    return messages


def size(value):
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")))


def compact_tool_result(atom):
    block = atom["content"]
    if block.get("type") != "tool_result":
        return atom
    text = "\n".join(b["text"] for b in block["content"])
    limit = MAX_TOOL_RESULT_CHARS
    if len(text) <= limit:
        return atom
    marker = f"\n[Bridge truncated tool result: {len(text)} original characters; request narrower file ranges or filtered command output for omitted data.]\n"
    half = max(0, (limit - len(marker)) // 2)
    shortened = text[:half] + marker + (text[-half:] if half else "")
    return {**atom, "content": {**block, "content": [{"type": "text", "text": shortened}]}}


def drop_oldest(atoms, limit):
    """Drop the oldest history to fit limit, cutting only at user input so tool pairs stay intact."""
    sizes = [size(a) for a in atoms]
    if sum(sizes) <= limit:
        return atoms, 0
    best, latest, total, unmatched = None, None, 0, set()
    for i in range(len(atoms) - 1, 0, -1):
        total += sizes[i]
        if total > limit and latest is not None:
            break
        block = atoms[i]["content"]
        if block.get("type") == "tool_result":
            unmatched.add(block.get("tool_use_id"))
        elif block.get("type") == "tool_use":
            unmatched.discard(block.get("id"))
        elif atoms[i]["role"] == "user" and not unmatched:
            latest = latest or i
            if total <= limit:
                best = i
    # If even the newest turn exceeds the limit, keep just that turn.
    best = best or latest
    return (atoms[best:], best) if best else (atoms, 0)


def incremental_prompt(prompt, record, budget=None):
    preamble, settings, atoms = context_parts(prompt)
    hashes = [digest(atom) for atom in atoms]
    setting_hashes = {k: digest(v) for k, v in settings.items()}
    snapshot = {"history": hashes, "settings": setting_hashes,
                "system_blocks": {k: [digest(b) for b in settings[k]] for k in ("system", "message_system")}}
    fingerprint = digest([settings, atoms])
    previous = record.get("snapshot", {})
    history = previous.get("history", [])
    append = bool(previous) and hashes[:len(history)] == history
    start = len(history) if append else 0
    # Acknowledged output is already present in Notion. Only relay assigned tool IDs,
    # not a second copy of its prose, code patches, or tool arguments.
    acknowledged = previous.get("assistant", [])
    if append and acknowledged and hashes[start:start + len(acknowledged)] == acknowledged:
        start += len(acknowledged)
    else:
        acknowledged = []
    mode = "initial" if not record else "append" if append else "resync"
    changes = {}
    for k, v in settings.items():
        if not previous or previous.get("settings", {}).get(k) != setting_hashes[k]:
            prior_blocks = previous.get("system_blocks", {}).get(k)
            if previous and prior_blocks is not None and isinstance(v, list):
                current_blocks = [digest(b) for b in v]
                if current_blocks[:len(prior_blocks)] == prior_blocks:
                    changes[k + "_append"] = v[len(prior_blocks):]
                    continue
            changes[k] = v
    pending_atoms = atoms[start:]
    kept = [compact_tool_result(atom) for atom in pending_atoms]
    truncated = sum(a != b for a, b in zip(pending_atoms, kept))
    omitted = 0
    if not append:
        # A full-history send must fit; the oldest turns are the cheapest to lose.
        overhead = len(preamble) + size({"mode": mode, **changes}) + 2000
        kept, omitted = drop_oldest(kept, (budget or MAX_SEND_CHARS) - overhead)
    payload = {"mode": mode, **changes, "messages": group(kept)}
    if omitted:
        payload["omitted_earlier_items"] = omitted
        payload["omitted_note"] = ("The oldest client history was omitted by the bridge to bound context; "
                                   "rely on the remaining history and re-read files if details are needed.")
    unset = sorted(set(previous.get("settings", {})) - set(settings)) if previous else []
    if unset:
        payload["unset"] = unset
    if acknowledged and previous.get("tool_ids"):
        payload["previous_tool_ids"] = previous["tool_ids"]
    snapshot["truncated_tool_results"] = truncated
    instructions = (
        "Continue the existing client conversation; do not repeat prior actions. Keep unchanged settings. "
        "Fields ending _append add to the named setting; unset removes settings. previous_tool_ids maps "
        "your previous tool calls in order to client-assigned IDs. Respond only to new messages, in the "
        "same JSON content envelope. Resync replaces client history with supplied history.\n"
    )
    # The complete protocol description is needed only on initial binding/resync.
    rendered = (preamble if not previous else "") + instructions + MARKER + json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"))
    return rendered, snapshot, fingerprint, mode


class SessionStore:
    def __init__(self, path: Path, namespace="", adopt_legacy=False):
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        os.close(fd)
        os.chmod(path, 0o600)
        self.db = sqlite3.connect(path)
        self.namespace = namespace
        self.db.execute("CREATE TABLE IF NOT EXISTS sessions (key TEXT PRIMARY KEY, data TEXT NOT NULL)")
        if namespace and adopt_legacy:
            # One-time migration of the original single-account installation.
            # Future account/workspace switches use a different namespace.
            self.db.execute("UPDATE sessions SET key = ? || ':' || key WHERE instr(key, ':') = 0", (namespace,))
        self.db.commit()

    def storage_key(self, key):
        return f"{self.namespace}:{key}" if self.namespace else key

    def close(self):
        self.db.close()

    def get(self, key):
        row = self.db.execute("SELECT data FROM sessions WHERE key = ?", (self.storage_key(key),)).fetchone()
        return json.loads(row[0]) if row else {}

    def save(self, key, record):
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO sessions VALUES (?, ?)", (self.storage_key(key), json.dumps(record)))

    async def run(self, upstream, key, prompt, effort, render, progress=None, stateless=False):
        record = self.get(key)
        sent, snapshot, fingerprint, mode = incremental_prompt(prompt, record)
        fingerprint = digest([fingerprint, effort])
        if record.get("fingerprint") == fingerprint and record.get("result"):
            return record["result"], {**record["audit"], "cached": True}
        pending = record.get("pending")
        superseded = None
        if stateless and not (pending and pending["fingerprint"] == fingerprint):
            # Side requests (classifiers, schema calls) rewrite their prompt every time, so
            # they can never append. Notion re-reads the whole thread per turn: send each one
            # on a fresh thread instead of resyncing onto an ever-growing one.
            record = {}
            sent, snapshot, _, mode = incremental_prompt(prompt, record, SIDE_MAX_SEND_CHARS or None)
            pending = None
        elif pending and pending["fingerprint"] != fingerprint:
            # A client may change context after a disconnect, timeout or failed
            # validation. It cannot reproduce the exact old fingerprint reliably.
            # Never append to that uncertain thread or deliver its stale tool calls.
            # Rebuild from the authoritative client history on a fresh thread.
            superseded = record
            record = {}
            sent, snapshot, _, mode = incremental_prompt(prompt, record)
            pending = None
        if not pending and THREAD_MAX_CHARS and record and mode == "append" and \
                record.get("thread_chars", 0) + len(sent) > THREAD_MAX_CHARS:
            # Restore the old binding if the fresh thread is rejected.
            superseded = record
            record = {}
            sent, snapshot, _, _ = incremental_prompt(prompt, record, ROLLOVER_SEED_CHARS)
            mode = "rollover"
        if not pending:
            if len(sent) > MAX_SEND_CHARS:
                # Nothing has been saved or dispatched yet, so the existing binding is untouched.
                raise UpstreamError(f"Turn payload {len(sent)} chars exceeds {MAX_SEND_CHARS}; "
                                    "compact the conversation", 413, rejected=True)
            continuation = bool(record)
            record = record or {"thread_id": str(uuid.uuid4()), "effort": effort}
            pending = {"fingerprint": fingerprint, "snapshot": snapshot, "mode": mode,
                       "input_chars": len(sent), "event_id": str(uuid.uuid4())}
            record["pending"] = pending
            # Commit BEFORE dispatch, so a disconnect/restart cannot create a second thread or turn.
            self.save(key, record)
            try:
                await upstream.start_turn(sent, effort, record["thread_id"], pending["event_id"], continuation)
            except UpstreamError as exc:
                if exc.rejected:
                    record.pop("pending")
                    if superseded is not None:
                        self.save(key, superseded)
                    elif continuation:
                        self.save(key, record)
                    else:
                        with self.db:
                            self.db.execute("DELETE FROM sessions WHERE key = ?", (self.storage_key(key),))
                raise
        if "answer" not in pending:
            text, cursor, sequence, turn_id = await upstream.wait_for_turn(
                record["thread_id"], record.get("cursor"), record.get("sequence", -1), record.get("turn_id"),
                **({"on_text": progress} if progress is not None else {}))
            pending.update(answer=text, cursor=cursor, sequence=sequence, turn_id=turn_id)
            record["thread_chars"] = record.get("thread_chars", 0) + pending["input_chars"] + len(text)
            self.save(key, record)
        # Keep raw answers private until schema validation succeeds. Retry the same
        # validation rather than generating a duplicate action if it failed previously.
        result = render(pending["answer"])
        output_atoms = delivered_atoms(result)
        pending["snapshot"]["assistant"] = [digest(a) for a in output_atoms]
        pending["snapshot"]["tool_ids"] = [{"id": a["content"]["id"], "name": a["content"]["name"]}
                                            for a in output_atoms if a["content"]["type"] == "tool_use"]
        info = {"client_input_chars": len(prompt),
                "truncated_tool_results": pending["snapshot"].get("truncated_tool_results", 0),
                "thread_id": record["thread_id"], "session_hash": key, "history_mode": pending["mode"],
                "upstream_input_chars": pending["input_chars"], "cached": False,
                "thread_effort": record["effort"], "thread_chars": record.get("thread_chars", 0)}
        record.update(snapshot=pending["snapshot"], fingerprint=fingerprint, result=result, audit=info,
                      cursor=pending["cursor"], sequence=pending["sequence"], turn_id=pending["turn_id"])
        record.pop("pending")
        self.save(key, record)
        return result, info
