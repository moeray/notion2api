"""Fixed-destination pass-through for the user's existing providers."""
import json
import logging
import re

import anyio
import httpx
from fastapi.responses import Response, StreamingResponse

logger = logging.getLogger(__name__)

HOP_HEADERS = {"host", "connection", "content-length", "transfer-encoding", "keep-alive",
               "proxy-authenticate", "proxy-authorization", "te", "trailer", "upgrade",
               "x-notion-gateway-key"}
# Two SSE line endings, without mistaking a single CRLF for an empty line.
SSE_BOUNDARY = re.compile(rb"(?:\r\n|\r(?!\n)|(?<!\r)\n){2}")
INTERRUPTED = "Existing provider connection was interrupted; please retry the request."


def stream_error(provider):
    if provider == "claude":
        event = {"type": "error", "error": {"type": "api_error", "message": INTERRUPTED}}
    else:
        event = {"type": "error", "code": "server_error", "message": INTERRUPTED, "param": None}
    return ("event: error\ndata: " + json.dumps(event) + "\n\n").encode()


def terminal_event(frame, provider):
    data = b"\n".join(line[5:].removeprefix(b" ") for line in frame.splitlines()
                      if line.startswith(b"data:"))
    if provider == "codex" and data == b"[DONE]":
        return True
    try:
        event = json.loads(data)
    except (ValueError, UnicodeError):
        return False
    terminal = ({"message_stop", "error"} if provider == "claude"
                else {"response.completed", "response.failed", "response.incomplete", "error"})
    kind = event.get("type") if isinstance(event, dict) else None
    return isinstance(kind, str) and kind in terminal


async def passthrough(request, body, provider, path):
    route = request.app.state.routes.get(provider)
    if not route:
        return Response("Original provider route is not configured", status_code=503)
    # Destinations come from the preserved local configuration, never request input.
    url = route["base_url"].rstrip("/") + "/" + path.lstrip("/")
    headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_HEADERS}
    headers.pop("cookie", None)
    if provider == "claude":
        headers.pop("authorization", None)
        headers.pop("x-api-key", None)
        headers["authorization"] = "Bearer " + route["auth_token"]
    elif request.headers.get("x-notion-gateway-key") is None:
        return Response("Codex routing key is required", status_code=401)
    headers["accept-encoding"] = "identity"
    client = request.app.state.proxy_http
    outgoing = client.build_request(request.method, url, content=body, headers=headers,
                                    params=list(request.query_params.multi_items()))
    try:
        response = await client.send(outgoing, stream=True)
    except httpx.HTTPError:
        return Response("Existing provider is unreachable", status_code=502)
    response_headers = {k: v for k, v in response.headers.items() if k.lower() not in HOP_HEADERS}

    def log_failure(exc):
        # Provider exception strings can include URLs/credentials; log only the class.
        logger.warning("Provider response interrupted: provider=%s path=%s error=%s",
                       provider, path, type(exc).__name__)

    async def close():
        # Starlette cancels the streaming task on client disconnect. Still release the
        # upstream socket under that cancellation scope, including awaitable cleanup.
        with anyio.CancelScope(shield=True):
            try:
                await response.aclose()
            except httpx.HTTPError as exc:
                log_failure(exc)

    is_sse = response.headers.get("content-type", "").split(";", 1)[0].strip().lower() == "text/event-stream"
    if not is_sse:
        # JSON/error bodies must finish before committing the downstream status.
        try:
            content = b"".join([chunk async for chunk in response.aiter_raw()])
            return Response(content, status_code=response.status_code, headers=response_headers)
        except httpx.HTTPError as exc:
            log_failure(exc)
            return Response(INTERRUPTED, status_code=502)
        finally:
            await close()

    # aiter_bytes also handles providers that ignore Accept-Encoding: identity.
    response_headers.pop("content-encoding", None)

    async def stream():
        pending = bytearray()
        try:
            async for chunk in response.aiter_bytes():
                scan_from = max(0, len(pending) - 3)
                pending.extend(chunk)
                # Forward complete frames immediately. A partial frame must never be
                # combined with the error event if the provider disconnects mid-JSON.
                end = 0
                for match in SSE_BOUNDARY.finditer(pending, scan_from):
                    frame = bytes(pending[end:match.end()])
                    end = match.end()
                    yield frame
                    # The protocol is complete even if the provider subsequently
                    # hangs or omits the HTTP chunked-transfer terminator.
                    if terminal_event(frame, provider):
                        return
                if end:
                    del pending[:end]
            if pending.strip():
                raise httpx.RemoteProtocolError("Provider ended with an incomplete SSE event")
            if pending:
                yield bytes(pending)
        except httpx.HTTPError as exc:
            log_failure(exc)
            # Headers are already sent: signal failure in the client's protocol.
            # Never replay a generation request or synthesize a success terminator.
            yield stream_error(provider)
        finally:
            await close()

    return StreamingResponse(stream(), status_code=response.status_code, headers=response_headers)
