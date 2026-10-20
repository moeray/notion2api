import asyncio
import json
import types
import unittest

import httpx
from starlette.requests import Request

from app.agent_gateway.routing import passthrough


class ProviderStream(httpx.AsyncByteStream):
    def __init__(self, chunks, error=None):
        self.chunks, self.error, self.closed = chunks, error, False

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk
        if self.error:
            raise self.error

    async def aclose(self):
        await asyncio.sleep(0)
        self.closed = True


class StreamFailureChecks(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.clients = []

    async def asyncTearDown(self):
        for client in self.clients:
            await client.aclose()

    async def route(self, stream, provider="claude", content_type="text/event-stream", status=200, headers=None):
        self.requests = []

        async def handler(req):
            self.requests.append(req)
            return httpx.Response(status, stream=stream, headers={"content-type": content_type, **(headers or {})})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        self.clients.append(client)
        app = types.SimpleNamespace(state=types.SimpleNamespace(proxy_http=client, routes={
            provider: {"base_url": "https://provider.example/api", "auth_token": "provider-key"}}))
        req = Request({"type": "http", "method": "POST", "path": "/test", "query_string": b"",
                       "headers": [(b"x-notion-gateway-key", b"local-key")], "app": app})
        return await passthrough(req, b'{"stream":true}', provider,
                                 "v1/messages" if provider == "claude" else "responses")

    async def body(self, response):
        return b"".join([chunk async for chunk in response.body_iterator])

    async def test_partial_event_is_discarded_and_disconnect_is_a_protocol_error(self):
        complete = b'event: content_block_delta\ndata: {"type":"content_block_delta"}\n\n'
        for provider in ("claude", "codex"):
            with self.subTest(provider=provider):
                stream = ProviderStream([complete, b'event: delta\ndata: {"text":"half'],
                                        httpx.RemoteProtocolError("private-upstream-details"))
                response = await self.route(stream, provider)
                body = await self.body(response)
                self.assertTrue(body.startswith(complete))
                tail = body[len(complete):]
                self.assertTrue(tail.startswith(b"event: error\ndata: "))
                event = json.loads(tail.split(b"data: ", 1)[1])
                self.assertEqual(event["type"], "error")
                if provider == "claude":
                    self.assertEqual(event["error"]["type"], "api_error")
                else:
                    self.assertEqual(event["code"], "server_error")
                self.assertNotIn(b"private-upstream-details", body)
                self.assertNotIn(b"half", body)
                self.assertNotIn(b"message_stop", body)
                self.assertNotIn(b"response.completed", body)
                self.assertTrue(stream.closed)
                self.assertEqual(len(self.requests), 1)

    async def test_read_errors_before_first_event_are_reported_and_closed(self):
        for error in (httpx.ReadTimeout("secret"), httpx.ReadError("secret"),
                      httpx.RemoteProtocolError("secret")):
            with self.subTest(error=type(error).__name__):
                stream = ProviderStream([], error)
                response = await self.route(stream)
                body = await self.body(response)
                self.assertTrue(body.startswith(b"event: error\n"))
                self.assertNotIn(b"secret", body)
                self.assertTrue(stream.closed)
                self.assertEqual(len(self.requests), 1)

    async def test_complete_sse_frames_survive_every_byte_boundary(self):
        for newline in (b"\n", b"\r\n", b"\r"):
            with self.subTest(newline=newline):
                body = newline.join([b": ping", b"", b'event: content_block_delta',
                                     b'data: {"type":"content_block_delta","text":"unchanged"}', b"", b""])
                stream = ProviderStream([body[i:i + 1] for i in range(len(body))])
                response = await self.route(stream)
                self.assertEqual(await self.body(response), body)
                self.assertTrue(stream.closed)

    async def test_complete_event_is_forwarded_before_upstream_finishes(self):
        complete = b'event: ping\ndata: {"type":"ping"}\n\n'
        stream = ProviderStream([complete], httpx.ReadTimeout("later"))
        response = await self.route(stream)
        self.assertEqual(await anext(response.body_iterator), complete)
        self.assertFalse(stream.closed)
        self.assertTrue((await self.body(response)).startswith(b"event: error\n"))
        self.assertTrue(stream.closed)

    async def test_terminal_event_finishes_without_waiting_for_broken_connection(self):
        for provider, terminal in (("claude", "message_stop"), ("claude", "error"),
                                   ("codex", "response.completed"), ("codex", "response.failed"),
                                   ("codex", "response.incomplete"), ("codex", "error")):
            with self.subTest(provider=provider, terminal=terminal):
                frame = f'event: {terminal}\ndata: {{"type":"{terminal}"}}\n\n'.encode()
                stream = ProviderStream([b": ping\n\n" + frame],
                                        AssertionError("Must not read past a terminal event"))
                response = await self.route(stream, provider)
                self.assertEqual(await self.body(response), b": ping\n\n" + frame)
                self.assertTrue(stream.closed)

    async def test_codex_done_marker_finishes_without_reading_more(self):
        stream = ProviderStream([b"data: [DONE]\n\n"], AssertionError("Must not read past DONE"))
        response = await self.route(stream, "codex")
        self.assertEqual(await self.body(response), b"data: [DONE]\n\n")
        self.assertTrue(stream.closed)

    async def test_gzip_sse_is_decoded_without_stale_encoding_header(self):
        import gzip

        body = b'event: message_stop\ndata: {"type":"message_stop"}\n\n'
        compressed = gzip.compress(body)
        stream = ProviderStream([compressed[:10], compressed[10:]])
        response = await self.route(stream, headers={"content-encoding": "gzip"})
        self.assertEqual(await self.body(response), body)
        self.assertNotIn("content-encoding", response.headers)
        self.assertTrue(stream.closed)

    async def test_unrecognized_sse_data_is_forwarded_without_interpretation(self):
        body = b'data: {"type":[]}\n\ndata: plain text\n\ndata: [1,2]\n\n'
        stream = ProviderStream([body])
        response = await self.route(stream)
        self.assertEqual(await self.body(response), body)
        self.assertTrue(stream.closed)

    async def test_unterminated_event_at_eof_is_an_error(self):
        stream = ProviderStream([b'data: {"type":"content_block_delta"}'])
        response = await self.route(stream)
        body = await self.body(response)
        self.assertTrue(body.startswith(b"event: error\n"))
        self.assertTrue(stream.closed)

    async def test_non_streaming_body_failure_returns_502_without_partial_json(self):
        stream = ProviderStream([b'{"partial":'], httpx.RemoteProtocolError("secret"))
        response = await self.route(stream, content_type="application/json")
        self.assertEqual(response.status_code, 502)
        self.assertNotIn(b"partial", response.body)
        self.assertNotIn(b"secret", response.body)
        self.assertTrue(stream.closed)
        self.assertEqual(len(self.requests), 1)

    async def test_complete_json_and_upstream_http_error_are_preserved(self):
        for status in (200, 429):
            with self.subTest(status=status):
                body = b'{"provider":"unchanged"}'
                stream = ProviderStream([body[:7], body[7:]])
                response = await self.route(stream, content_type="application/json", status=status)
                self.assertEqual(response.status_code, status)
                self.assertEqual(response.body, body)
                self.assertTrue(stream.closed)

    async def test_downstream_disconnect_closes_upstream(self):
        import anyio
        from starlette.responses import StreamingResponse

        waiting, disconnected = asyncio.Event(), asyncio.Event()

        class WaitingStream(ProviderStream):
            async def __aiter__(self):
                yield b'event: ping\ndata: {}\n\n'
                waiting.set()
                await asyncio.Event().wait()

        stream = WaitingStream([])
        response = await self.route(stream)
        self.assertIsInstance(response, StreamingResponse)
        sent = []

        async def receive():
            await waiting.wait()
            disconnected.set()
            return {"type": "http.disconnect"}

        async def send(message):
            sent.append(message)

        with anyio.fail_after(2):
            await response({"type": "http", "asgi": {"spec_version": "2.3"}}, receive, send)
        self.assertTrue(disconnected.is_set())
        self.assertTrue(stream.closed)
        self.assertFalse(any(b"event: error" in m.get("body", b"") for m in sent))

    async def test_real_incomplete_chunked_response_does_not_escape_asgi(self):
        from starlette.applications import Starlette
        from starlette.middleware.base import BaseHTTPMiddleware
        from starlette.routing import Route

        complete = b'event: ping\ndata: {"type":"ping"}\n\n'
        upstream_closed = asyncio.Event()
        request_count = 0

        async def broken_provider(reader, writer):
            nonlocal request_count
            request_count += 1
            try:
                await reader.readuntil(b"\r\n\r\n")
                writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                             b"Transfer-Encoding: chunked\r\n\r\n")
                writer.write(f"{len(complete):x}\r\n".encode() + complete + b"\r\n")
                # Advertise a larger chunk and close early, exactly as in the reported traceback.
                writer.write(b"100\r\ndata: {\"partial\":")
                await writer.drain()
            finally:
                writer.close()
                await writer.wait_closed()
                upstream_closed.set()

        async def endpoint(request):
            return await passthrough(request, await request.body(), "claude", "v1/messages")

        async def middleware(request, call_next):
            return await call_next(request)

        server = await asyncio.start_server(broken_provider, "127.0.0.1", 0)
        async with server, httpx.AsyncClient(trust_env=False) as upstream:
            port = server.sockets[0].getsockname()[1]
            app = Starlette(routes=[Route("/messages", endpoint, methods=["POST"])])
            app.add_middleware(BaseHTTPMiddleware, dispatch=middleware)
            app.state.proxy_http = upstream
            app.state.routes = {"claude": {"base_url": f"http://127.0.0.1:{port}", "auth_token": "test"}}
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app),
                                         base_url="http://testserver") as client:
                response = await asyncio.wait_for(client.post("/messages", json={"stream": True}), 3)
            await asyncio.wait_for(upstream_closed.wait(), 1)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.content.startswith(complete))
        self.assertEqual(response.content.count(b"event: error\n"), 1)
        self.assertNotIn(b"partial", response.content)
        self.assertEqual(request_count, 1)


class RoutingChecks(unittest.TestCase):
    def check_route(self, provider):
        async def run():
            observed = []

            class Chunks(httpx.AsyncByteStream):
                async def __aiter__(self):
                    yield b'event: original\n'
                    yield b'data: unchanged\n\n'

            async def handler(req):
                observed.append(req)
                return httpx.Response(200, stream=Chunks(), headers={"content-type": "text/event-stream"})

            client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            app = types.SimpleNamespace(state=types.SimpleNamespace(proxy_http=client, routes={
                "claude": {"base_url": "https://claude.example/api", "auth_token": "old-claude-key"},
                "codex": {"base_url": "https://codex.example/backend"}}))
            req = Request({"type": "http", "method": "POST", "path": "/test", "query_string": b"beta=true",
                           "headers": [(b"authorization", b"Bearer client-oauth"), (b"x-api-key", b"local-key"),
                                       (b"x-notion-gateway-key", b"local-routing-key"), (b"cookie", b"do-not-forward"),
                                       (b"host", b"localhost")], "app": app})
            response = await passthrough(req, b'{"model":"original"}', provider, "responses" if provider == "codex" else "v1/messages")
            result = b"".join([chunk async for chunk in response.body_iterator])
            self.assertEqual(result, b'event: original\ndata: unchanged\n\n')
            sent = observed[0]
            self.assertEqual(sent.content, b'{"model":"original"}')
            self.assertNotIn("x-notion-gateway-key", sent.headers)
            self.assertNotIn("cookie", sent.headers)
            self.assertEqual(sent.headers["authorization"], "Bearer old-claude-key" if provider == "claude" else "Bearer client-oauth")
            if provider == "claude":
                self.assertNotIn("x-api-key", sent.headers)
            await client.aclose()
        asyncio.run(run())

    def test_claude_uses_only_original_provider_credentials(self):
        self.check_route("claude")

    def test_codex_preserves_oauth_and_stream(self):
        self.check_route("codex")


if __name__ == "__main__":
    unittest.main()
