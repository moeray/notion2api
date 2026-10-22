"""Native image transport, history continuity, and upload failure boundaries."""
import base64
import json
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx

from app.agent_gateway.media import ImagePrompt, FilePrompt, Image, MAX_IMAGE_BYTES
from app.agent_gateway.protocol import prepare_request, parse_answer, ProtocolError
from app.agent_gateway.sessions import SessionStore, incremental_prompt, digest, delivered_atoms
from app.agent_gateway.upstream import NotionAgent, UpstreamError

PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=')
ANSWER = '{"content":[{"type":"text","text":"A red square"}]}'


def image_block(data=PNG):
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png",
            "data": base64.b64encode(data).decode()}}


def body(messages=None):
    return {"model": "claude-opus-5-5", "max_tokens": 128, "messages": messages or [
        {"role": "user", "content": [image_block(), {"type": "text", "text": "What is shown?"}]}]}


def render(text):
    content, stop = parse_answer(text, {}, {"type": "auto"})
    return {"content": content, "stop_reason": stop}


class ImageProtocolChecks(unittest.TestCase):
    def test_bytes_are_outside_history_and_fingerprints_distinguish_images(self):
        request = body()
        prompt = prepare_request(request)[0]
        self.assertIsInstance(prompt, ImagePrompt)
        self.assertNotIn(request["messages"][0]["content"][0]["source"]["data"], prompt)
        self.assertEqual(next(iter(prompt.images.values())).data, PNG)
        changed = body([{"role": "user", "content": [image_block(PNG + b"different")]}])
        self.assertNotEqual(incremental_prompt(prompt, {})[2],
                            incremental_prompt(prepare_request(changed)[0], {})[2])

    def test_invalid_images_fail_explicitly(self):
        for source in [None, {"type": "url", "url": "http://127.0.0.1/private"},
                       {"type": "base64", "media_type": "image/svg+xml", "data": "YWJj"},
                       {"type": "base64", "media_type": "image/png", "data": "not base64!"},
                       {"type": "base64", "media_type": "image/png", "data": "YWJj"},
                       {"type": "base64", "media_type": "image/png", "data": ""},
                       {"type": "base64", "media_type": "image/png", "data": "A" * (4 * ((MAX_IMAGE_BYTES + 2) // 3) + 1)}]:
            with self.subTest(source_type=type(source)), self.assertRaises(ProtocolError):
                prepare_request(body([{"role": "user", "content": [{"type": "image", "source": source}]}]))

    def test_image_bytes_do_not_consume_text_budget(self):
        prompt = prepare_request(body([{"role": "user", "content": [image_block(PNG + b"x" * 3_000_000)]}]))[0]
        self.assertLess(len(incremental_prompt(prompt, {})[0]), 5000)

    def test_repeated_images_deduplicate_and_latest_turn_limit_is_explicit(self):
        prompt = prepare_request(body([{"role": "user", "content": [image_block()] * 25}]))[0]
        self.assertEqual(len(incremental_prompt(prompt, {})[0].images), 1)
        prompt = prepare_request(body([{"role": "user", "content": [image_block(PNG + bytes([i])) for i in range(20)]}]))[0]
        with self.assertRaises(UpstreamError) as ctx:
            incremental_prompt(prompt, {})
        self.assertEqual(ctx.exception.status, 413)

    def test_long_tool_results_keep_image_order_and_tool_pair(self):
        messages = [{"role": "assistant", "content": [{"type": "tool_use", "id": "read1", "name": "Read", "input": {}}]},
                    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "read1", "content": [
                        {"type": "text", "text": "prefix" * 12000}, image_block(), {"type": "text", "text": "suffix"}]}]}]
        prompt = prepare_request(body(messages))[0]
        sent = incremental_prompt(prompt, {})[0]
        context = json.loads(sent.split("CLIENT_CONTEXT_JSON:\n")[1])
        result = context["messages"][-1]["content"][0]
        self.assertEqual(result["tool_use_id"], "read1")
        self.assertEqual([b["type"] for b in result["content"]], ["text", "image"])
        self.assertIn("truncated tool result", result["content"][0]["text"])
        self.assertEqual(len(sent.images), 1)

    def test_incremental_followup_omits_old_upload_and_failover_replays_it(self):
        request = body()
        prompt = prepare_request(request)[0]
        _, snapshot, _, _ = incremental_prompt(prompt, {})
        result = render(ANSWER)
        snapshot["assistant"] = [digest(a) for a in delivered_atoms(result)]
        request["messages"] += [{"role": "assistant", "content": result["content"]},
                                {"role": "user", "content": "What color?"}]
        followup = prepare_request(request)[0]
        sent, _, _, mode = incremental_prompt(followup, {"snapshot": snapshot})
        self.assertEqual(mode, "append")
        self.assertFalse(getattr(sent, "images", None))
        self.assertEqual(len(incremental_prompt(followup, {})[0].images), 1)

    def test_compaction_and_resync_do_not_split_image_from_its_question(self):
        request = body([{"role": "user", "content": [image_block(), {"type": "text", "text": "x" * 80000}]}])
        prompt = prepare_request(request)[0]
        with patch("app.agent_gateway.sessions.NEW_THREAD_MAX_CHARS", 8000):
            sent, snapshot, _, _ = incremental_prompt(prompt, {})
        self.assertEqual(len(sent.images), 1)
        self.assertIn('"type":"image"', sent)
        self.assertIn("Bridge omitted", sent)
        request["messages"][0]["content"][1]["text"] = "Revised question"
        sent, _, _, mode = incremental_prompt(prepare_request(request)[0], {"snapshot": snapshot})
        self.assertEqual(mode, "resync")
        self.assertEqual(len(sent.images), 1)
        self.assertIn("Revised question", sent)

    def test_omitted_old_image_is_not_uploaded(self):
        request = body([{"role": "user", "content": [image_block(), {"type": "text", "text": "old" * 50000}]},
                        {"role": "assistant", "content": "Acknowledged"},
                        {"role": "user", "content": "A new text-only question"}])
        with patch("app.agent_gateway.sessions.NEW_THREAD_MAX_CHARS", 6000):
            sent = incremental_prompt(prepare_request(request)[0], {})[0]
        self.assertFalse(getattr(sent, "images", None))
        self.assertIn("omitted_earlier_items", sent)

    def test_multiple_images_remain_associated_with_ordered_message_blocks(self):
        request = body([{"role": "user", "content": [image_block(), {"type": "text", "text": "first image"},
                        image_block(PNG + b"second"), {"type": "text", "text": "second image"}]}])
        sent = incremental_prompt(prepare_request(request)[0], {})[0]
        blocks = json.loads(sent.split("CLIENT_CONTEXT_JSON:\n")[1])["messages"][0]["content"]
        self.assertEqual(list(sent.images), [b["image_id"] for b in blocks if b["type"] == "image"])


class ImageUploadChecks(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = SessionStore(Path(self.tmp.name) / "sessions.sqlite3")
        self.agent = object.__new__(NotionAgent)
        self.agent.space_id = "space"
        self.agent.image_files = OrderedDict()
        self.agent.upload_image = AsyncMock(return_value="file-one")
        self.agent.post = AsyncMock(return_value={})
        self.agent.wait_for_turn = AsyncMock(return_value=(ANSWER, "cursor", 10, "turn"))

    async def asyncTearDown(self):
        self.store.close()
        self.tmp.cleanup()

    async def test_initial_native_attachment_and_cached_retry_without_image_persistence(self):
        prompt = prepare_request(body())[0]
        first, _ = await self.store.run(self.agent, "client", prompt, "medium", render)
        endpoint, submitted = self.agent.post.call_args.args
        self.assertEqual(endpoint, "createAgentThread")
        self.assertEqual(submitted["content"][1], {"type": "file", "file_id": "file-one"})
        self.assertEqual(self.agent.upload_image.call_args.args[2], {"type": "user"})
        persisted = json.dumps(self.store.get("client"))
        self.assertNotIn(base64.b64encode(PNG).decode(), persisted)
        self.assertNotIn("file-one", persisted)
        again, info = await self.store.run(self.agent, "client", prompt, "medium", render)
        self.assertEqual(first, again)
        self.assertTrue(info["cached"])
        self.agent.upload_image.assert_awaited_once()
        self.agent.post.assert_awaited_once()

    async def test_image_continuation_uses_existing_thread_and_new_file(self):
        request = body([{"role": "user", "content": "Hello"}])
        result, _ = await self.store.run(self.agent, "client", prepare_request(request)[0], "medium", render)
        thread = self.store.get("client")["thread_id"]
        request["messages"] += [{"role": "assistant", "content": result["content"]}, body()["messages"][0]]
        await self.store.run(self.agent, "client", prepare_request(request)[0], "medium", render)
        self.assertEqual(self.store.get("client")["thread_id"], thread)
        self.assertEqual(self.agent.upload_image.call_args.args[2], {"type": "thread", "threadId": thread})
        endpoint, submitted = self.agent.post.call_args.args
        self.assertEqual(endpoint, "sendEventToAgentThread")
        self.assertEqual(submitted["event"]["content"][1]["file_id"], "file-one")

    async def test_upload_failure_and_cancellation_do_not_leave_pending_generation(self):
        import asyncio
        for error in [UpstreamError("upload failed"), asyncio.CancelledError()]:
            self.agent.upload_image.side_effect = error
            with self.assertRaises(type(error)):
                await self.store.run(self.agent, "client", prepare_request(body())[0], "medium", render)
            self.assertEqual(self.store.get("client"), {})
        self.agent.post.assert_not_awaited()
        self.agent.wait_for_turn.assert_not_awaited()

    async def test_failed_upload_preserves_existing_binding_and_history(self):
        request = body([{"role": "user", "content": "hello"}])
        result, _ = await self.store.run(self.agent, "client", prepare_request(request)[0], "medium", render)
        before = self.store.get("client")
        request["messages"] += [{"role": "assistant", "content": result["content"]}, body()["messages"][0]]
        self.agent.upload_image.side_effect = UpstreamError("upload failed")
        with self.assertRaises(UpstreamError):
            await self.store.run(self.agent, "client", prepare_request(request)[0], "medium", render)
        self.assertEqual(self.store.get("client"), before)

    async def test_upload_cache_is_scoped_to_thread_and_safe_to_retry(self):
        prompt = prepare_request(body())[0]
        await self.agent.prepare_images(prompt, "thread1", False)
        await self.agent.prepare_images(prompt, "thread1", False)
        self.agent.upload_image.assert_awaited_once()
        await self.agent.prepare_images(prompt, "thread2", False)
        self.assertEqual(self.agent.upload_image.await_count, 2)

    async def test_real_upload_flow_uses_an_isolated_storage_client(self):
        requests = []
        def storage_response(request):
            requests.append(request)
            return httpx.Response(200)
        storage = httpx.AsyncClient(transport=httpx.MockTransport(storage_response))
        self.agent.post = AsyncMock(side_effect=[{
            "file": {"id": "file-one"}, "upload": {"type": "single_part", "method": "PUT",
                "url": "https://prod-files-secure.s3.us-west-2.amazonaws.com/image?signature=secret",
                "headers": {"Content-Length": str(len(PNG))}}}, {"id": "file-one"}])
        with patch("app.agent_gateway.upstream.httpx.AsyncClient", return_value=storage):
            file_id = await NotionAgent.upload_image(self.agent, "a" * 64, Image("image/png", PNG), {"type": "user"})
        self.assertEqual(file_id, "file-one")
        self.assertEqual(requests[0].content, PNG)
        self.assertNotIn("cookie", requests[0].headers)
        self.assertNotIn("authorization", requests[0].headers)
        self.assertEqual(self.agent.post.call_args.args[0], "completeAgentServiceFileUpload")
        self.assertNotIn("parts", self.agent.post.call_args.args[1])

    async def test_untrusted_upload_destination_and_headers_are_rejected(self):
        client = AsyncMock()
        for url in ["http://prod-files-secure.s3.us-west-2.amazonaws.com/file", "https://127.0.0.1/file",
                    "https://prod-files-secure.s3.us-west-2.amazonaws.com.evil.test/file",
                    "https://user:secret@prod-files-secure.s3.us-west-2.amazonaws.com/file"]:
            with self.assertRaises(ValueError):
                await NotionAgent.upload_part(client, {"url": url, "method": "PUT"}, PNG)
        with self.assertRaises(ValueError):
            await NotionAgent.upload_part(client, {"url": "https://prod-files-secure.s3.us-west-2.amazonaws.com/file",
                "method": "PUT", "headers": {"Cookie": "private"}}, PNG)
        client.put.assert_not_awaited()

    async def test_storage_error_does_not_expose_signed_urls_or_body(self):
        client = AsyncMock()
        client.put.return_value = httpx.Response(403, text="private upload response")
        with self.assertRaises(UpstreamError) as ctx:
            await NotionAgent.upload_part(client, {"url": "https://prod-files-secure.s3.us-west-2.amazonaws.com/file?secret=token",
                "method": "PUT", "headers": {}}, PNG)
        self.assertNotIn("private", str(ctx.exception))
        self.assertNotIn("token", str(ctx.exception))
        self.assertTrue(ctx.exception.rejected)

    async def test_multipart_upload_preserves_bytes_order_and_etags(self):
        requests = []
        def respond(request):
            requests.append(request)
            return httpx.Response(200, headers={"etag": '"part-' + str(len(requests)) + '"'})
        storage = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        upload = {"type": "multi_part", "part_size_bytes": 40, "parts": [
            {"part_number": i, "method": "PUT", "url": f"https://prod-files-secure.s3.us-west-2.amazonaws.com/part{i}"}
            for i in range(1, (len(PNG) + 39) // 40 + 1)]}
        self.agent.post = AsyncMock(side_effect=[{"file": {"id": "file-one"}, "upload": upload}, {"id": "file-one"}])
        with patch("app.agent_gateway.upstream.httpx.AsyncClient", return_value=storage):
            await NotionAgent.upload_image(self.agent, "a" * 64, Image("image/png", PNG), {"type": "user"})
        self.assertEqual(b"".join(r.content for r in requests), PNG)
        parts = self.agent.post.call_args.args[1]["parts"]
        self.assertEqual(parts, [{"partNumber": i, "etag": f'"part-{i}"'} for i in range(1, len(requests) + 1)])
