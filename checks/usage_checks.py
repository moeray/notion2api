import asyncio
import copy
import json
import tempfile
import unittest
import weakref
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
from fastapi import FastAPI

from app.agent_gateway import server
from app.agent_gateway.pool import WorkspacePool
from app.agent_gateway.protocol import prepare_request
from app.agent_gateway.sessions import SessionStore, client_session
from app.agent_gateway.upstream import UpstreamError
from app.agent_gateway.usage import (UsageReporter, context_usage, desktop_footer_enabled,
                                     footer_text, quota_snapshot)
from session_checks import ScopedNotion


QUOTA = {"status": "within_limit", "window": {"window": "6h", "used": 0, "limit": 100},
         "billingPeriodWindow": {"used": 38.13, "limit": 100, "periodEndMs": 1794920400000},
         "allowances": [{"bucketId": "standard", "cadence": "billing_period", "state": "resolved",
                         "enforcement": "enforced", "used": 38.13, "limit": 100,
                         "periodStartMs": 1792242000000, "periodEndMs": 1794920400000}]}
DESKTOP_TOOL = {"name": "mcp__ccd_turn__wrap_up", "input_schema": {"type": "object"}}


def upstream(scope="primary", quota=None):
    agent = ScopedNotion(scope)
    agent.space_id = "space-" + scope
    agent.post = AsyncMock(return_value=copy.deepcopy(QUOTA if quota is None else quota))
    return agent


class UsageFormatChecks(unittest.TestCase):
    def test_actual_windows_are_not_renamed_to_five_hours_or_thirty_days(self):
        value = quota_snapshot({**QUOTA, "private_debug": "SECRET"})
        text = footer_text(value, 12345)
        self.assertIn("6 小时", text)
        self.assertIn("38.13%", text)
        self.assertIn("61.87%", text)
        self.assertIn("11/17", text)
        self.assertNotIn("5 小时", text)
        self.assertNotIn("30 天", text)
        self.assertNotIn("SECRET", json.dumps(value))
        self.assertNotIn("resets_at", value["rolling"])
        self.assertIn("starts_at", value["billing_period"])

    def test_missing_invalid_and_over_limit_data_do_not_imply_unused_quota(self):
        for bad in ({}, [], {"window": {"used": float("nan"), "limit": 100}},
                    {"window": {"used": 0, "limit": True}},
                    {"window": {"used": -1, "limit": 100}},
                    {"window": {"used": 0, "limit": 0}}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                quota_snapshot(bad)
        value = quota_snapshot({"window": {"window": "5h", "used": 240, "limit": 200}})
        self.assertEqual(value["rolling"]["percent_used"], 120)
        self.assertEqual(value["rolling"]["percent_remaining"], 0)
        self.assertIn("5 小时", footer_text(value, 100))

    def test_preview_buckets_cannot_override_the_enforced_allowance(self):
        value = quota_snapshot({"allowances": [
            {"bucketId": "auto", "cadence": "rolling", "state": "resolved", "enforcement": "preview",
             "window": "5h", "used": 99, "limit": 100},
            {"bucketId": "standard", "cadence": "rolling", "state": "resolved", "enforcement": "enforced",
             "window": "6h", "used": 7, "limit": 100}]})
        self.assertEqual(value["rolling"]["percent_used"], 7)

    def test_context_capacity_is_unknown_and_input_is_explicitly_estimated(self):
        value = context_usage(9000)
        self.assertEqual(value["request_input_tokens_estimate"], 9000)
        self.assertIsNone(value["notion_context_used_tokens"])
        self.assertIsNone(value["notion_context_limit_tokens"])
        self.assertIsNone(value["notion_context_remaining_tokens"])
        text = footer_text(quota_snapshot(QUOTA), 9000)
        self.assertIn("文本估算，不含图片", text)
        self.assertIn("官方上限和剩余量未提供", text)

    def test_only_desktop_or_explicit_opt_in_and_unconstrained_answers_get_footer(self):
        options = {"schema": None, "stop_sequences": []}
        body = {"tools": [DESKTOP_TOOL]}
        with patch.dict("os.environ", {"NOTION_GATEWAY_DESKTOP_USAGE_FOOTER": "true"}):
            self.assertTrue(desktop_footer_enabled({}, body, options))
            self.assertTrue(desktop_footer_enabled({"x-notion-usage-display": "footer"}, {}, options))
            self.assertFalse(desktop_footer_enabled({}, {}, options))
            self.assertFalse(desktop_footer_enabled({"x-notion-usage-display": "off"}, body, options))
            self.assertFalse(desktop_footer_enabled({}, body, options, preserve_history=True))
            self.assertFalse(desktop_footer_enabled({}, body, {**options, "schema": {}}))
            self.assertFalse(desktop_footer_enabled({}, body, {**options, "stop_sequences": ["END"]}))


class UsageReadChecks(unittest.IsolatedAsyncioTestCase):
    async def test_concurrent_reads_are_coalesced_and_scoped_per_workspace(self):
        reporter = UsageReporter()
        primary, backup = upstream(), upstream("backup")
        await asyncio.gather(*(reporter.read(primary) for _ in range(10)))
        await reporter.read(backup)
        primary.post.assert_awaited_once_with("getCreditRateLimitStatus", {"spaceId": primary.space_id})
        backup.post.assert_awaited_once()

    async def test_failure_is_unavailable_or_labeled_stale_and_never_zero_filled(self):
        reporter = UsageReporter(ttl=0)
        agent = upstream()
        agent.post.side_effect = UpstreamError("PRIVATE", 429, exhausted=True)
        result = await reporter.read(agent)
        self.assertEqual(result["status"], "unavailable")
        self.assertIsNone(result["rolling"])
        self.assertNotIn("0%", footer_text(result, 20))
        self.assertNotIn("PRIVATE", json.dumps(result))
        agent.post.side_effect = None
        good = await reporter.read(agent)
        agent.post.side_effect = UpstreamError("PRIVATE", 401)
        stale = await reporter.read(agent)
        self.assertEqual(stale["status"], "stale")
        self.assertEqual(stale["observed_at"], good["observed_at"])
        self.assertEqual(stale["billing_period"]["percent_used"], 38.13)
        self.assertIn("刷新失败", footer_text(stale, 20))

    async def test_timeout_does_not_block_the_answer_or_touch_tools(self):
        reporter = UsageReporter(timeout=0.005)
        agent = upstream()

        async def slow(*args):
            await asyncio.sleep(1)
        agent.post.side_effect = slow
        answer = {"stop_reason": "end_turn", "content": [{"type": "text", "text": "OK"}],
                  "usage": {"input_tokens": 20, "output_tokens": 1}}
        decorated = await reporter.decorate(answer, agent)
        self.assertIn("额度暂不可读取", decorated["content"][0]["text"])
        for stop in ("tool_use", "max_tokens", "stop_sequence"):
            value = {**answer, "stop_reason": stop}
            self.assertIs(await reporter.decorate(value, agent), value)
        self.assertEqual(agent.post.await_count, 1)

    async def test_footer_is_cached_and_acknowledged_on_same_thread_after_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sessions.sqlite3"
            store = SessionStore(path)
            agent = upstream()
            reporter = UsageReporter()
            history = [{"role": "user", "content": "Remember violet."}]

            def prompt():
                return prepare_request({"model": "claude-opus-5-5", "max_tokens": 1000,
                                        "tools": [DESKTOP_TOOL], "messages": history})[0]

            def render(text):
                return {**json.loads(text), "stop_reason": "end_turn",
                        "usage": {"input_tokens": 100, "output_tokens": 1}}

            result, first = await store.run(agent, "s", prompt(), "medium", render, decorate=reporter.decorate)
            cached, info = await store.run(agent, "s", prompt(), "medium", render, decorate=reporter.decorate)
            self.assertTrue(info["cached"])
            self.assertEqual(result, cached)
            self.assertEqual(len(agent.writes), 1)
            self.assertEqual(result["content"][0]["text"].count("**Notion 用量**"), 1)
            store.close()
            store = SessionStore(path)
            try:
                history += [{"role": "assistant", "content": result["content"]},
                            {"role": "user", "content": "Continue."}]
                _, second = await store.run(agent, "s", prompt(), "medium", render, decorate=reporter.decorate)
                self.assertEqual(first["thread_id"], second["thread_id"])
                self.assertEqual(second["history_mode"], "append")
                self.assertNotIn("Notion 用量", agent.writes[-1][2])
                self.assertNotIn("pending", store.get("s"))
            finally:
                store.close()


class UsageEndpointChecks(unittest.IsolatedAsyncioTestCase):
    async def test_authenticated_endpoint_and_stream_use_the_bound_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SessionStore(Path(tmp) / "sessions.sqlite3", namespace="primary")
            a, b = upstream(), upstream("backup", {"window": {"window": "6h", "used": 73, "limit": 100}})
            pool = WorkspacePool([("default", a), ("space", b)])
            app = FastAPI()
            app.middleware("http")(server.local_auth)
            app.add_api_route("/v1/notion/usage", server.notion_usage, methods=["GET"])
            app.add_api_route("/v1/messages", server.messages, methods=["POST"])
            app.state.key = "test-local-key-not-a-real-credential"
            app.state.sessions, app.state.pool = store, pool
            app.state.usage_reporter = UsageReporter()
            app.state.locks = weakref.WeakValueDictionary()
            headers = {"x-api-key": app.state.key, "x-claude-code-session-id": "usage-check"}
            key, _ = client_session(headers, {}, "messages")
            store.save(key, {"scope": "backup", "result": {"usage": {"input_tokens": 321}}})
            try:
                with patch.object(server, "app", app), patch.object(server, "audit"):
                    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                                base_url="http://127.0.0.1") as client:
                        self.assertEqual((await client.get("/v1/notion/usage")).status_code, 401)
                        denied = await client.get("/v1/notion/usage", headers={**headers, "origin": "https://evil.invalid"})
                        self.assertEqual(denied.status_code, 403)
                        response = await client.get("/v1/notion/usage", headers=headers)
                        self.assertEqual(response.headers["cache-control"], "no-store")
                        data = response.json()
                        self.assertEqual(data["profile"], "space")
                        self.assertEqual(data["rolling"]["percent_used"], 73)
                        self.assertEqual(data["context"]["request_input_tokens_estimate"], 321)
                        a.post.assert_not_awaited()
                        store.save(key, {"scope": "removed-workspace"})
                        missing = (await client.get("/v1/notion/usage", headers=headers)).json()
                        self.assertEqual(missing["status"], "unavailable")
                        self.assertIsNone(missing["profile"])
                        self.assertIsNone(missing["rolling"])
                        a.post.assert_not_awaited()
                        # Force explicit exhaustion, then prove the successful
                        # workspace supplies the footer without a second switch.
                        store.save(key, {})
                        a.exhausted = True
                        response = await client.post("/v1/messages", headers=headers, json={
                            "model": "claude-opus-5-5", "max_tokens": 1000, "stream": True,
                            "tools": [DESKTOP_TOOL], "messages": [{"role": "user", "content": "Hello"}]})
                        self.assertIn("73%", response.text)
                        self.assertIn("message_stop", response.text)
                        self.assertNotIn("event: error", response.text)
                        self.assertEqual(store.get(key)["scope"], "backup")
                        self.assertEqual(len(b.writes), 1)
                        self.assertNotIn("pending", store.get(key))
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
