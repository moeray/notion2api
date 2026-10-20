import json
import tempfile
import unittest
from pathlib import Path

from app.agent_gateway.protocol import prepare_request
from app.agent_gateway.sessions import SessionStore, client_session, incremental_prompt, is_away_summary
from app.agent_gateway.upstream import Transcript, UpstreamError
from app.agent_gateway.pool import WorkspacePool


def prompt(messages, **extra):
    return prepare_request({"model": "claude-opus-5-5", "max_tokens": 1000,
                            "messages": messages, **extra})[0]


class FakeNotion:
    def __init__(self):
        self.writes = []
        self.polls = []
        self.timeout = False
        self.reject = False

    async def start_turn(self, text, effort, thread, event, continuation):
        if self.reject:
            raise UpstreamError("rejected", rejected=True)
        self.writes.append((thread, continuation, text))

    async def wait_for_turn(self, thread, cursor, sequence, turn_id):
        self.polls.append((thread, cursor, sequence, turn_id))
        if self.timeout:
            raise TimeoutError()
        n = len(self.writes)
        return '{"content":[{"type":"text","text":"OK"}]}', f"cursor{n}", n * 10, f"turn{n}"


class ScopedNotion(FakeNotion):
    def __init__(self, scope):
        super().__init__()
        self.account_scope = scope
        self.exhausted = False
        self.wait_error = None

    async def start_turn(self, text, effort, thread, event, continuation):
        if self.exhausted:
            raise UpstreamError("quota", 429, rejected=True, exhausted=True)
        await super().start_turn(text, effort, thread, event, continuation)

    async def wait_for_turn(self, thread, cursor, sequence, turn_id):
        if self.wait_error is not None:
            raise self.wait_error
        return await super().wait_for_turn(thread, cursor, sequence, turn_id)


class SessionChecks(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "sessions.db"
        self.store = SessionStore(self.path)
        self.up = FakeNotion()
        self.first = [{"role": "user", "content": "Remember violet."}]

    async def asyncTearDown(self):
        self.store.close()
        self.tmp.cleanup()

    async def run_turn(self, key="one", messages=None):
        return await self.store.run(self.up, key, prompt(messages or self.first), "medium", json.loads)

    async def test_side_request_uses_fresh_thread_per_request(self):
        def run(messages):
            return self.store.run(self.up, "side", prompt(messages), "medium", json.loads, stateless=True)
        _, first = await run(self.first)
        _, again = await run(self.first)
        messages = self.first + [{"role": "assistant", "content": "OK"}, {"role": "user", "content": "Next?"}]
        _, second = await run(messages)
        self.assertTrue(again["cached"])
        self.assertNotEqual(first["thread_id"], second["thread_id"])
        self.assertEqual([w[1] for w in self.up.writes], [False, False])
        self.assertIn("Remember violet.", self.up.writes[1][2])

    async def test_exhausted_workspace_fails_over_and_stays(self):
        a, b = ScopedNotion("a"), ScopedNotion("b")
        pool = WorkspacePool([("a", a), ("b", b)])

        def run(messages):
            return pool.run(self.store, "one", prompt(messages), "medium", json.loads)
        _, first = await run(self.first)
        a.exhausted = True
        messages = self.first + [{"role": "assistant", "content": "OK"}, {"role": "user", "content": "What color?"}]
        _, second = await run(messages)
        self.assertEqual((first["profile"], second["profile"]), ("a", "b"))
        self.assertEqual(second["history_mode"], "failover")
        self.assertNotEqual(first["thread_id"], second["thread_id"])
        self.assertEqual(b.writes[0][1], False)
        self.assertIn("Remember violet.", b.writes[0][2])
        a.exhausted = False
        pool.cooling.clear()
        _, third = await run(messages + [{"role": "assistant", "content": "OK"},
                                         {"role": "user", "content": "And now?"}])
        self.assertEqual(third["profile"], "b")
        self.assertEqual(third["thread_id"], second["thread_id"])
        self.assertEqual(b.writes[-1][1], True)

    async def test_no_progress_keeps_workspace_and_retry_polls_same_turn(self):
        for error in (TimeoutError(), UpstreamError("Notion turn made no progress", 504)):
            with self.subTest(error=type(error).__name__):
                a, b = ScopedNotion("a"), ScopedNotion("b")
                pool = WorkspacePool([("a", a), ("b", b)])
                key = type(error).__name__

                def run(messages):
                    return pool.run(self.store, key, prompt(messages), "medium", json.loads)

                _, first = await run(self.first)
                messages = self.first + [{"role": "assistant", "content": "OK"},
                                         {"role": "user", "content": "What color?"}]
                a.wait_error = error
                with self.assertRaises(type(error)):
                    await run(messages)
                pending = self.store.get(key)
                self.assertEqual(pending["scope"], "a")
                self.assertEqual(pending["thread_id"], first["thread_id"])
                self.assertIn("pending", pending)
                self.assertEqual(pool.cooling, {})
                self.assertEqual(b.writes, [])

                a.wait_error = None
                _, retried = await run(messages)
                self.assertEqual(retried["profile"], "a")
                self.assertEqual(retried["thread_id"], first["thread_id"])
                self.assertEqual(retried["history_mode"], "append")
                self.assertEqual(len(a.writes), 2)
                self.assertEqual(b.writes, [])

    def test_transcript_activity_ignores_own_message_and_quota_banner(self):
        state = Transcript(5)
        state.update({"patches": [
            {"op": "put", "entity": {"id": "u", "kind": "user_message", "sequence": 6}},
            {"op": "put", "entity": {"id": "rate-limit-status:s", "kind": "status", "sequence": 7}}]})
        self.assertFalse(state.active)
        state.update({"patches": [{"op": "put", "entity": {"id": "t", "kind": "thinking", "sequence": 8}}]})
        self.assertTrue(state.active)

    async def test_new_thread_input_is_capped(self):
        from app.agent_gateway.sessions import wire_size
        from unittest.mock import patch
        notion = FakeNotion()
        huge = [{"role": "user", "content": "START-MARK " + "x" * 300000 + " END-MARK"}]
        with patch("app.agent_gateway.sessions.NEW_THREAD_MAX_CHARS", 60000):
            await self.store.run(notion, "big", prompt(huge), "medium", json.loads, stateless=True)
        sent = notion.writes[0][2]
        self.assertLessEqual(wire_size(sent), 62000)
        self.assertIn("START-MARK", sent)
        self.assertIn("END-MARK", sent)
        self.assertIn("Bridge omitted", sent)

    async def test_all_workspaces_exhausted_returns_429(self):
        a, b = ScopedNotion("a"), ScopedNotion("b")
        a.exhausted = b.exhausted = True
        pool = WorkspacePool([("a", a), ("b", b)])
        with self.assertRaises(UpstreamError) as ctx:
            await pool.run(self.store, "one", prompt(self.first), "medium", json.loads)
        self.assertEqual(ctx.exception.status, 429)
        self.assertEqual(set(pool.cooling), {"a", "b"})

    async def test_resume_after_restart_and_new_session_isolation(self):
        _, first = await self.run_turn()
        self.store.close()
        self.store = SessionStore(self.path)
        messages = self.first + [{"role": "assistant", "content": "OK"}, {"role": "user", "content": "What color?"}]
        _, second = await self.run_turn(messages=messages)
        _, other = await self.run_turn(key="two")
        self.assertEqual(first["thread_id"], second["thread_id"])
        self.assertNotEqual(first["thread_id"], other["thread_id"])
        self.assertEqual([w[1] for w in self.up.writes], [False, True, False])
        self.assertNotIn("Remember violet", self.up.writes[1][2])
        self.assertIn("What color?", self.up.writes[1][2])
        self.assertEqual(self.up.polls[1][1:], ("cursor1", 10, "turn1"))
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)

    async def test_identical_retry_returns_cached_response_without_new_turn(self):
        a, _ = await self.run_turn()
        b, info = await self.run_turn()
        self.assertEqual(a, b)
        self.assertTrue(info["cached"])
        self.assertEqual(len(self.up.writes), 1)

    async def test_timeout_then_restart_polls_original_turn_without_resubmission(self):
        self.up.timeout = True
        with self.assertRaises(TimeoutError):
            await self.run_turn()
        self.store.close()
        self.store = SessionStore(self.path)
        self.up.timeout = False
        await self.run_turn()
        self.assertEqual(len(self.up.writes), 1)
        self.assertEqual(len(self.up.polls), 2)

    async def test_pending_different_request_drains_and_continues_same_thread(self):
        self.up.timeout = True
        with self.assertRaises(TimeoutError):
            await self.run_turn()
        old_thread = self.up.writes[0][0]
        self.store.close()
        self.store = SessionStore(self.path)
        self.up.timeout = False
        messages = self.first + [{"role": "user", "content": "different"}]
        _, info = await self.run_turn(messages=messages)
        self.assertEqual(info["thread_id"], old_thread)
        self.assertEqual(info["history_mode"], "append")
        self.assertTrue(info["recovered_pending"])
        self.assertTrue(self.up.writes[-1][1])
        self.assertNotIn("Remember violet", self.up.writes[-1][2])
        self.assertIn("different", self.up.writes[-1][2])
        self.assertIn('"previous_response_not_delivered":true', self.up.writes[-1][2])
        self.assertEqual(self.up.polls[-1][1:], ("cursor1", 10, "turn1"))
        await self.run_turn(messages=messages)
        self.assertEqual(len(self.up.writes), 2)

    async def test_changed_settings_can_recover_failed_validation(self):
        def invalid(text):
            raise ValueError("invalid answer")
        with self.assertRaises(ValueError):
            await self.store.run(self.up, "one", prompt(self.first), "medium", invalid)
        old = self.store.get("one")
        self.assertIn("answer", old["pending"])
        _, info = await self.store.run(self.up, "one", prompt(self.first, system="Updated context"),
                                       "medium", json.loads)
        self.assertEqual(info["thread_id"], old["thread_id"])
        self.assertEqual(len(self.up.writes), 2)
        self.assertEqual(len(self.up.polls), 2)

    async def test_rejected_recovery_preserves_completed_cursor_on_same_thread(self):
        self.up.timeout = True
        with self.assertRaises(TimeoutError):
            await self.run_turn()
        original = self.store.get("one")
        self.up.timeout = False
        self.up.reject = True
        messages = self.first + [{"role": "user", "content": "different"}]
        with self.assertRaises(UpstreamError):
            await self.run_turn(messages=messages)
        checkpoint = self.store.get("one")
        self.assertEqual(checkpoint["thread_id"], original["thread_id"])
        self.assertNotIn("pending", checkpoint)
        self.assertEqual(checkpoint["sequence"], 10)
        self.assertTrue(checkpoint["discarded_turn"])
        self.up.reject = False
        _, info = await self.run_turn(messages=messages)
        self.assertEqual(info["thread_id"], original["thread_id"])
        self.assertEqual(len(self.up.writes), 2)

    async def test_changed_request_during_unfinished_turn_keeps_binding_without_new_send(self):
        self.up.timeout = True
        with self.assertRaises(TimeoutError):
            await self.run_turn()
        messages = [{"role": "user", "content": "different"}]
        original = self.store.get("one")
        with self.assertRaises(TimeoutError):
            await self.run_turn(messages=messages)
        self.assertEqual(self.store.get("one"), original)
        self.assertEqual(len(self.up.writes), 1)
        self.up.timeout = False
        _, info = await self.run_turn(messages=messages)
        self.assertEqual(info["thread_id"], original["thread_id"])
        self.assertEqual(len(self.up.writes), 2)

    async def test_background_recap_keeps_main_snapshot_and_uses_same_thread_after_restart(self):
        result, first = await self.run_turn()
        main = self.first + [{"role": "assistant", "content": result["content"]}]
        recap = main + [{"role": "user", "content":
            "The user stepped away and is coming back. Recap in under 40 words, "
            "1-2 plain sentences, no markdown. Lead with the overall goal and current task."}]
        self.assertTrue(is_away_summary({"messages": recap}))
        self.assertTrue(is_away_summary({"messages": recap + [
            {"role": "system", "content": "<total_tokens>123 tokens left</total_tokens>"}]}))
        self.assertFalse(is_away_summary({"messages": recap + [{"role": "user", "content": "continue"}]}))
        original = self.store.get("one")["snapshot"]
        _, summary = await self.store.run(self.up, "one", prompt(recap), "medium", json.loads,
                                          preserve_history=True)
        self.assertEqual(first["thread_id"], summary["thread_id"])
        self.assertEqual(self.store.get("one")["snapshot"], original)
        self.store.close()
        self.store = SessionStore(self.path)
        _, cached = await self.store.run(self.up, "one", prompt(recap), "medium", json.loads,
                                         preserve_history=True)
        self.assertTrue(cached["cached"])
        self.assertEqual(len(self.up.writes), 2)
        _, followup = await self.run_turn(messages=main + [{"role": "user", "content": "What color?"}])
        self.assertEqual(followup["thread_id"], first["thread_id"])
        self.assertEqual(followup["history_mode"], "append")
        self.assertNotIn("Remember violet", self.up.writes[-1][2])
        self.assertNotIn("Recap in under", self.up.writes[-1][2])
        self.assertEqual([w[1] for w in self.up.writes], [False, True, True])

    async def test_interrupted_recap_does_not_replace_main_snapshot_or_deliver_stale_tools(self):
        result, first = await self.run_turn()
        main = self.first + [{"role": "assistant", "content": result["content"]}]
        self.up.timeout = True
        with self.assertRaises(TimeoutError):
            await self.store.run(self.up, "one",
                                 prompt(main + [{"role": "user", "content": "recap"}], max_tokens=64),
                                 "medium", json.loads, preserve_history=True)
        self.up.timeout = False
        rendered = []
        def render(text):
            rendered.append(text)
            return json.loads(text)
        _, next_turn = await self.store.run(self.up, "one",
            prompt(main + [{"role": "user", "content": "What color?"}]), "medium", render)
        self.assertEqual(next_turn["thread_id"], first["thread_id"])
        self.assertEqual(next_turn["history_mode"], "append")
        self.assertEqual(len(rendered), 1)
        self.assertNotIn("Remember violet", self.up.writes[-1][2])
        self.assertEqual(self.up.polls[-1][2], 20)
        self.assertIn('"max_output_tokens":1000', self.up.writes[-1][2])

    async def test_main_restores_settings_changed_by_background_recap(self):
        result, _ = await self.run_turn()
        main = self.first + [{"role": "assistant", "content": result["content"]}]
        await self.store.run(self.up, "one",
            prompt(main + [{"role": "user", "content": "recap"}],
                   max_tokens=64, tool_choice={"type": "none"}),
            "medium", json.loads, preserve_history=True)
        await self.run_turn(messages=main + [{"role": "user", "content": "continue"}])
        payload = json.loads(self.up.writes[-1][2].split("CLIENT_CONTEXT_JSON:\n", 1)[1])
        self.assertEqual(payload["max_output_tokens"], 1000)
        self.assertEqual(payload["tool_choice"], {"type": "auto"})
        self.assertEqual(payload["mode"], "append")

    async def test_failed_terminal_turn_can_recover_but_unknown_failure_cannot(self):
        up = ScopedNotion("a")
        up.wait_error = UpstreamError("still pending", 504)
        with self.assertRaises(UpstreamError):
            await self.store.run(up, "one", prompt(self.first), "medium", json.loads)
        original = self.store.get("one")
        messages = self.first + [{"role": "user", "content": "continue"}]
        with self.assertRaises(UpstreamError):
            await self.store.run(up, "one", prompt(messages), "medium", json.loads)
        self.assertEqual(self.store.get("one"), original)
        error = UpstreamError("terminal failure")
        error.completed_turn = {"cursor": "failed-cursor", "sequence": 20, "turn_id": "failed"}
        original_wait = up.wait_for_turn
        async def complete_once(*args):
            if up.wait_error:
                up.wait_error = None
                raise error
            return await original_wait(*args)
        up.wait_for_turn = complete_once
        _, info = await self.store.run(up, "one", prompt(messages), "medium", json.loads)
        self.assertEqual(info["thread_id"], original["thread_id"])
        self.assertEqual(up.polls[-1][1:], ("failed-cursor", 20, "failed"))

    async def test_definitive_rejection_does_not_leave_pending_binding(self):
        self.up.reject = True
        with self.assertRaises(UpstreamError):
            await self.run_turn()
        self.assertEqual(self.store.get("one"), {})

    async def test_compaction_resyncs_existing_thread(self):
        _, first = await self.run_turn()
        _, second = await self.run_turn(messages=[{"role": "user", "content": "Compacted summary. Next task."}])
        self.assertEqual(first["thread_id"], second["thread_id"])
        self.assertEqual(second["history_mode"], "resync")
        self.assertIn('"mode":"resync"', self.up.writes[1][2])

    def test_headers_metadata_fallback_and_agent_isolation(self):
        a, source = client_session({"X-Claude-Code-Session-Id": "a"}, {}, "messages")
        b, _ = client_session({}, {"metadata": {"user_id": json.dumps({"session_id": "a"})}}, "messages")
        self.assertEqual(a, b)
        self.assertEqual(source, "claude-header")
        child, _ = client_session({"x-claude-code-session-id": "a", "x-claude-code-agent-id": "child"}, {}, "messages")
        self.assertNotEqual(a, child)
        c, _ = client_session({}, {"client_metadata": {"thread_id": "c", "session_id": "ephemeral"}}, "responses")
        d, _ = client_session({}, {"client_metadata": {"thread_id": "c", "session_id": "changed"}}, "responses")
        self.assertEqual(c, d)
        self.assertNotEqual(client_session({}, {}, "messages")[0], client_session({}, {}, "messages")[0])

    def test_delta_handles_merged_messages_tool_ids_and_changed_system(self):
        tool = {"name": "Read", "input_schema": {"type": "object"}}
        first = [{"role": "user", "content": "Read file"}, {"role": "system", "content": "budget 1"}]
        _, snapshot, _, _ = incremental_prompt(prompt(first, tools=[tool]), {})
        next_messages = [first[0], {"role": "assistant", "content": [
            {"type": "text", "text": "Reading"}, {"type": "tool_use", "id": "local-id", "name": "Read", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "local-id", "content": "contents"}]},
            {"role": "system", "content": "budget 2"}]
        delta, _, _, mode = incremental_prompt(prompt(next_messages, tools=[tool]), {"snapshot": snapshot})
        context = json.loads(delta.split("CLIENT_CONTEXT_JSON:\n", 1)[1])
        self.assertEqual(mode, "append")
        self.assertNotIn("tools", context)
        self.assertEqual(context["messages"][0]["role"], "assistant")
        self.assertEqual(context["messages"][1]["content"][0]["tool_use_id"], "local-id")
        self.assertIn("budget 2", delta)

    def test_stale_completion_cannot_complete_new_turn(self):
        t = Transcript(after_sequence=10, previous_turn_id="old")
        t.update({"patches": [
            {"op": "put", "entity": {"kind": "turn_completed", "id": "old-end", "sequence": 10,
                                      "turn_id": "old", "stop_reason": "end_turn"}}]})
        self.assertIsNone(t.result())
        t.update({"patches": [
            {"op": "put", "entity": {"kind": "assistant_message", "id": "new-text", "sequence": 11,
                                      "turn_id": "new", "content": [{"type": "text", "text": "NEW"}]}},
            {"op": "put", "entity": {"kind": "turn_completed", "id": "new-end", "sequence": 12,
                                      "turn_id": "new", "stop_reason": "end_turn"}}]})
        self.assertEqual(t.result(), "NEW")
