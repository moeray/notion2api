"""Desktop regressions: reused turn IDs, bounded schema repair and safe retry."""
import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx

from app.agent_gateway.protocol import prepare_request, parse_answer, AnswerError
from app.agent_gateway.sessions import SessionStore
from app.agent_gateway.upstream import NotionAgent, Transcript, UpstreamError


PLAN = {'name': 'update_plan', 'input_schema': {'type': 'object', 'properties': {
    'items': {'type': 'array', 'items': {'type': 'object', 'properties': {
        'content': {'type': 'string'}, 'status': {'enum': ['pending', 'completed']}},
        'required': ['content', 'status'], 'additionalProperties': False}}}, 'required': ['items']}}
INVALID = json.dumps({'content': [{'type': 'text', 'text': 'DO NOT DELIVER'},
    {'type': 'tool_use', 'name': 'update_plan', 'input': {'items': [{'title': 'bad'}]}}]})
VALID = json.dumps({'content': [{'type': 'tool_use', 'name': 'update_plan',
    'input': {'items': [{'content': 'Check', 'status': 'pending'}]}}]})
TEXT = '{"content":[{"type":"text","text":"OK"}]}'


def render(text):
    blocks, stop = parse_answer(text, {'update_plan': PLAN}, {'type': 'auto'})
    return {'content': blocks, 'stop_reason': stop}


class ScriptedNotion:
    def __init__(self, answers):
        self.answers, self.writes, self.polls = answers, [], []
        self.wait_error = None

    async def start_turn(self, text, effort, thread, event, continuation):
        self.writes.append((text, thread, event, continuation))

    async def wait_for_turn(self, thread, cursor, sequence, turn_id):
        self.polls.append((thread, cursor, sequence, turn_id))
        if self.wait_error:
            raise self.wait_error
        n = len(self.writes)
        return self.answers[min(n - 1, len(self.answers) - 1)], f'cursor{n}', n * 10, f'turn{n}'


class RecoveryChecks(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / 'sessions.sqlite3'
        self.store = SessionStore(self.path)
        self.prompt = prepare_request({'model': 'claude-opus-5-5', 'max_tokens': 100,
            'tools': [PLAN], 'messages': [{'role': 'user', 'content': 'Plan the check'}]})[0]

    async def asyncTearDown(self):
        self.store.close()
        self.tmp.cleanup()

    async def run_turn(self, upstream, prompt=None):
        return await self.store.run(upstream, 'desktop', prompt or self.prompt, 'medium', render)

    async def test_schema_error_corrects_on_same_thread_and_caches_only_valid_tool_ids(self):
        up = ScriptedNotion([INVALID, VALID])
        result, info = await self.run_turn(up)
        self.assertEqual(info['repair_attempts'], 1)
        self.assertEqual(len({w[1] for w in up.writes}), 1)
        self.assertEqual([w[3] for w in up.writes], [False, True])
        self.assertIn('"required":["content","status"]', up.writes[1][0])
        self.assertNotIn('DO NOT DELIVER', json.dumps(result))
        cached, again = await self.run_turn(up)
        self.assertEqual(cached, result)
        self.assertTrue(again['cached'])
        self.assertEqual(len(up.writes), 2)
        self.assertNotIn('pending', self.store.get('desktop'))

    async def test_second_correction_can_succeed(self):
        up = ScriptedNotion([INVALID, INVALID, VALID])
        _, info = await self.run_turn(up)
        self.assertEqual(info['repair_attempts'], 2)
        self.assertEqual(len(up.writes), 3)

    async def test_repair_limit_survives_restart_and_identical_retries(self):
        up = ScriptedNotion([INVALID])
        with self.assertRaises(UpstreamError) as ctx:
            await self.run_turn(up)
        self.assertEqual(ctx.exception.status, 422)
        self.assertEqual(ctx.exception.diagnostics['error_code'], 'invalid_tool_arguments')
        self.assertEqual(len(up.writes), 3)
        self.store.close()
        self.store = SessionStore(self.path)
        with self.assertRaises(UpstreamError) as again:
            await self.run_turn(up)
        self.assertEqual(again.exception.status, 422)
        self.assertEqual(len(up.writes), 3)

    async def test_legacy_cached_invalid_answer_is_repaired_without_resending_user_request(self):
        up = ScriptedNotion([INVALID, VALID])
        def old_validator(text):
            raise ValueError('legacy validation failure')
        with self.assertRaises(ValueError):
            await self.store.run(up, 'desktop', self.prompt, 'medium', old_validator)
        record = self.store.get('desktop')
        self.assertIn('answer', record['pending'])
        _, info = await self.run_turn(up)
        self.assertEqual(info['repair_attempts'], 1)
        self.assertTrue(up.writes[-1][0].startswith('PROTOCOL CORRECTION:'))
        self.assertEqual(len(up.writes), 2)

    async def test_accepted_correction_response_loss_only_polls_after_restart(self):
        up = ScriptedNotion([INVALID, VALID])
        start = up.start_turn
        async def lost_response(*args):
            await start(*args)
            if len(up.writes) == 2:
                raise UpstreamError('response lost')
        up.start_turn = lost_response
        with self.assertRaises(UpstreamError):
            await self.run_turn(up)
        self.store.close()
        self.store = SessionStore(self.path)
        up.turn_is_absent = AsyncMock(return_value=False)
        up.start_turn = AsyncMock(side_effect=AssertionError('duplicate generation'))
        _, info = await self.run_turn(up)
        self.assertEqual(info['repair_attempts'], 1)
        up.start_turn.assert_not_awaited()

    async def test_unsent_correction_retry_reuses_exact_correction_and_event_id(self):
        up = ScriptedNotion([INVALID, VALID])
        start = up.start_turn
        async def fail_correction(*args):
            if len(up.writes) == 1:
                raise UpstreamError('send failed')
            await start(*args)
        up.start_turn = fail_correction
        with self.assertRaises(UpstreamError):
            await self.run_turn(up)
        pending = self.store.get('desktop')['pending']
        up.start_turn = start
        up.turn_is_absent = AsyncMock(return_value=True)
        _, info = await self.run_turn(up)
        self.assertEqual(up.writes[-1][0], pending['repair_prompt'])
        self.assertEqual(up.writes[-1][2], pending['event_id'])
        self.assertEqual(info['repair_attempts'], 1)
        self.assertEqual(len(up.writes), 2)

    async def test_changed_request_drains_correction_without_delivering_stale_tools(self):
        up = ScriptedNotion([INVALID, VALID, TEXT])
        start = up.start_turn
        async def pause(*args):
            await start(*args)
            if len(up.writes) == 2:
                up.wait_error = TimeoutError()
        up.start_turn = pause
        with self.assertRaises(TimeoutError):
            await self.run_turn(up)
        up.wait_error = None
        up.start_turn = start
        new = prepare_request({'model': 'claude-opus-5-5', 'max_tokens': 100,
            'tools': [PLAN], 'messages': [{'role': 'user', 'content': 'Stop planning; only reply OK'}]})[0]
        result, info = await self.run_turn(up, new)
        self.assertEqual(result['content'], [{'type': 'text', 'text': 'OK'}])
        self.assertEqual(len({w[1] for w in up.writes}), 1)
        self.assertTrue(info['recovered_pending'])

    async def test_completed_generation_failure_can_be_corrected_without_new_thread(self):
        up = ScriptedNotion([TEXT, VALID])
        poll = up.wait_for_turn
        error = UpstreamError('max_tokens')
        error.completed_turn = {'cursor': 'failed-cursor', 'sequence': 10, 'turn_id': 'failed'}
        async def terminal_once(*args):
            if len(up.writes) == 1:
                raise error
            return await poll(*args)
        up.wait_for_turn = terminal_once
        _, info = await self.run_turn(up)
        self.assertEqual(info['repair_attempts'], 1)
        self.assertEqual(up.polls[-1][1:], ('failed-cursor', 10, 'failed'))

    async def test_no_partial_prose_is_sent_to_progress_callback_before_validation(self):
        up = ScriptedNotion([INVALID, VALID])
        progress = []
        result, _ = await self.store.run(up, 'desktop', self.prompt, 'medium', render, progress.append)
        self.assertEqual(progress, [])
        self.assertNotIn('DO NOT DELIVER', json.dumps(result))

    async def test_unsent_continuation_gets_one_safe_internal_retry(self):
        up = ScriptedNotion([TEXT, TEXT])
        await self.run_turn(up)
        start = up.start_turn
        calls = []
        async def first_fails(*args):
            calls.append(args)
            if len(calls) == 1:
                raise UpstreamError('temporary failure')
            await start(*args)
        up.start_turn = first_fails
        up.turn_is_absent = AsyncMock(return_value=True)
        _, info = await self.run_turn(up, self.prompt.replace('Plan the check', 'Next check'))
        self.assertEqual(calls[0], calls[1])
        self.assertEqual(info['submission_retries'], 1)

    async def test_uncertain_accepted_write_is_not_internally_replayed(self):
        up = ScriptedNotion([TEXT])
        await self.run_turn(up)
        up.start_turn = AsyncMock(side_effect=UpstreamError('lost acknowledgement'))
        up.turn_is_absent = AsyncMock(return_value=False)
        with self.assertRaises(UpstreamError):
            await self.run_turn(up, self.prompt.replace('Plan the check', 'Next check'))
        self.assertEqual(up.start_turn.await_count, 1)
        self.assertIn('pending', self.store.get('desktop'))


class TranscriptRecoveryChecks(unittest.TestCase):
    def test_reused_turn_id_completes_only_after_a_new_user_message(self):
        state = Transcript(518, 'reused')
        state.update({'patches': [
            {'op': 'put', 'entity': {'id': 'old', 'kind': 'assistant_message', 'sequence': 516,
                'turn_id': 'reused', 'content': [{'type': 'text', 'text': 'OLD'}]}},
            {'op': 'put', 'entity': {'id': 'input', 'kind': 'user_message', 'sequence': 519}},
            {'op': 'put', 'entity': {'id': 'new', 'kind': 'assistant_message', 'sequence': 522,
                'turn_id': 'reused', 'content': [{'type': 'text', 'text': 'NEW'}]}},
            {'op': 'put', 'entity': {'id': 'done', 'kind': 'turn_completed', 'sequence': 524,
                'turn_id': 'reused', 'stop_reason': 'end_turn'}}]})
        self.assertEqual(state.partial_text(), 'NEW')
        self.assertEqual(state.result(), 'NEW')

    def test_same_id_snapshot_without_new_input_is_not_a_new_answer(self):
        state = Transcript(518, 'reused')
        state.update({'patches': [{'op': 'put', 'entity': {'id': 'done', 'kind': 'turn_completed',
            'sequence': 524, 'turn_id': 'reused', 'stop_reason': 'end_turn'}}]})
        self.assertIsNone(state.result())

    def test_newer_queued_user_message_cannot_receive_prior_turn_completion(self):
        state = Transcript(10, 'prior')
        state.update({'patches': [
            {'op': 'put', 'entity': {'id': 'input', 'kind': 'user_message', 'sequence': 20}},
            {'op': 'put', 'entity': {'id': 'done', 'kind': 'turn_completed', 'sequence': 19,
                'turn_id': 'other', 'stop_reason': 'end_turn'}}]})
        self.assertIsNone(state.result())


class ReadRetryChecks(unittest.IsolatedAsyncioTestCase):
    async def test_poll_deadline_is_504_with_explicit_preserved_turn_diagnostic(self):
        agent = object.__new__(NotionAgent)
        agent.timeout = 300
        agent._wait_for_turn = AsyncMock(side_effect=TimeoutError())
        with self.assertRaises(UpstreamError) as ctx:
            await agent.wait_for_turn('thread')
        self.assertEqual(ctx.exception.status, 504)
        self.assertEqual(ctx.exception.diagnostics['error_code'], 'upstream_timeout')
        self.assertFalse(ctx.exception.exhausted)

    async def test_read_retries_transport_and_503_but_write_is_only_sent_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'session.json'
            path.write_text(json.dumps({'origin': 'https://app.notion.com', 'space_id': 's', 'user_id': 'u', 'cookies': []}))
            agent = NotionAgent(path)
            await agent.client.aclose()
            calls = []
            def handler(request):
                calls.append(request.url.path)
                if len(calls) == 1:
                    raise httpx.ReadTimeout('private transport detail')
                return httpx.Response(503 if len(calls) == 2 else 200, json={})
            agent.client = httpx.AsyncClient(base_url='https://app.notion.com', transport=httpx.MockTransport(handler))
            with patch('app.agent_gateway.upstream.asyncio.sleep', new_callable=AsyncMock):
                self.assertEqual(await agent.post('getThreadTranscript', {}), {})
            self.assertEqual(len(calls), 3)
            calls.clear()
            with self.assertRaises(UpstreamError) as ctx:
                await agent.post('sendEventToAgentThread', {})
            self.assertEqual(len(calls), 1)
            self.assertNotIn('private transport detail', str(ctx.exception))
            await agent.close()

    async def test_auth_and_quota_are_not_retried_as_transient_reads(self):
        for status in (401, 403, 429):
            with self.subTest(status=status):
                agent = object.__new__(NotionAgent)
                agent.client = httpx.AsyncClient(base_url='https://app.notion.com',
                    transport=httpx.MockTransport(lambda request: httpx.Response(status, json={})))
                with patch('app.agent_gateway.upstream.asyncio.sleep', new_callable=AsyncMock) as sleep:
                    with self.assertRaises(UpstreamError) as ctx:
                        await agent.post('getThreadTranscript', {})
                    self.assertEqual(ctx.exception.status, status)
                    sleep.assert_not_awaited()
                await agent.close()


class StreamingRecoveryChecks(unittest.IsolatedAsyncioTestCase):
    async def test_messages_stream_emits_only_validated_result_without_progress_callback(self):
        from types import SimpleNamespace
        from starlette.requests import Request
        from app.agent_gateway.server import messages
        async def run(*args, **kwargs):
            self.assertIsNone(args[5])
            return ({'content': [{'type': 'text', 'text': 'VALIDATED'}], 'stop_reason': 'end_turn',
                     'usage': {'input_tokens': 1, 'output_tokens': 1}}, {})
        app = SimpleNamespace(state=SimpleNamespace(locks={}, sessions=None,
            pool=SimpleNamespace(run=run)))
        body = json.dumps({'model': 'claude-opus-5-5', 'max_tokens': 32, 'stream': True,
                           'messages': [{'role': 'user', 'content': 'test'}]}).encode()
        request = Request({'type': 'http', 'method': 'POST', 'path': '/v1/messages',
            'headers': [], 'app': app}, AsyncMock(return_value={
                'type': 'http.request', 'body': body, 'more_body': False}))
        with patch('app.agent_gateway.server.app', app), patch('app.agent_gateway.server.audit'):
            response = await messages(request)
            stream = ''.join([chunk async for chunk in response.body_iterator])
        self.assertIn('VALIDATED', stream)
        self.assertIn('event: message_stop', stream)
        self.assertNotIn('event: error', stream)

    async def test_exhausted_protocol_repairs_end_stream_without_success_or_retryable_502(self):
        from types import SimpleNamespace
        from starlette.requests import Request
        from app.agent_gateway.server import messages
        app = SimpleNamespace(state=SimpleNamespace(locks={}, sessions=None,
            pool=SimpleNamespace(run=AsyncMock(side_effect=UpstreamError('Correction limit reached', 422)))))
        body = json.dumps({'model': 'claude-opus-5-5', 'max_tokens': 32, 'stream': True,
                           'messages': [{'role': 'user', 'content': 'test'}]}).encode()
        request = Request({'type': 'http', 'method': 'POST', 'path': '/v1/messages',
            'headers': [], 'app': app}, AsyncMock(return_value={
                'type': 'http.request', 'body': body, 'more_body': False}))
        with patch('app.agent_gateway.server.app', app), patch('app.agent_gateway.server.audit'):
            response = await messages(request)
            stream = ''.join([chunk async for chunk in response.body_iterator])
        self.assertIn('invalid_request_error', stream)
        self.assertNotIn('event: message_stop', stream)
        self.assertNotIn('event: content_block_start', stream)
