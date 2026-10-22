import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch
import httpx
from app.agent_gateway.upstream import NotionAgent, UpstreamError


class UpstreamChecks(unittest.IsolatedAsyncioTestCase):
    async def test_absent_turn_requires_repeated_explicit_idle_checkpoint(self):
        agent = object.__new__(NotionAgent)
        agent.base_url = 'https://app.notion.com'
        agent.space_id = 's'
        idle = {'session': {'status': 'idle', 'sequence': 10}, 'session_pending': False,
                'forward_after_sequence': 10, 'has_more_forward': False, 'patches': []}
        agent.post = AsyncMock(return_value=idle)
        with patch('app.agent_gateway.upstream.asyncio.sleep', new_callable=AsyncMock):
            self.assertTrue(await agent.turn_is_absent('thread', 'cursor', 10, 'prior'))
        self.assertEqual(agent.post.await_count, 3)
        self.assertTrue(all(call.args[1]['cursor'] == 'cursor' for call in agent.post.await_args_list))
        for changed in (
            {**idle, 'session_pending': True},
            {**idle, 'session_pending': None},
            {**idle, 'session': {'status': 'running', 'sequence': 10}},
            {**idle, 'session': {'status': 'idle', 'sequence': 11}},
            {**idle, 'forward_after_sequence': 11},
            {**idle, 'has_more_forward': True},
            {**idle, 'patches': [{'op': 'put', 'entity': {'kind': 'user_message', 'sequence': 11}}]},
            {},
        ):
            with self.subTest(page=changed):
                agent.post = AsyncMock(side_effect=[idle, changed])
                with patch('app.agent_gateway.upstream.asyncio.sleep', new_callable=AsyncMock):
                    self.assertFalse(await agent.turn_is_absent('thread', 'cursor', 10, 'prior'))

    async def test_absent_turn_cannot_infer_rejection_from_missing_thread_or_failed_read(self):
        agent = object.__new__(NotionAgent)
        agent.base_url = 'https://app.notion.com'
        agent.space_id = 's'
        agent.post = AsyncMock(side_effect=UpstreamError('read failed'))
        self.assertFalse(await agent.turn_is_absent('thread', None, -1, None))
        agent.post.assert_not_awaited()
        with self.assertRaises(UpstreamError):
            await agent.turn_is_absent('thread', 'cursor', 10, 'prior')

    async def test_terminal_failure_exposes_cursor_for_same_thread_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = Path(tmp) / 'session.json'
            session.write_text(json.dumps({'origin':'https://app.notion.com','space_id':'s',
                                           'user_id':'u','cookies':[]}))
            agent = NotionAgent(session)
            closer = getattr(agent.client, 'aclose', None) or getattr(agent.client, 'close', None)
            if closer is not None:
                await closer()
            agent.client = httpx.AsyncClient(base_url='https://app.notion.com',
                transport=httpx.MockTransport(lambda request: httpx.Response(200, json={
                    'forward_cursor':'done-cursor','forward_after_sequence':20,
                    'patches':[{'op':'put','entity':{'kind':'turn_completed','id':'done',
                        'sequence':20,'turn_id':'failed-turn','stop_reason':'max_tokens'}}]})))
            with self.assertRaises(UpstreamError) as ctx:
                await agent.wait_for_turn('thread', after_sequence=10, previous_turn_id='prior')
            self.assertEqual(ctx.exception.completed_turn,
                             {'cursor':'done-cursor','sequence':20,'turn_id':'failed-turn'})
            self.assertFalse(ctx.exception.exhausted)
            await agent.close()

    async def test_dynamodb_size_error_is_413_with_safe_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = Path(tmp) / 'session.json'
            session.write_text(json.dumps({'origin':'https://app.notion.com','space_id':'s',
                                           'user_id':'u','cookies':[]}))
            agent = NotionAgent(session)
            closer = getattr(agent.client, 'aclose', None) or getattr(agent.client, 'close', None)
            if closer is not None:
                await closer()
            def respond(request):
                return httpx.Response(400, json={'name':'ValidationError','errorId':'test-id',
                    'debugMessage':'create_session DynamoDB transaction action is 411174 bytes; max is 409600',
                    'message':'private content'})
            agent.client = httpx.AsyncClient(base_url='https://app.notion.com',
                                             transport=httpx.MockTransport(respond))
            with self.assertRaises(UpstreamError) as ctx:
                await agent.start_turn('private prompt', 'medium','thread','event')
            self.assertEqual(ctx.exception.status,413)
            self.assertTrue(ctx.exception.rejected)
            self.assertFalse(ctx.exception.exhausted)
            self.assertNotIn('private',str(ctx.exception)+json.dumps(ctx.exception.diagnostics))
            self.assertEqual(ctx.exception.diagnostics['upstream_name'],'ValidationError')
            await agent.close()

    async def test_streamed_oversize_is_not_a_retryable_api_error(self):
        import asyncio
        from types import SimpleNamespace
        from unittest.mock import AsyncMock, patch
        from starlette.requests import Request
        from app.agent_gateway.server import messages
        body = json.dumps({'model':'claude-opus-5-5','max_tokens':64,'stream':True,
                           'messages':[{'role':'user','content':'test'}]}).encode()
        app = SimpleNamespace(state=SimpleNamespace(locks={},sessions=None,
            pool=SimpleNamespace(run=AsyncMock(side_effect=UpstreamError('too large',413,True)))))
        receive=AsyncMock(return_value={'type':'http.request','body':body,'more_body':False})
        request=Request({'type':'http','method':'POST','path':'/v1/messages',
                         'headers':[],'app':app},receive)
        with patch('app.agent_gateway.server.audit'), patch('app.agent_gateway.server.app',app):
            response=await messages(request)
            stream=''.join([chunk async for chunk in response.body_iterator])
        errors=[json.loads(line[6:]) for line in stream.splitlines() if line.startswith('data: ')
                and json.loads(line[6:]).get('type')=='error']
        self.assertEqual(errors[0]['error']['type'],'invalid_request_error')
