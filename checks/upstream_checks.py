import json
import tempfile
import unittest
from pathlib import Path
import httpx
from app.agent_gateway.upstream import NotionAgent, UpstreamError


class UpstreamChecks(unittest.IsolatedAsyncioTestCase):
    async def test_dynamodb_size_error_is_413_with_safe_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = Path(tmp) / 'session.json'
            session.write_text(json.dumps({'origin':'https://app.notion.com','space_id':'s',
                                           'user_id':'u','cookies':[]}))
            agent = NotionAgent(session)
            await agent.client.aclose()
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
        app = SimpleNamespace(state=SimpleNamespace(lock=asyncio.Semaphore(1),sessions=None,
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
