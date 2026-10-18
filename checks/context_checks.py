import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.agent_gateway.protocol import prepare_request
from app.agent_gateway.sessions import SessionStore, incremental_prompt, MARKER
from session_checks import FakeNotion


def make(messages, system=None, tools=None):
    return prepare_request({'model':'claude-opus-5-5','max_tokens':1000,'messages':messages,
                            'system':system or '', 'tools':tools or []})[0]


class ContextChecks(unittest.IsolatedAsyncioTestCase):
    async def test_large_client_history_sends_only_delta_after_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'state.db'; store=SessionStore(path); up=FakeNotion()
            history=[{'role':'user','content':'Historical context '+ 'x'*170000}]
            result,first=await store.run(up,'session',make(history),'medium',json.loads)
            store.close(); store=SessionStore(path)
            history += [{'role':'assistant','content':result['content']},
                        {'role':'user','content':'new instruction '+ 'y'*12000}]
            full=make(history)
            self.assertGreater(len(full),181051)
            _,second=await store.run(up,'session',full,'medium',json.loads)
            self.assertEqual(first['thread_id'],second['thread_id'])
            self.assertLess(second['upstream_input_chars'],13000)
            self.assertNotIn('Historical context',up.writes[-1][2])
            self.assertNotIn('"role":"assistant"',up.writes[-1][2])
            store.close()

    def test_volatile_headers_and_token_counters_do_not_resend_system(self):
        def build(n):
            return make([{'role':'user','content':'remember'},
                         {'role':'system','content':f'Keep files safe.\n<total_tokens>{n} tokens left</total_tokens>'}],
                        system=[{'type':'text','text':f'x-anthropic-billing-header: nonce={n}'},
                                {'type':'text','text':'Important instructions '+ 'a'*50000}])
        _,snap,_,_=incremental_prompt(build(99),{})
        sent,_,_,mode=incremental_prompt(build(98),{'snapshot':snap})
        self.assertEqual(mode,'append')
        self.assertNotIn('Important instructions',sent)
        self.assertNotIn('Keep files safe',sent)
        self.assertNotIn('billing-header',sent)
        self.assertLess(len(sent),1000)

    async def test_tool_acknowledgment_does_not_resend_generated_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=SessionStore(Path(tmp)/'state.db');up=FakeNotion()
            tools=[{'name':'Write','input_schema':{'type':'object'}}]
            original=[{'role':'user','content':'write a file'}]
            response={'content':[{'type':'tool_use','id':'tool-1','name':'Write','input':{'content':'z'*50000}}]}
            await store.run(up,'s',make(original,tools=tools),'medium',lambda _:response)
            continued=original+[{'role':'assistant','content':response['content']},
                                {'role':'user','content':[{'type':'tool_result','tool_use_id':'tool-1','content':'File saved'}]}]
            _,info=await store.run(up,'s',make(continued,tools=tools),'medium',json.loads)
            payload=json.loads(up.writes[-1][2].split(MARKER)[1])
            self.assertEqual(payload['previous_tool_ids'],[{'id':'tool-1','name':'Write'}])
            self.assertNotIn('tools',payload)
            self.assertEqual(payload['messages'][0]['content'][0]['tool_use_id'],'tool-1')
            self.assertLess(info['upstream_input_chars'],1500)
            store.close()

    def test_only_large_tool_results_are_shortened_not_user_instructions(self):
        tools=[{'name':'Read','input_schema':{'type':'object'}}]
        messages=[{'role':'user','content':'instruction '+ 'u'*20000},
                  {'role':'assistant','content':[{'type':'tool_use','id':'a','name':'Read','input':{}}]},
                  {'role':'user','content':[{'type':'tool_result','tool_use_id':'a','content':'HEAD'+ 'x'*90000+'TAIL'}]}]
        sent,snap,_,_=incremental_prompt(make(messages,tools=tools),{})
        self.assertIn('u'*20000,sent)
        self.assertIn('Bridge truncated tool result',sent)
        self.assertIn('HEAD',sent);self.assertIn('TAIL',sent)
        self.assertEqual(snap['truncated_tool_results'],1)
        self.assertLess(len(sent),40000)

    def test_real_instruction_changes_and_removed_settings_are_propagated(self):
        base=make([{'role':'user','content':'do task'}],system='A')
        _,snap,_,_=incremental_prompt(base,{})
        snap['settings']['output_json_schema']='old'
        sent,_,_,_=incremental_prompt(make([{'role':'user','content':'do task'}],system='B'),{'snapshot':snap})
        payload=json.loads(sent.split(MARKER)[1])
        self.assertEqual(payload['system'][0]['text'],'B')
        self.assertIn('output_json_schema',payload['unset'])

    async def test_oversized_first_request_not_silently_discarded(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=SessionStore(Path(tmp)/'state.db');up=FakeNotion()
            from app.agent_gateway.upstream import UpstreamError
            with patch('app.agent_gateway.sessions.MAX_SEND_CHARS',10000),self.assertRaises(UpstreamError):
                await store.run(up,'s',make([{'role':'user','content':'u'*20000}]),'medium',json.loads)
            self.assertEqual(store.get('s'),{});self.assertEqual(up.writes,[])
            store.close()
