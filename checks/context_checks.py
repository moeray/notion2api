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

    def test_resync_does_not_resend_unchanged_large_system_or_tools(self):
        system='Safety instructions '+ 's'*100000
        tools=[{'name':'Read','input_schema':{'type':'object'}}]
        _,snap,_,_=incremental_prompt(make([{'role':'user','content':'classify action A'}],system=system,tools=tools),{})
        sent,_,_,mode=incremental_prompt(make([{'role':'user','content':'classify action B'}],system=system,tools=tools),{'snapshot':snap})
        payload=json.loads(sent.split(MARKER)[1])
        self.assertEqual(mode,'resync')
        self.assertNotIn('system',payload)
        self.assertNotIn('tools',payload)
        self.assertIn('classify action B',sent)
        self.assertLess(len(sent),1000)

    def test_billing_header_filter_preserves_instructions_in_same_block(self):
        sent,_,_,_=incremental_prompt(make([{'role':'user','content':'hello'}],
            system='x-anthropic-billing-header: tracking\nNever modify protected files.'),{})
        self.assertIn('Never modify protected files.',sent)
        self.assertNotIn('x-anthropic-billing-header',sent)


class BudgetChecks(unittest.IsolatedAsyncioTestCase):
    def test_oversized_resync_drops_oldest_turns_keeping_tool_pairs(self):
        tools=[{'name':'Read','input_schema':{'type':'object'}}]
        messages=[]
        for i in range(30):
            messages += [{'role':'user','content':f'q{i} '+'x'*3000},
                         {'role':'assistant','content':[{'type':'tool_use','id':f't{i}','name':'Read','input':{}}]},
                         {'role':'user','content':[{'type':'tool_result','tool_use_id':f't{i}','content':'y'*3000}]}]
        with patch('app.agent_gateway.sessions.MAX_SEND_CHARS',40000):
            sent,snap,_,mode=incremental_prompt(make(messages,tools=tools),{})
        payload=json.loads(sent.split(MARKER)[1])
        self.assertLess(len(sent),40000)
        self.assertGreater(payload['omitted_earlier_items'],0)
        first=payload['messages'][0]
        self.assertEqual(first['role'],'user');self.assertEqual(first['content'][0]['type'],'text')
        self.assertIn('q29',sent);self.assertNotIn('q0 ',sent)
        self.assertEqual(len(snap['history']),90)

    async def test_long_thread_rolls_over_to_fresh_trimmed_thread(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=SessionStore(Path(tmp)/'state.db');up=FakeNotion()
            history=[{'role':'user','content':'old '+'o'*30000}]
            with patch('app.agent_gateway.sessions.THREAD_MAX_CHARS',50000), \
                 patch('app.agent_gateway.sessions.ROLLOVER_SEED_CHARS',20000):
                result,first=await store.run(up,'s',make(history),'medium',json.loads)
                history += [{'role':'assistant','content':result['content']},
                            {'role':'user','content':'next '+'n'*25000}]
                _,second=await store.run(up,'s',make(history),'medium',json.loads)
            self.assertNotEqual(first['thread_id'],second['thread_id'])
            self.assertEqual(second['history_mode'],'rollover')
            self.assertNotIn('o'*1000,up.writes[-1][2])
            self.assertIn('n'*1000,up.writes[-1][2])
            store.close()
