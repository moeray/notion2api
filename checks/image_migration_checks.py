"""Existing text-only sessions must survive native-image rollout without history replay."""
import base64
import json
import unittest
from unittest.mock import patch

from app.agent_gateway.protocol import prepare_request
from app.agent_gateway.sessions import incremental_prompt, digest, delivered_atoms, wire_size

PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=')
PLACEHOLDER = {'type': 'text', 'text': '[image (image/png, ~0KB) omitted: this bridge is text-only]'}


def image(n=0):
    return {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png',
        'data': base64.b64encode(PNG + bytes([n])).decode()}}


def prompt(messages):
    return prepare_request({'model': 'claude-opus-5-5', 'max_tokens': 1000, 'messages': messages})[0]


def payload(sent):
    return json.loads(sent.split('CLIENT_CONTEXT_JSON:\n', 1)[1])


class ImageMigrationChecks(unittest.TestCase):
    def test_old_screenshot_tool_result_migrates_without_replaying_long_history(self):
        old = [{'role': 'user', 'content': 'Original task ' + 'x' * 120000},
               {'role': 'assistant', 'content': [{'type': 'tool_use', 'id': 'shot', 'name': 'Screenshot', 'input': {}}]},
               {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'shot',
                   'content': [PLACEHOLDER, {'type': 'text', 'text': 'Screenshot captured'}]}]}]
        with patch('app.agent_gateway.sessions.NEW_THREAD_MAX_CHARS', 0):
            _, snapshot, _, _ = incremental_prompt(prompt(old), {})
        result = {'content': [{'type': 'text', 'text': 'Earlier work finished'}]}
        snapshot['assistant'] = [digest(a) for a in delivered_atoms(result)]
        current = old[:2] + [{'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'shot',
            'content': [image(), {'type': 'text', 'text': 'Screenshot captured'}]}]},
            {'role': 'assistant', 'content': result['content']},
            {'role': 'user', 'content': 'Validate the changes and produce the screenshot report'}]
        sent, migrated, _, mode = incremental_prompt(prompt(current), {'snapshot': snapshot})
        data = payload(sent)
        self.assertEqual(mode, 'append')
        self.assertEqual(migrated['omitted_earlier_items'], 0)
        self.assertLess(wire_size(sent), 4000)
        self.assertNotIn('Original task', sent)
        self.assertNotIn('"type":"tool_use"', sent)
        self.assertEqual(data['messages'], [current[-1] | {'content': [
            {'type': 'text', 'text': current[-1]['content']}]}])
        self.assertEqual(len(sent.images), 1)
        self.assertEqual(data['historical_image_context'][0]['tool_use_id'], 'shot')
        self.assertEqual(data['historical_image_context'][0]['history_item'], 2)
        # Persisted canonical hashes make subsequent requests ordinary appends.
        next_sent, _, _, next_mode = incremental_prompt(prompt(current + [
            {'role': 'assistant', 'content': 'Report complete'}, {'role': 'user', 'content': 'Continue'}]),
            {'snapshot': migrated})
        self.assertEqual(next_mode, 'append')
        self.assertFalse(getattr(next_sent, 'images', None))
        self.assertNotIn('historical_image_context', payload(next_sent))

    def test_user_image_and_question_boundary_metadata_are_legacy_compatible(self):
        old = [{'role': 'user', 'content': [PLACEHOLDER, {'type': 'text', 'text': 'Check this image'}]}]
        _, snapshot, _, _ = incremental_prompt(prompt(old), {})
        current = [{'role': 'user', 'content': [image(), {'type': 'text', 'text': 'Check this image'}]},
                   {'role': 'assistant', 'content': 'Done'}, {'role': 'user', 'content': 'Next task'}]
        sent, _, _, mode = incremental_prompt(prompt(current), {'snapshot': snapshot})
        self.assertEqual(mode, 'append')
        self.assertEqual(len(sent.images), 1)
        self.assertNotIn('Check this image', sent)

    def test_real_history_edit_after_legacy_image_is_not_hidden_by_migration(self):
        old = [{'role': 'user', 'content': [PLACEHOLDER, {'type': 'text', 'text': 'Original image task'}]},
               {'role': 'assistant', 'content': 'Done'}, {'role': 'user', 'content': 'Old requirement'}]
        _, snapshot, _, _ = incremental_prompt(prompt(old), {})
        current = [{'role': 'user', 'content': [image(), {'type': 'text', 'text': 'Original image task'}]},
                   {'role': 'assistant', 'content': 'Done'}, {'role': 'user', 'content': 'Corrected requirement'}]
        sent, _, _, mode = incremental_prompt(prompt(current), {'snapshot': snapshot})
        self.assertEqual(mode, 'resync')
        self.assertIn('Corrected requirement', sent)
        self.assertNotIn('Old requirement', sent)
        self.assertNotIn('Original image task', sent)
        self.assertEqual(len(sent.images), 1)

    def test_overflow_historical_images_are_deferred_and_finish_next_turn(self):
        old = [{'role': 'user', 'content': [PLACEHOLDER, {'type': 'text', 'text': f'Old image {i}'}]}
               for i in range(22)]
        _, snapshot, _, _ = incremental_prompt(prompt(old), {})
        current = [{'role': 'user', 'content': [image(i), {'type': 'text', 'text': f'Old image {i}'}]}
                   for i in range(22)]
        current += [{'role': 'user', 'content': [image(99), {'type': 'text', 'text': 'New image'}]}]
        sent, migrated, _, mode = incremental_prompt(prompt(current), {'snapshot': snapshot})
        self.assertEqual(mode, 'append')
        self.assertEqual(len(sent.images), 19)
        new_id = payload(sent)['messages'][0]['content'][0]['image_id']
        self.assertIn(new_id, sent.images)
        self.assertEqual(len(migrated['pending_image_upgrades']), 4)
        self.assertEqual(payload(sent)['deferred_historical_images'], 4)
        followup, final_snapshot, _, mode = incremental_prompt(prompt(current + [
            {'role': 'user', 'content': 'Continue the same task'}]), {'snapshot': migrated})
        self.assertEqual(mode, 'append')
        self.assertEqual(len(followup.images), 4)
        self.assertNotIn('pending_image_upgrades', final_snapshot)
        self.assertNotIn(new_id, followup.images)

    def test_different_old_text_is_not_treated_as_an_image_placeholder(self):
        old = [{'role': 'user', 'content': 'Unrelated actual instruction'}]
        _, snapshot, _, _ = incremental_prompt(prompt(old), {})
        sent, _, _, mode = incremental_prompt(prompt([{'role': 'user', 'content': [image()]}]),
                                               {'snapshot': snapshot})
        self.assertEqual(mode, 'resync')
        self.assertNotIn('historical_image_context', payload(sent))
        self.assertEqual(len(sent.images), 1)


class ImageMigrationSessionChecks(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import AsyncMock
        from app.agent_gateway.media import FilePrompt
        from app.agent_gateway.sessions import SessionStore
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / 'sessions.sqlite3'
        self.store = SessionStore(self.path)
        self.up = SimpleNamespace(account_scope='original-workspace',
            start_turn=AsyncMock(), turn_is_absent=AsyncMock(return_value=True),
            wait_for_turn=AsyncMock(return_value=('{"content":[{"type":"text","text":"Done"}]}', 'cursor', 10, 'turn')),
            prepare_images=AsyncMock(side_effect=lambda text, *args: FilePrompt(str(text), ['uploaded-image'])))
        old = [{'role': 'user', 'content': [PLACEHOLDER, {'type': 'text', 'text': 'Existing task'}]}]
        result, info = await self.store.run(self.up, 'client', prompt(old), 'medium', json.loads)
        self.original_thread = info['thread_id']
        current = [{'role': 'user', 'content': [image(), {'type': 'text', 'text': 'Existing task'}]},
                   {'role': 'assistant', 'content': result['content']},
                   {'role': 'user', 'content': 'Produce the screenshot report'}]
        self.current = prompt(current)
        record = self.store.get('client')
        _, snapshot, fingerprint, _ = incremental_prompt(self.current, record)
        record['pending'] = {'fingerprint': digest([fingerprint, 'medium']), 'snapshot': snapshot,
            'mode': 'resync', 'input_chars': 142665, 'input_bytes': 170772,
            'event_id': 'original-failed-event', 'submitted_at': 0, 'dispatch_accepted': False,
            'preserve_history': False}
        self.store.save('client', record)
        self.before = record
        self.store.save('unrelated', {'thread_id': 'keep-this-thread'})
        self.up.start_turn.reset_mock()

    async def asyncTearDown(self):
        self.store.close()
        self.tmp.cleanup()

    async def test_existing_failed_session_migrates_and_cached_retry_survives_restart(self):
        from app.agent_gateway.sessions import SessionStore
        result, info = await self.store.run(self.up, 'client', self.current, 'medium', json.loads)
        self.assertEqual(info['thread_id'], self.original_thread)
        self.assertEqual(info['history_mode'], 'append')
        self.assertEqual(self.up.start_turn.call_args.args[3], 'original-failed-event')
        submitted = self.up.start_turn.call_args.args[0]
        self.assertLess(wire_size(submitted), 4000)
        self.assertEqual(submitted.file_ids, ['uploaded-image'])
        self.assertEqual(len(self.up.prepare_images.call_args.args[0].images), 1)
        self.assertEqual(self.store.get('unrelated'), {'thread_id': 'keep-this-thread'})
        self.store.close()
        self.store = SessionStore(self.path)
        cached, again = await self.store.run(self.up, 'client', self.current, 'medium', json.loads)
        self.assertEqual(cached, result)
        self.assertTrue(again['cached'])
        self.up.prepare_images.assert_awaited_once()
        self.up.start_turn.assert_awaited_once()

    async def test_failed_historical_image_upload_preserves_existing_pending_record(self):
        from app.agent_gateway.upstream import UpstreamError
        self.up.prepare_images.side_effect = UpstreamError('upload interrupted')
        with self.assertRaises(UpstreamError):
            await self.store.run(self.up, 'client', self.current, 'medium', json.loads)
        self.assertEqual(self.store.get('client'), self.before)
        self.up.start_turn.assert_not_awaited()
        self.assertEqual(self.store.get('unrelated'), {'thread_id': 'keep-this-thread'})
