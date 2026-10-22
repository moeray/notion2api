"""A failed checkout validation must not take down a healthy running gateway."""
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from scripts import notion_proxy


class LauncherChecks(unittest.TestCase):
    def run_main(self, validation_status):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / 'notion-session.json').touch()
            python = folder / 'python'
            python.touch()
            order = Mock()
            with patch('sys.argv', ['notion_proxy.py']), patch.dict(os.environ, {}), \
                 patch.object(notion_proxy, 'profile_path', return_value=folder), \
                 patch.object(notion_proxy, 'configured_profiles', return_value=['default']), \
                 patch.object(notion_proxy, 'venv_python', return_value=python), \
                 patch.object(notion_proxy.subprocess, 'run', return_value=subprocess.CompletedProcess([], validation_status)) as check, \
                 patch.object(notion_proxy, 'stop_previous') as stop, \
                 patch.object(notion_proxy.os, 'chdir'), \
                 patch.object(notion_proxy.os, 'execv') as launch:
                order.attach_mock(check, 'check')
                order.attach_mock(stop, 'stop')
                order.attach_mock(launch, 'launch')
                if validation_status:
                    with self.assertRaisesRegex(SystemExit, 'existing proxy was not stopped'):
                        notion_proxy.main()
                    stop.assert_not_called()
                    launch.assert_not_called()
                else:
                    with patch('builtins.print'):
                        notion_proxy.main()
                    self.assertEqual([call[0] for call in order.mock_calls], ['check', 'stop', 'launch'])
                self.assertIn('*_checks.py', check.call_args.args[0])

    def test_failed_checks_preserve_existing_service(self):
        self.run_main(1)

    def test_passing_checks_precede_stop_and_launch(self):
        self.run_main(0)
