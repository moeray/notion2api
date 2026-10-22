import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.agent_gateway import config


class ConfigChecks(unittest.TestCase):
    def test_saved_backups_are_loaded_without_flags(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(config, "LOCAL", Path(folder)):
            root = Path(folder)
            (root / "notion-session.json").touch()
            for name in ("space", "release-check"):
                path = root / "profiles" / name
                path.mkdir(parents=True)
                (path / "notion-session.json").touch()
            (root / "profiles" / "unfinished").mkdir()
            self.assertEqual(config.configured_profiles("default"), ["default", "release-check", "space"])
            self.assertEqual(config.configured_profiles("space", "auto"), ["space", "default", "release-check"])
            self.assertEqual(config.configured_profiles("default", ""), ["default"])
            self.assertEqual(config.configured_profiles("default", "space,space"), ["default", "space"])

    def test_explicit_invalid_profile_is_rejected(self):
        with self.assertRaises(ValueError):
            config.configured_profiles("default", "../secret")

    def test_launcher_honors_environment_and_cli_override(self):
        # Exercise the actual argument parser, stopping before it can replace the service.
        code = '''
from unittest.mock import patch
from scripts import notion_proxy
with patch.object(notion_proxy, "configured_profiles", side_effect=lambda primary, value: print(value) or []), \\
     patch.object(notion_proxy.Path, "exists", return_value=True), \\
     patch.object(notion_proxy, "run_checks"), \\
     patch.object(notion_proxy, "stop_previous", side_effect=SystemExit):
    notion_proxy.main()
'''
        for args, expected in (([], "space"), (["--failover", ""], ""), (["--failover", "auto"], "auto")):
            result = subprocess.run([sys.executable, "-c", code, *args], cwd=config.ROOT,
                                    env={**os.environ, "NOTION_GATEWAY_FAILOVER_PROFILES": "space"},
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), expected)


if __name__ == "__main__":
    unittest.main()
