#!/usr/bin/env python3
"""One manual command to start the gateway; never installs login autostart."""
import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.agent_gateway.config import configured_profiles, profile_path, venv_python


def gateway_pids():
    """PIDs of gateways started from this checkout (any port, including hung ones)."""
    pattern = str(ROOT / 'scripts' / 'local_gateway.py')
    try:
        out = subprocess.run(['pgrep', '-f', pattern], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [int(p) for p in out.split() if p.isdigit() and int(p) != os.getpid()]


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def stop_previous(grace=8.0):
    pids = gateway_pids()
    if not pids:
        return
    print(f'Stopping previous Notion proxy (pid {", ".join(map(str, pids))})...', flush=True)
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in pids:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + (grace if sig == signal.SIGTERM else 3)
        while time.monotonic() < deadline and any(alive(p) for p in pids):
            time.sleep(0.2)
        pids = [p for p in pids if alive(p)]
        if not pids:
            return
    raise SystemExit(f'Could not stop previous Notion proxy (pid {", ".join(map(str, pids))})')


def run_checks(python):
    """Validate the checkout before replacing a working local gateway."""
    result = subprocess.run([str(python), '-m', 'unittest', 'discover',
                             '-s', str(ROOT / 'checks'), '-p', '*_checks.py', '-q'], cwd=ROOT)
    if result.returncode:
        raise SystemExit('Gateway pre-start checks failed; the existing proxy was not stopped.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', default='default')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--failover', default=os.getenv('NOTION_GATEWAY_FAILOVER_PROFILES', 'auto'),
                        help='Comma-separated backup profiles; default: auto-discover saved profiles. Use an empty string to disable.')
    args = parser.parse_args()
    try:
        folder = profile_path(args.profile)
    except ValueError as exc:
        parser.error(str(exc))
    if not 1 <= args.port <= 65535:
        parser.error('port must be between 1 and 65535')
    try:
        profiles = configured_profiles(args.profile, args.failover)
    except ValueError as exc:
        parser.error(str(exc))
    for name in profiles[1:]:
        try:
            if not (profile_path(name) / 'notion-session.json').exists():
                parser.error(f'Failover profile {name} is not configured')
        except ValueError as exc:
            parser.error(str(exc))
    if not (folder / 'notion-session.json').exists():
        parser.error('Profile is not configured. Run scripts/setup_gateway.py --help first.')
    python = venv_python()
    if not python.exists():
        parser.error('Missing .venv; create it and install requirements.txt first')
    run_checks(python)
    stop_previous()
    os.environ['NOTION_GATEWAY_PROFILE'] = args.profile
    os.environ['NOTION_GATEWAY_FAILOVER_PROFILES'] = args.failover
    print(f'Starting Notion proxy at http://127.0.0.1:{args.port} (profile: {args.profile}). Ctrl+C stops it.', flush=True)
    os.chdir(ROOT)
    os.execv(str(python), [str(python), str(ROOT / 'scripts/local_gateway.py'), '--port', str(args.port)])


if __name__ == '__main__':
    main()
