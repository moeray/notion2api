"""Local paths shared by the server and first-run tools."""
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCAL = ROOT / ".local"
CATALOG = Path(__file__).with_name("model_catalog.json")


def profile_path(name):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", name):
        raise ValueError("Profile must contain 1-64 letters, digits, underscores or hyphens")
    return LOCAL if name == "default" else LOCAL / "profiles" / name


PROFILE = os.getenv("NOTION_GATEWAY_PROFILE", "default")
PROFILE_LOCAL = profile_path(PROFILE)


def configured_profiles(primary, failover=None):
    """Auto-discover saved profiles unless an explicit list (including empty) is given."""
    profile_path(primary)
    if failover is None or failover.strip() == "auto":
        names = ["default"] if (LOCAL / "notion-session.json").is_file() else []
        names += [p.parent.name for p in sorted((LOCAL / "profiles").glob("*/notion-session.json"))]
        names = [n for n in names if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", n)]
    else:
        names = [n.strip() for n in failover.split(",") if n.strip()]
        for name in names:
            profile_path(name)
    return list(dict.fromkeys([primary, *names]))


# Workspaces to continue in, in order, when the active one runs out of quota.
PROFILES = configured_profiles(PROFILE, os.getenv("NOTION_GATEWAY_FAILOVER_PROFILES"))


def venv_python():
    return ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
