"""PATCHED-BY-OPS: model slug -> Notion upstream model id.

The shipped gateway hard-codes a single model (`albuquerque-quinn`, i.e. Opus
5.5) in upstream.py, so every request runs on that one model even though the
workspace exposes dozens via `getAvailableModels` and `createAgentThread`
accepts any of them.

`model_map.json` is generated from the live workspace by
`C:\\UGit\\gen_model_map.py` (slugifies each modelMessage). Resolve falls back
to the built-in default so an unknown slug still behaves like before.
"""
from __future__ import annotations

import json
from pathlib import Path

from .upstream import MODEL

_MAP_PATH = Path(__file__).with_name("model_map.json")


def _load() -> dict[str, str]:
    try:
        data = json.loads(_MAP_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items()
            if isinstance(k, str) and isinstance(v, str)}


MAP: dict[str, str] = _load()


AUTO_SLUG = "notion-auto"


def resolve(slug) -> str:
    """Upstream model id for a client-facing slug (defaults to Opus 5.5).

    `notion-auto` resolves to the literal "auto", which upstream.py turns into
    "let Notion choose" (no model / reasoningEffort on thread creation). That
    path keeps working when the explicit paid models are out of AI credits.
    """
    if not slug:
        return MODEL
    if str(slug).lower() == AUTO_SLUG:
        return "auto"
    return MAP.get(str(slug)) or MODEL


def entries() -> list[tuple[str, str]]:
    return list(MAP.items())


def known(slug) -> bool:
    """True when `slug` is one of the workspace's mapped models (or auto)."""
    if not isinstance(slug, str):
        return False
    return slug in MAP or slug.lower() == AUTO_SLUG
