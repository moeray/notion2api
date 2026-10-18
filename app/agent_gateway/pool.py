"""Ordered Notion workspaces; a session moves on when its workspace runs out of quota.

Failover happens only before an answer is delivered to the client (a rejected turn
start or an unfinished turn), so local tools never run twice. Sessions stay on the
workspace they moved to; cooling workspaces are retried only as a last resort.
"""
from __future__ import annotations

import os
import time

from .upstream import UpstreamError

COOLDOWN = float(os.getenv("NOTION_GATEWAY_EXHAUSTED_COOLDOWN", "1800"))
# Seconds a workspace may stay silent after accepting a turn before the next one is tried.
STALL_TIMEOUT = float(os.getenv("NOTION_GATEWAY_STALL_TIMEOUT", "10"))


class WorkspacePool:
    def __init__(self, members, skipped=()):
        # members: [(profile name, NotionAgent)], primary first.
        self.members = list(members)
        self.skipped = list(skipped)
        self.cooling = {}

    def order(self, preferred_scope=None):
        now = time.time()
        healthy = [m for m in self.members if self.cooling.get(m[0], 0) <= now]
        healthy.sort(key=lambda m: getattr(m[1], "account_scope", None) != preferred_scope)
        cooling = sorted((m for m in self.members if self.cooling.get(m[0], 0) > now),
                         key=lambda m: self.cooling[m[0]])
        return healthy + cooling

    def status(self):
        now = time.time()
        rows = []
        for name, _ in self.members:
            until = self.cooling.get(name, 0)
            rows.append({"profile": name, "available": until <= now,
                         **({"retry_in_s": round(until - now)} if until > now else {})})
        return rows + [{**s, "available": False} for s in self.skipped]

    async def run(self, sessions, key, *args, **kwargs):
        preferred = sessions.get(key).get("scope") or sessions.namespace or None
        candidates = self.order(preferred)
        for index, (name, upstream) in enumerate(candidates):
            # The last candidate gets the normal timeout: a slow answer beats no answer.
            stall = STALL_TIMEOUT if STALL_TIMEOUT > 0 and index < len(candidates) - 1 else None
            try:
                result, info = await sessions.run(upstream, key, *args, **kwargs, stall_timeout=stall)
            except UpstreamError as exc:
                if not exc.exhausted:
                    raise
                self.cooling[name] = time.time() + (exc.retry_after or COOLDOWN)
                continue
            self.cooling.pop(name, None)
            return result, {**info, "profile": name}
        raise UpstreamError("All configured Notion workspaces are rate-limited or out of quota", 429)
