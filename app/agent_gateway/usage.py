"""Read Notion allowances without inferring billing or context capacity."""
from __future__ import annotations

import asyncio
import math
import os
import re
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import httpx

from .protocol import estimate_tokens
from .upstream import UpstreamError


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def timestamp(value):
    # Notion timestamps are milliseconds. Do not guess units or a reset date.
    if number(value) and 0 < value < 4102444800000:
        return datetime.fromtimestamp(value / 1000, timezone.utc).isoformat()
    return None


def window(value, cadence):
    if not isinstance(value, dict):
        return None
    used, limit = value.get("used"), value.get("limit")
    if not number(used) or not number(limit) or used < 0 or limit <= 0:
        return None
    result = {"cadence": cadence, "used": used, "limit": limit,
              "percent_used": round(used / limit * 100, 2),
              "percent_remaining": round(max(0, 1 - used / limit) * 100, 2)}
    duration = value.get("window")
    if isinstance(duration, str) and re.fullmatch(r"[1-9][0-9]{0,3}[mhd]", duration):
        result["window"] = duration
    for key, source in (("resets_at", "periodEndMs"), ("starts_at", "periodStartMs")):
        if (date := timestamp(value.get(source))) is not None:
            result[key] = date
    return result


def quota_snapshot(data):
    if not isinstance(data, dict):
        raise ValueError("Malformed allowance response")
    rolling = window(data.get("window"), "rolling")
    monthly = window(data.get("billingPeriodWindow"), "billing_period")
    allowances = data.get("allowances")
    for allowance in allowances if isinstance(allowances, list) else []:
        if not isinstance(allowance, dict) or allowance.get("state") != "resolved" \
                or allowance.get("enforcement") != "enforced" or allowance.get("bucketId") != "standard":
            continue
        # Supplement the legacy verdict, using only the same enforced bucket.
        if allowance.get("cadence") == "rolling" and rolling is None:
            rolling = window(allowance, "rolling")
        if allowance.get("cadence") == "billing_period":
            if monthly is None:
                monthly = window(allowance, "billing_period")
            elif (start := timestamp(allowance.get("periodStartMs"))) is not None \
                    and monthly.get("resets_at") == timestamp(allowance.get("periodEndMs")):
                monthly["starts_at"] = start
    if rolling is None and monthly is None:
        raise ValueError("No allowance windows supplied")
    return {"source": "notion", "status": "ok", "observed_at": datetime.now(timezone.utc).isoformat(),
            "rolling": rolling, "billing_period": monthly}


def context_usage(input_tokens=None):
    return {"source": "local_estimate", "request_input_tokens_estimate": input_tokens,
            "includes_image_tokens": False, "notion_context_used_tokens": None,
            "notion_context_limit_tokens": None, "notion_context_remaining_tokens": None}


def desktop_footer_enabled(headers, body, options, preserve_history=False):
    if os.getenv("NOTION_GATEWAY_DESKTOP_USAGE_FOOTER", "true").lower() in {"0", "false", "off"}:
        return False
    if preserve_history or options["schema"] is not None or options["stop_sequences"]:
        return False
    preference = headers.get("x-notion-usage-display", "")
    if preference == "off":
        return False
    return preference == "footer" or any(t.get("name", "").startswith("mcp__ccd_")
                                          for t in body.get("tools", []))


def percent(value):
    return f"{value:g}%"


def footer_text(quota, input_tokens):
    parts = []
    rolling, monthly = quota.get("rolling"), quota.get("billing_period")
    if rolling:
        duration = rolling.get("window")
        label = "滚动额度"
        if duration:
            label = duration[:-1] + {"m": " 分钟", "h": " 小时", "d": " 天"}[duration[-1]]
        parts.append(f"{label}已用 **{percent(rolling['percent_used'])}**"
                     f"（剩余 {percent(rolling['percent_remaining'])}）")
    if monthly:
        reset = ""
        if monthly.get("resets_at"):
            date = datetime.fromisoformat(monthly["resets_at"]).astimezone(ZoneInfo("Asia/Shanghai"))
            reset = f"，{date.month}/{date.day} {date:%H:%M} 重置（北京时间）"
        parts.append(f"月度账期已用 **{percent(monthly['percent_used'])}**"
                     f"（剩余 {percent(monthly['percent_remaining'])}{reset}）")
    if not parts:
        parts.append("额度暂不可读取")
    if quota["status"] == "stale":
        parts.append("刷新失败，以上为上次读取值")
    if quota.get("observed_at"):
        date = datetime.fromisoformat(quota["observed_at"]).astimezone(ZoneInfo("Asia/Shanghai"))
        parts.append(f"更新于 {date.month}/{date.day} {date:%H:%M:%S}")
    context = (f"本次输入约 **{input_tokens:,} tokens**（文本估算，不含图片）"
               if isinstance(input_tokens, int) else "本次输入量未提供")
    return ("\n\n---\n> **Notion 用量** · " + " · ".join(parts)
            + "\n> 上下文：" + context + "；Notion 官方上限和剩余量未提供。")


class UsageReporter:
    def __init__(self, ttl=30, timeout=5):
        self.ttl, self.timeout = ttl, timeout
        self.cache = {}
        self.locks = {}

    async def read(self, upstream):
        scope = upstream.account_scope
        lock = self.locks.setdefault(scope, asyncio.Lock())
        async with lock:
            cached, expires = self.cache.get(scope, (None, 0))
            if time.monotonic() < expires:
                return cached
            try:
                async with asyncio.timeout(self.timeout):
                    data = await upstream.post("getCreditRateLimitStatus", {"spaceId": upstream.space_id})
                result = quota_snapshot(data)
                lifetime = self.ttl
            except (UpstreamError, httpx.HTTPError, TimeoutError, ValueError, OverflowError):
                # Quota display must never fail a generation or switch its workspace.
                result = ({**cached, "status": "stale"} if cached and cached.get("observed_at") else
                          {"source": "notion", "status": "unavailable", "rolling": None,
                           "billing_period": None, "observed_at": None})
                lifetime = min(5, self.ttl)
            self.cache[scope] = result, time.monotonic() + lifetime
            return result

    async def decorate(self, result, upstream):
        # Never add prose to tool calls, truncations or structured responses.
        if result.get("stop_reason") != "end_turn" or not result.get("content") \
                or any(b.get("type") != "text" for b in result["content"]):
            return result
        quota = await self.read(upstream)
        content = [dict(b) for b in result["content"]]
        content[-1]["text"] += footer_text(quota, result.get("usage", {}).get("input_tokens"))
        return {**result, "content": content,
                "usage": {**result.get("usage", {}), "output_tokens": estimate_tokens(content)}}
