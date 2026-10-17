#!/usr/bin/env python3
"""
刷新 accounts.json 里的 token_v2。

Notion 的 token_v2 会过期，重新在 Chrome 里登录 Notion 后运行本脚本即可，
不需要重新填写 space_id / user_id 等字段。

    python refresh_token.py
"""
import json
import os
import re
import sqlite3
import subprocess
import sys
import urllib.parse

from Crypto.Cipher import AES
from Crypto.Protocol.KDF import PBKDF2

ACCOUNTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "accounts.json")
CHROME_DIR = os.path.expanduser("~/Library/Application Support/Google/Chrome")


def _key() -> bytes:
    pw = subprocess.run(
        ["security", "find-generic-password", "-w", "-s", "Chrome Safe Storage", "-a", "Chrome"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().encode()
    return PBKDF2(pw, b"saltysalt", 16, 1003)


def _decrypt(blob: bytes, key: bytes):
    if blob[:3] != b"v10":
        return None
    pt = AES.new(key, AES.MODE_CBC, b" " * 16).decrypt(blob[3:])
    body = pt[16:]
    pad = body[-1]
    if isinstance(pad, int) and 1 <= pad <= 16 and body[-pad:] == bytes([pad]) * pad:
        body = body[:-pad]
    m = re.search(rb"[\x20-\x7e]{6,}$", body)
    return m.group(0).decode("ascii") if m else None


def read_cookies():
    key = _key()
    out = {}
    for prof in ("Default", "Profile 1", "Profile 2", "Profile 3"):
        path = os.path.join(CHROME_DIR, prof, "Cookies")
        if not os.path.exists(path):
            continue
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        rows = con.execute(
            "select name, encrypted_value from cookies where host_key like '%notion%'"
        ).fetchall()
        con.close()
        for name, ev in rows:
            if name not in ("token_v2", "notion_user_id") or name in out:
                continue
            v = _decrypt(ev, key) if ev else None
            if v:
                out[name] = urllib.parse.unquote(v)
    return out


def main():
    ck = read_cookies()
    if not ck.get("token_v2"):
        print("未在 Chrome 中找到 Notion 的 token_v2 —— 请先在 Chrome 登录 https://www.notion.so")
        return 1
    accounts = json.load(open(ACCOUNTS, encoding="utf-8"))
    for acc in accounts:
        acc["token_v2"] = ck["token_v2"]
        if ck.get("notion_user_id"):
            acc["user_id"] = ck["notion_user_id"]
    json.dump(accounts, open(ACCOUNTS, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"已更新 {len(accounts)} 个账号的 token_v2（{ck['token_v2'][:12]}...）")
    print("重启服务生效： ./stop.sh && ./start.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
