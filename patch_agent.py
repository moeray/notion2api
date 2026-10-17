#!/usr/bin/env python3
"""
patch_agent.py —— 让 Notion AI（Opus 5.5）产出 unified diff，再在本地落地。

设计前提：Notion 反代不支持 function calling，模型没法自己动手改文件。
所以让模型只负责「出补丁」，落地这一步由本地 git 完成 —— 人始终在环上。

用法：
    # 只生成补丁，不落盘（默认，先看再改）
    python patch_agent.py "给 calc.py 加一个 subtract 函数" -f calc.py

    # 直接应用（会先 git apply --check 校验）
    python patch_agent.py "修复 login.py 的空指针" -f login.py --apply

    # 指定模型 / 多文件
    python patch_agent.py "把 utils 里的函数加上类型注解" -f a.py -f b.py -m claude-opus5.5

环境变量（可选覆盖）：
    NOTION_API_BASE  默认 http://127.0.0.1:8000/v1
    NOTION_API_KEY   默认 sk-notion-local
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.request

BASE = os.getenv("NOTION_API_BASE", "http://127.0.0.1:8000/v1")
KEY = os.getenv("NOTION_API_KEY", "sk-notion-local")

PROMPT_TEMPLATE = """你现在的任务是输出一段可直接被 `git apply` 应用的 unified diff。

要求（必须严格遵守）：
1. 只输出 diff，不要任何解释、不要寒暄、不要 Markdown 之外的文字。
2. 格式必须是标准 unified diff：以 `--- a/<路径>` 和 `+++ b/<路径>` 开头，带 @@ 段落头。
3. 路径要与下面给出的文件相对路径完全一致。
4. 新增文件用 `--- /dev/null` + `+++ b/<路径>`。
5. 上下文行前后各留 3 行，行首保留一个空格（+ 表示新增，- 表示删除，空格表示上下文）。
6. 不要改动与任务无关的代码。

任务：
{task}

下面是相关文件的完整内容：

{files}

现在输出 diff：
"""


def call_model(messages, model, stream=False, timeout=300):
    req = urllib.request.Request(
        f"{BASE}/chat/completions",
        data=json.dumps({"model": model, "messages": messages, "stream": stream}).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def extract_diff(text: str) -> str:
    """从回复里抠出 unified diff（优先 ```diff 代码块，其次裸 diff）。"""
    m = re.search(r"```(?:diff|patch)?\s*\n(.*?)```", text, re.S)
    if m:
        text = m.group(1)
    lines = text.splitlines()
    start = None
    for i, ln in enumerate(lines):
        if ln.startswith("--- ") or ln.startswith("diff --git"):
            start = i
            break
    if start is None:
        return ""
    return "\n".join(lines[start:]).rstrip() + "\n"


def main():
    ap = argparse.ArgumentParser(description="让 Notion AI 生成 unified diff 并在本地落地")
    ap.add_argument("task", help="用自然语言描述要做的改动")
    ap.add_argument("-f", "--file", action="append", default=[], required=True,
                    help="要修改的文件，可重复；也支持目录（会递归收集 .py/.js/.ts/.go/.java/.rs/.md 等）")
    ap.add_argument("-m", "--model", default="claude-opus5.5")
    ap.add_argument("--apply", action="store_true", help="校验通过后直接 git apply")
    ap.add_argument("--max-chars", type=int, default=60000, help="单个文件内容截断长度")
    args = ap.parse_args()

    paths = []
    for p in args.file:
        if os.path.isdir(p):
            for root, _, names in os.walk(p):
                if any(x in root for x in (".git", "node_modules", "__pycache__", ".venv")):
                    continue
                for n in names:
                    if n.rsplit(".", 1)[-1] in ("py", "js", "ts", "tsx", "jsx", "go", "java",
                                                "rs", "rb", "c", "h", "cpp", "md", "json", "yaml"):
                        paths.append(os.path.join(root, n))
        else:
            paths.append(p)

    blocks = []
    for p in paths:
        if not os.path.exists(p):
            print(f"[warn] 文件不存在，按新增处理：{p}")
            blocks.append(f"=== {p} (尚不存在) ===")
            continue
        with open(p, encoding="utf-8", errors="replace") as fh:
            content = fh.read()
        if len(content) > args.max_chars:
            content = content[: args.max_chars] + f"\n...[截断 {len(content) - args.max_chars} 字符]"
        blocks.append(f"=== {p} ===\n{content}")
    if not blocks:
        print("没有可读的文件"); return 1

    prompt = PROMPT_TEMPLATE.format(task=args.task, files="\n\n".join(blocks))
    print(f"[i] 请求 {args.model}，上下文 {len(prompt)} 字符 …", file=sys.stderr)
    try:
        resp = call_model([{"role": "user", "content": prompt}], args.model)
    except Exception as e:
        print(f"[x] 调用失败：{e}"); return 1

    raw = resp["choices"][0]["message"]["content"]
    diff = extract_diff(raw)
    if not diff:
        print("[x] 模型没有返回可识别的 diff，原始回复如下：\n")
        print(raw)
        return 1

    fd, path = tempfile.mkstemp(suffix=".patch")
    os.write(fd, diff.encode()); os.close(fd)

    print(f"\n===== 补丁已生成 ({len(diff.splitlines())} 行) =====\n")
    print(diff)

    if not args.apply:
        print(f"[i] 补丁文件：{path}")
        print("[i] 确认无误后执行：git apply " + path + "   （或加 --apply 让我直接落盘）")
        return 0

    # --recount：模型算错 @@ 行号/行数是常态，让 git 自己重算
    for extra in (["--recount"], ["--recount", "--3way"], ["--recount", "--ignore-whitespace"]):
        check = subprocess.run(["git", "apply", "--check", *extra, path],
                               capture_output=True, text=True)
        if check.returncode == 0:
            subprocess.run(["git", "apply", *extra, path], check=True)
            print(f"[✓] 已应用改动（git apply {' '.join(extra)}），用 `git diff` 复核。")
            return 0
    # 最后兜底：GNU patch
    if subprocess.run(["patch", "-p1", "-i", path], capture_output=True).returncode == 0:
        print("[✓] 已用 patch -p1 应用，用 `git diff` 复核。")
        return 0
    print("[!] 所有应用方式都失败了：", check.stderr.strip()[:500])
    print(f"[i] 补丁保留在 {path}，可手动 `git apply --recount --3way {path}`")
    return 1
    print("[✓] 已应用改动，用 `git diff` 复核。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
