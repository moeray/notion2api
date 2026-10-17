#!/bin/bash
# 启动 Notion2API（本地 OpenAI 兼容反代）
cd "$(dirname "$0")"
PY=/Users/ticture/.workbuddy/binaries/python/envs/default/bin/python
LOG="$(pwd)/server.log"
if lsof -ti tcp:8000 >/dev/null 2>&1; then
  echo "端口 8000 已被占用，服务可能已在运行：http://127.0.0.1:8000"
  exit 0
fi
nohup "$PY" -m uvicorn app.server:app --host 127.0.0.1 --port 8000 >> "$LOG" 2>&1 &
sleep 4
echo "已启动：http://127.0.0.1:8000   (日志: $LOG)"
curl -s http://127.0.0.1:8000/health || true
echo
