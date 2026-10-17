#!/bin/bash
# 停止 Notion2API
PID=$(lsof -ti tcp:8000 2>/dev/null)
if [ -n "$PID" ]; then
  kill $PID && echo "已停止服务 (pid $PID)"
else
  echo "端口 8000 上没有运行中的服务"
fi
