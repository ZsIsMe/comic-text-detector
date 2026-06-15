#!/bin/bash
set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

if command -v python3 >/dev/null 2>&1; then
  PYTHON="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PYTHON="$(command -v python)"
else
  echo "需要 Python 3.10 或更新版本。"
  echo "請從 https://www.python.org/downloads/ 安裝 Python。"
  read -r -p "按 Enter 關閉..."
  exit 1
fi

"$PYTHON" "$SCRIPT_DIR/bootstrap.py"
STATUS=$?

if [ "$STATUS" -ne 0 ]; then
  echo
  read -r -p "按 Enter 關閉..."
fi

exit "$STATUS"
