#!/bin/bash
set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

if [ -x "$SCRIPT_DIR/.venv/bin/python" ]; then
  PYTHON="$SCRIPT_DIR/.venv/bin/python"
elif [ -x "$SCRIPT_DIR/../.venv/bin/python" ]; then
  PYTHON="$SCRIPT_DIR/../.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PYTHON="$(command -v python)"
else
  echo "找不到 Python。請先安裝 Python 3.10 或 3.11。"
  read -r -p "按 Enter 關閉..."
  exit 1
fi

"$PYTHON" "$SCRIPT_DIR/solid_inpaint_ui.py"
STATUS=$?

if [ "$STATUS" -ne 0 ]; then
  echo
  echo "塗白啟動失敗，請先安裝依賴："
  echo "  cd \"$SCRIPT_DIR\""
  echo "  python3 -m venv .venv"
  echo "  .venv/bin/python -m pip install -r requirements.txt"
  echo
  read -r -p "按 Enter 關閉..."
fi

exit "$STATUS"
