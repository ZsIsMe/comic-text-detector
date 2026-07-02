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
else
  CURRENT_TTY="$(tty)"
  osascript -e "delay 0.4" \
    -e "tell application \"Terminal\"" \
    -e "repeat with w in windows" \
    -e "repeat with t in tabs of w" \
    -e "if tty of t is \"$CURRENT_TTY\" then" \
    -e "close w" \
    -e "return" \
    -e "end if" \
    -e "end repeat" \
    -e "end repeat" \
    -e "end tell" >/dev/null 2>&1 </dev/null &
fi

exit "$STATUS"
