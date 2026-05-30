#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="/Users/zs/Documents/to_edit"
PROJECT_DIR="/Users/zs/Documents/codex_projects/comic-text-detector"
DETECTOR="$PROJECT_DIR/new_detect_folder.py"
PYTHON="$PROJECT_DIR/.venv/bin/python"

series_list=(
  "昭和的钱进球场"
  "夏之介的青春"
  "江川与西本"
)

cd "$BASE_DIR"

if [[ ! -x "$PYTHON" ]]; then
  echo "[error] venv python not found: $PYTHON" >&2
  echo "Run this first: cd \"$PROJECT_DIR\" && python3 -m venv .venv && source .venv/bin/activate" >&2
  exit 1
fi

for series in "${series_list[@]}"; do
  if [[ ! -d "$series" ]]; then
    echo "[skip] not found: $BASE_DIR/$series" >&2
    continue
  fi

  find "$series" -mindepth 1 -maxdepth 1 -type d -print0 \
    | sort -zV \
    | while IFS= read -r -d '' dir; do
        echo "===== running: $BASE_DIR/$dir ====="
        "$PYTHON" "$DETECTOR" "$BASE_DIR/$dir"
      done
done
