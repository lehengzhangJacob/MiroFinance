#!/usr/bin/env bash
# Monitor FinSearchComp 391-task run and regenerate baseline report when complete.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="/tmp/finsearch_full391.log"
TARGET=391
INTERVAL="${INTERVAL:-300}"

cd "$ROOT"
export PATH="$HOME/.local/bin:$PATH"

echo "[watch] Monitoring logs/finsearch_full391 (target ${TARGET} tasks), interval ${INTERVAL}s"

while true; do
  DONE=$(ls logs/finsearch_full391/*_attempt_1.json 2>/dev/null | wc -l)
  TS=$(date '+%Y-%m-%d %H:%M:%S')
  echo "[$TS] progress: ${DONE}/${TARGET}"

  if [[ "$DONE" -ge "$TARGET" ]]; then
    echo "[watch] All tasks complete. Generating final baseline report..."
    uv run python scripts/generate_baseline_report.py
    echo "[watch] Done. Report: logs/baseline_report.md"
    exit 0
  fi

  if ! pgrep -f "output_dir=logs/finsearch_full391" >/dev/null 2>&1; then
    echo "[watch] WARNING: benchmark process not found. Last log lines:"
    tail -20 "$LOG" 2>/dev/null || true
    if [[ "$DONE" -lt "$TARGET" ]]; then
      echo "[watch] Run may have crashed. Check /tmp/finsearch_full391.log"
    fi
  fi

  sleep "$INTERVAL"
done
