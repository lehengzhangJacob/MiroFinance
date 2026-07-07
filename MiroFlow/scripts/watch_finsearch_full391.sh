#!/usr/bin/env bash
# Monitor a FinSearchComp 391-task run and regenerate the baseline report on completion.
# Usage: RUN_DIR=logs/finsearch_full391_ds RUN_LOG=/tmp/finsearch_full391_ds.log ./watch_finsearch_full391.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN_DIR="${RUN_DIR:-logs/finsearch_full391_ds}"
LOG="${RUN_LOG:-/tmp/finsearch_full391_ds.log}"
TARGET=391
INTERVAL="${INTERVAL:-300}"

cd "$ROOT"
export PATH="$HOME/.local/bin:$PATH"

echo "[watch] Monitoring ${RUN_DIR} (target ${TARGET} tasks), interval ${INTERVAL}s"

while true; do
  DONE=$(ls "${RUN_DIR}"/*_attempt_1.json 2>/dev/null | wc -l)
  TS=$(date '+%Y-%m-%d %H:%M:%S')
  echo "[$TS] progress: ${DONE}/${TARGET}"

  if [[ "$DONE" -ge "$TARGET" ]]; then
    echo "[watch] All tasks complete. Generating final baseline report..."
    uv run python scripts/generate_baseline_report.py --finsearch-dir "${RUN_DIR}"
    echo "[watch] Done. Report: logs/baseline_report.md"
    exit 0
  fi

  if ! pgrep -f "output_dir=${RUN_DIR}" >/dev/null 2>&1; then
    echo "[watch] WARNING: benchmark process not found. Last log lines:"
    tail -20 "$LOG" 2>/dev/null || true
    if [[ "$DONE" -lt "$TARGET" ]]; then
      echo "[watch] Run may have crashed. Check ${LOG}"
    fi
  fi

  sleep "$INTERVAL"
done
