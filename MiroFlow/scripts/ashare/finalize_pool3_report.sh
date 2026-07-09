#!/usr/bin/env bash
# Wait for both Kimi A-share pool3 full runs to exit, then build the report.
set -u
AGENT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
FLOW="$AGENT_ROOT/MiroFlow"
MEM="$AGENT_ROOT/MiroMemSkill"

log() { echo "[$(date '+%F %T')] $*"; }

log "waiting for ashare_pool3_kimi runs to finish"
while pgrep -f "common-benchmark.*ashare_pool3_kimi" > /dev/null; do
    sleep 300
done
log "both runs exited"

cd "$FLOW"
uv run python scripts/ashare/backtest.py \
    --run baseline-kimi="$FLOW/logs/ashare_pool3_kimi" \
    --run memskill-v3-kimi="$MEM/logs/ashare_pool3_kimi" \
    --out "$FLOW/logs/ashare_report_pool3_kimi.md"
log "report -> $FLOW/logs/ashare_report_pool3_kimi.md"
