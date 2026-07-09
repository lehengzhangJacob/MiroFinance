#!/usr/bin/env bash
# Wait for the pool3 MemSkill-v4 rerun to exit, then rebuild the 3-arm report.
set -u
AGENT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
FLOW="$AGENT_ROOT/MiroFlow"
MEM="$AGENT_ROOT/MiroMemSkill"
PYTHON="/home/msj_team/.conda/envs/Miro/bin/python"

log() { echo "[$(date '+%F %T')] $*"; }

log "waiting for ashare_pool3_v4_kimi run to finish"
sleep 60  # let the launcher's process settle before first check
while pgrep -f "common-benchmark.*ashare_pool3_v4_kimi" > /dev/null; do
    sleep 300
done
log "v4 run exited"

cd "$FLOW"
"$PYTHON" scripts/ashare/backtest.py \
    --run baseline-kimi="$FLOW/logs/ashare_pool3_kimi" \
    --run memskill-v3-kimi="$MEM/logs/ashare_pool3_kimi" \
    --run memskill-v4-kimi="$MEM/logs/ashare_pool3_v4_kimi" \
    --out "$FLOW/logs/ashare_report_pool3_kimi.md"
log "report -> $FLOW/logs/ashare_report_pool3_kimi.md"
