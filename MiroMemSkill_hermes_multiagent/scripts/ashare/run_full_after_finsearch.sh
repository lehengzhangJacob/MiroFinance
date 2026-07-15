#!/usr/bin/env bash
# Watchdog: wait for the FinSearchComp runs to finish, then run the full
# 72-task A-share prediction benchmark on both frameworks and build the report.
#
# Launch:  nohup bash MiroFlow/scripts/ashare/run_full_after_finsearch.sh \
#              > /tmp/ashare_full_watchdog.log 2>&1 &
set -u
AGENT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
FLOW="$AGENT_ROOT/MiroFlow"
MEM="$AGENT_ROOT/MiroMemSkill"
MIN_BALANCE_CNY=15

log() { echo "[$(date '+%F %T')] $*"; }

deepseek_balance() {
    local key
    key=$(grep -E '^DEEPSEEK_API_KEY=' "$AGENT_ROOT/llm_key" | cut -d= -f2 | tr -d '"')
    curl -sS -m 20 -H "Authorization: Bearer $key" \
        "https://api.deepseek.com/user/balance" \
        | python3 -c "import json,sys;print(json.load(sys.stdin)['balance_infos'][0]['total_balance'])" 2>/dev/null || echo "0"
}

log "watchdog started; waiting for FinSearchComp processes to exit"
while pgrep -f "finsearch_full391_ds|finsearch_memskill_ds" > /dev/null; do
    sleep 600
done
log "FinSearchComp processes gone"

while :; do
    bal=$(deepseek_balance)
    log "DeepSeek balance: CNY $bal"
    ok=$(python3 -c "print(1 if float('$bal' or 0) >= $MIN_BALANCE_CNY else 0)")
    [ "$ok" = "1" ] && break
    log "balance below CNY $MIN_BALANCE_CNY, waiting for recharge (retry in 30 min)"
    sleep 1800
done

log "launching baseline full run (72 tasks)"
cd "$FLOW"
uv run main.py common-benchmark \
    --config_file_name=agent_ashare_deepseek \
    benchmark=ashare-pred \
    output_dir=logs/ashare_full \
    >> /tmp/ashare_full_flow.log 2>&1
log "baseline full run finished"

log "launching memskill full run (72 tasks)"
cd "$MEM"
uv run main.py common-benchmark \
    --config_file_name=agent_ashare_memskill_deepseek \
    benchmark=ashare-pred \
    output_dir=logs/ashare_full \
    >> /tmp/ashare_full_mem.log 2>&1
log "memskill full run finished"

cd "$FLOW"
uv run python scripts/ashare/backtest.py \
    --run baseline="$FLOW/logs/ashare_full" \
    --run memskill="$MEM/logs/ashare_full" \
    --out "$FLOW/logs/ashare_report.md"
log "report written to $FLOW/logs/ashare_report.md"
