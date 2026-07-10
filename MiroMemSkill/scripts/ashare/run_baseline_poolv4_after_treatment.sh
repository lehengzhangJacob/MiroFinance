#!/usr/bin/env bash
# Watchdog: when the pool-v4 treatment run (agent_ashare_memskill_poolv4_kimi)
# exits, launch the matched MiroFlow control arm (same 16-stock tasks, same
# ashare tools incl. ashare_ml_signal, no memory/skills), then build the
# combined comparison report.
#
# Launch:  nohup bash scripts/ashare/run_baseline_poolv4_after_treatment.sh \
#              > logs/tmpfiles/poolv4_baseline_watchdog.log 2>&1 &
set -u
cd "$(dirname "$0")/../.."   # MiroMemSkill/
PY=/home/msj_team/.conda/envs/Miro/bin/python

log() { echo "[$(date '+%F %T')] $*"; }

log "waiting for pool-v4 treatment to finish"
while pgrep -f "agent_ashare_memskill_poolv4_kimi" > /dev/null; do
    sleep 300
done
log "treatment gone; launching MiroFlow baseline (192 tasks)"

cd ../MiroFlow
mkdir -p logs/tmpfiles
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
    -u all_proxy -u ALL_PROXY -u no_proxy -u NO_PROXY \
    uv run main.py common-benchmark \
    --config_file_name=agent_ashare_kimi \
    benchmark=ashare-pred \
    output_dir=logs/ashare_poolv4_baseline \
    benchmark.execution.max_concurrent=6 \
    >> logs/tmpfiles/poolv4_baseline.log 2>&1
log "baseline finished"

cd ../MiroMemSkill
log "building combined pool-v4 comparison report"
"$PY" scripts/ashare/backtest.py \
    --run baseline=../MiroFlow/logs/ashare_poolv4_baseline \
    --run memskill=logs/ashare_poolv4_memskill \
    --out logs/ashare_report_poolv4.md
log "report -> MiroMemSkill/logs/ashare_report_poolv4.md"
