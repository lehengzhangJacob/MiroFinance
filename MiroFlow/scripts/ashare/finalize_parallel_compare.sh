#!/usr/bin/env bash
set -euo pipefail

RUN_TAG="${1:?run_tag required}"
MODE="${2:-full}"
COMPARE_PID="${3:-}"
SNAPSHOT="${ASHARE_COMPARE_SNAPSHOT:-/home/msj_team/Jacob/agent/shared/ashare_open_stocks_glm52_20260714}"
AGENT_ROOT="/home/msj_team/Jacob/agent"
FLOW_ROOT="${AGENT_ROOT}/MiroFlow"
MEM_ROOT="${AGENT_ROOT}/MiroMemSkill"
KEY_FILE="${AGENT_ROOT}/llm_key"
TOKEN_FILE="${TUSHARE_TOKEN_FILE:-${AGENT_ROOT}/tushare_token}"
ARM_DIR="${SNAPSHOT}/arms/${RUN_TAG}_${MODE}"
FLOW_OUT="${FLOW_ROOT}/logs/ashare_trader_open_flow_glm_${RUN_TAG}_${MODE}"
MEM_OUT="${MEM_ROOT}/logs/ashare_trader_open_memskill_glm_${RUN_TAG}_${MODE}"
REPORT="${FLOW_ROOT}/logs/tmpfiles/ashare_open_flow_vs_memskill_${RUN_TAG}_${MODE}.md"
TASK_FILE="${SNAPSHOT}/tasks/ashare_trader_open/standardized_data.jsonl"
SOURCE_DB="${SNAPSHOT}/ashare_pools_snapshot.db"
NOHUP_DIR="${FLOW_ROOT}/logs/nohup"
LOG="${NOHUP_DIR}/open_trader_compare_${RUN_TAG}_${MODE}_finalize.log"

mkdir -p "${NOHUP_DIR}" "$(dirname "${REPORT}")"

log() {
  echo "[$(date -Is)] $*" | tee -a "${LOG}"
}

is_running() {
  local pattern="$1"
  pgrep -af "${pattern}" 2>/dev/null | rg -qv 'pgrep|rg |finalize_parallel_compare'
}

log "finalize watcher started for ${RUN_TAG}_${MODE}"

FLOW_PATTERN="output_dir=${FLOW_OUT}"
MEM_PATTERN="output_dir=${MEM_OUT}"

if [[ -n "${COMPARE_PID}" ]]; then
  log "waiting for Flow arm while compare pid=${COMPARE_PID} is alive"
  while is_running "${FLOW_PATTERN}"; do
    if ! kill -0 "${COMPARE_PID}" 2>/dev/null; then
      break
    fi
    sleep 10
  done
  if kill -0 "${COMPARE_PID}" 2>/dev/null; then
    log "Flow finished; stopping compare wrapper pid=${COMPARE_PID} to avoid duplicate MemSkill launch"
    kill "${COMPARE_PID}" 2>/dev/null || true
    sleep 2
  fi
else
  log "waiting for Flow arm to finish"
  while is_running "${FLOW_PATTERN}"; do sleep 15; done
fi

log "waiting for MemSkill arm to finish"
while is_running "${MEM_PATTERN}"; do sleep 30; done

set -a
# shellcheck disable=SC1090
source "${KEY_FILE}"
set +a
export ASHARE_COMPARE_SNAPSHOT="${SNAPSHOT}"
export TUSHARE_TOKEN_FILE="${TOKEN_FILE}"

log "running paired evaluation"
(
  cd "${FLOW_ROOT}"
  conda run --no-capture-output -n Miro python \
    scripts/ashare/eval_open_trader.py \
    --snapshot "${SNAPSHOT}" \
    --tasks "${TASK_FILE}" \
    --db "${SOURCE_DB}" \
    --run "MiroFlow-Plain(GLM-5.2)=${FLOW_OUT}" \
    --run "MiroMemSkill-MemSkill(GLM-5.2)=${MEM_OUT}" \
    --random-seeds 100 \
    --out "${REPORT}"
)

log "paired ${MODE} done"
log "report -> ${REPORT}"
