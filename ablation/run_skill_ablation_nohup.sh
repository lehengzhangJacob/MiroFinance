#!/usr/bin/env bash
# Memory-ON skill ablation (baseline vs R1-best, 24 months) using own_glm3.
#
#   nohup bash ablation/run_skill_ablation_nohup.sh \
#     > ablation/logs/mem_ablation_24m.log 2>&1 &
set -Eeuo pipefail

AGENT_ROOT="/home/msj_team/Jacob/agent"
ABLATION="${AGENT_ROOT}/ablation"
PYTHON="/home/msj_team/.conda/envs/Miro/bin/python"
RUN_ID="${1:-mem_ablation_24m}"
LOG_DIR="${ABLATION}/logs"
PID_FILE="${LOG_DIR}/${RUN_ID}.pid"
LOCK_FILE="${LOG_DIR}/${RUN_ID}.lock"

mkdir -p "${LOG_DIR}"
cd "${AGENT_ROOT}"

# Domestic GLM: do not use local HTTP proxies for this job.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
unset all_proxy ALL_PROXY no_proxy NO_PROXY
export PATH="/home/msj_team/.conda/envs/Miro/bin:${PATH}"
export PYTHONUNBUFFERED=1

[[ -x "${PYTHON}" ]] || { echo "missing Python: ${PYTHON}"; exit 1; }
[[ -f "${AGENT_ROOT}/own_glm3" ]] || { echo "missing own_glm3 key file"; exit 1; }
[[ -d "${AGENT_ROOT}/shared/ashare_open_stocks_glm52_24m_20260715" ]] || {
  echo "missing 24m snapshot"; exit 1;
}
curl -sf --max-time 5 http://127.0.0.1:6333/collections >/dev/null || {
  echo "Qdrant is not reachable on 127.0.0.1:6333 (required for memory)"; exit 1;
}

exec 9>"${LOCK_FILE}"
flock -n 9 || { echo "ablation ${RUN_ID} already running"; exit 1; }
printf '%s\n' "$$" >"${PID_FILE}"

cleanup() {
  rm -f "${PID_FILE}"
}
trap cleanup EXIT

echo "=== [$(date '+%F %T')] START memory-ON ablation run_id=${RUN_ID} key=own_glm3 ==="
"${PYTHON}" "${ABLATION}/run_skill_ablation.py" --run_id="${RUN_ID}" --cleanup_db=True
echo "=== [$(date '+%F %T')] DONE ${RUN_ID} ==="
