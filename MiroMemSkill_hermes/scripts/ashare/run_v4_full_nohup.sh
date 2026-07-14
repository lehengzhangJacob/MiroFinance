#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/msj_team/Jacob/agent/MiroMemSkill"
KEY_FILE="/home/msj_team/Jacob/agent/llm_key"
RUN_TAG="${1:-20260713_run1}"
OUT_DIR="${ROOT}/logs/ashare_trader_open_v4_glm_${RUN_TAG}"
NOHUP_DIR="${ROOT}/logs/nohup"
LOG_FILE="${NOHUP_DIR}/ashare_trader_open_v4_glm_${RUN_TAG}.log"
PID_FILE="${NOHUP_DIR}/ashare_trader_open_v4_glm_${RUN_TAG}.pid"
REPORT_FILE="${ROOT}/logs/tmpfiles/ashare_open_v4_${RUN_TAG}.md"

mkdir -p "${NOHUP_DIR}" "${ROOT}/logs/tmpfiles"
cd "${ROOT}"

if [[ ! -f "${KEY_FILE}" ]]; then
  echo "missing key file: ${KEY_FILE}" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "${KEY_FILE}"
set +a

echo "=== v4 full run start $(date -Is) ===" | tee -a "${LOG_FILE}"
echo "run_tag=${RUN_TAG}" | tee -a "${LOG_FILE}"
echo "out_dir=${OUT_DIR}" | tee -a "${LOG_FILE}"
echo "report=${REPORT_FILE}" | tee -a "${LOG_FILE}"

conda run --no-capture-output -n Miro python main.py common-benchmark \
  --config_file_name=agent_ashare_trader_open_v4_glm \
  "output_dir=${OUT_DIR}" \
  2>&1 | tee -a "${LOG_FILE}"

conda run --no-capture-output -n Miro python scripts/ashare/eval_open_trader.py \
  --run "Agent-Open-v1(GLM-5.2)=logs/ashare_trader_open_glm_20260713_full" \
  --run "Agent-Open-v2(GLM-5.2)=logs/ashare_trader_open_v2_glm_20260713_full" \
  --run "Agent-Open-v3(GLM-5.2)=logs/ashare_trader_open_v3_glm_20260713_full" \
  --run "Agent-Open-v4(GLM-5.2)=${OUT_DIR}" \
  --random-seeds 100 \
  --out "${REPORT_FILE}" \
  2>&1 | tee -a "${LOG_FILE}"

echo "=== v4 full run done $(date -Is) ===" | tee -a "${LOG_FILE}"
echo "report -> ${REPORT_FILE}" | tee -a "${LOG_FILE}"
