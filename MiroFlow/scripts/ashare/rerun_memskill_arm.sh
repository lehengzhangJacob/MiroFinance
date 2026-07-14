#!/usr/bin/env bash
# Rerun only the MiroMemSkill arm of the paired open-market comparison
# (after the universe-gating fix), then evaluate against an existing
# MiroFlow arm.
set -euo pipefail

RUN_TAG="${1:?new run_tag required (e.g. 20260714_memfix02)}"
FLOW_REF_TAG="${2:?existing flow run_tag to reuse (e.g. 20260714_memfix01)}"
MODE="full"
SNAPSHOT="${ASHARE_COMPARE_SNAPSHOT:-/home/msj_team/Jacob/agent/shared/ashare_open_stocks_glm52_20260714}"
AGENT_ROOT="/home/msj_team/Jacob/agent"
FLOW_ROOT="${AGENT_ROOT}/MiroFlow"
MEM_ROOT="${AGENT_ROOT}/MiroMemSkill"
KEY_FILE="${AGENT_ROOT}/llm_key"
TOKEN_FILE="${TUSHARE_TOKEN_FILE:-${AGENT_ROOT}/tushare_token}"

SOURCE_DB="${SNAPSHOT}/ashare_pools_snapshot.db"
SERVER="${SNAPSHOT}/code/ashare_open_mcp_server.py"
TASK_FILE="${SNAPSHOT}/tasks/ashare_trader_open/standardized_data.jsonl"
ARM_DIR="${SNAPSHOT}/arms/${RUN_TAG}_${MODE}"
MEM_DB="${ARM_DIR}/miromemskill.db"
FLOW_OUT="${FLOW_ROOT}/logs/ashare_trader_open_flow_glm_${FLOW_REF_TAG}_${MODE}"
MEM_OUT="${MEM_ROOT}/logs/ashare_trader_open_memskill_glm_${RUN_TAG}_${MODE}"
REPORT="${FLOW_ROOT}/logs/tmpfiles/ashare_open_flow_vs_memskill_${RUN_TAG}_${MODE}.md"

if [[ ! -d "${FLOW_OUT}" ]]; then
  echo "missing reusable flow arm output: ${FLOW_OUT}" >&2
  exit 1
fi
if [[ -e "${MEM_OUT}" || -e "${ARM_DIR}" ]]; then
  echo "refusing to reuse existing paths: ${MEM_OUT} / ${ARM_DIR}" >&2
  exit 1
fi

SERVER_SHA="$(
  python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["artifacts"]["server"]["sha256"])' \
    "${SNAPSHOT}/manifest.json"
)"
DB_SHA="$(
  python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["artifacts"]["database"]["sha256"])' \
    "${SNAPSHOT}/manifest.json"
)"
echo "${SERVER_SHA}  ${SERVER}" | sha256sum -c -
echo "${DB_SHA}  ${SOURCE_DB}" | sha256sum -c -
echo "${SERVER_SHA}  ${MEM_ROOT}/src/tool/mcp_servers/ashare_open_mcp_server.py" | sha256sum -c -

mkdir -p "${ARM_DIR}" "$(dirname "${REPORT}")"
cp --reflink=auto --preserve=mode,timestamps "${SOURCE_DB}" "${MEM_DB}"

set -a
# shellcheck disable=SC1090
source "${KEY_FILE}"
set +a
export ASHARE_COMPARE_SNAPSHOT="${SNAPSHOT}"
export ASHARE_OPEN_TASK_ROOT="${SNAPSHOT}/tasks"
export ASHARE_OPEN_SERVER="${SERVER}"
export ASHARE_OPEN_SERVER_SHA256="${SERVER_SHA}"
export TUSHARE_TOKEN_FILE="${TOKEN_FILE}"
export CHINESE_CONTEXT="${CHINESE_CONTEXT:-true}"
export ASHARE_TRADER_RUN_ID="open_${RUN_TAG}_${MODE}"
export MEM0_TELEMETRY="false"

echo "=== MiroMemSkill rerun ${RUN_TAG} start $(date -Is) ==="
(
  cd "${MEM_ROOT}"
  export ASHARE_OPEN_DB="${MEM_DB}"
  export DATA_DIR="${SNAPSHOT}/tasks"
  conda run --no-capture-output -n Miro python main.py common-benchmark \
    --config_file_name=agent_ashare_trader_open_memskill_glm \
    "output_dir=${MEM_OUT}"
)

echo "=== paired evaluation start $(date -Is) ==="
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

echo "=== rerun ${RUN_TAG} done $(date -Is) ==="
echo "report -> ${REPORT}"
