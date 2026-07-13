#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
RUN_TAG="${2:-20260714_pair01}"
SNAPSHOT="${ASHARE_COMPARE_SNAPSHOT:-/home/msj_team/Jacob/agent/shared/ashare_open_stocks_glm52_20260714}"
AGENT_ROOT="/home/msj_team/Jacob/agent"
FLOW_ROOT="${AGENT_ROOT}/MiroFlow"
MEM_ROOT="${AGENT_ROOT}/MiroMemSkill"
KEY_FILE="${AGENT_ROOT}/llm_key"
TOKEN_FILE="${TUSHARE_TOKEN_FILE:-${AGENT_ROOT}/tushare_token}"

if [[ "${MODE}" != "smoke" && "${MODE}" != "full" ]]; then
  echo "usage: $0 [smoke|full] [run_tag]" >&2
  exit 2
fi
if [[ ! -f "${SNAPSHOT}/manifest.json" ]]; then
  echo "missing comparison manifest: ${SNAPSHOT}/manifest.json" >&2
  exit 1
fi
if [[ ! -f "${KEY_FILE}" ]]; then
  echo "missing key file: ${KEY_FILE}" >&2
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
TASK_SHA="$(
  python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["artifacts"]["tasks"]["sha256"])' \
    "${SNAPSHOT}/manifest.json"
)"

SOURCE_DB="${SNAPSHOT}/ashare_pools_snapshot.db"
SERVER="${SNAPSHOT}/code/ashare_open_mcp_server.py"
TASK_FILE="${SNAPSHOT}/tasks/ashare_trader_open/standardized_data.jsonl"
ARM_DIR="${SNAPSHOT}/arms/${RUN_TAG}_${MODE}"
FLOW_DB="${ARM_DIR}/miroflow.db"
MEM_DB="${ARM_DIR}/miromemskill.db"
FLOW_OUT="${FLOW_ROOT}/logs/ashare_trader_open_flow_glm_${RUN_TAG}_${MODE}"
MEM_OUT="${MEM_ROOT}/logs/ashare_trader_open_memskill_glm_${RUN_TAG}_${MODE}"
REPORT="${FLOW_ROOT}/logs/tmpfiles/ashare_open_flow_vs_memskill_${RUN_TAG}_${MODE}.md"
RUN_MANIFEST="${ARM_DIR}/run_manifest.json"

for path in "${FLOW_OUT}" "${MEM_OUT}" "${ARM_DIR}"; do
  if [[ -e "${path}" ]]; then
    echo "refusing to reuse existing paired-run path: ${path}" >&2
    exit 1
  fi
done

echo "${SERVER_SHA}  ${SERVER}" | sha256sum -c -
echo "${DB_SHA}  ${SOURCE_DB}" | sha256sum -c -
echo "${TASK_SHA}  ${TASK_FILE}" | sha256sum -c -
CURRENT_MEM_SERVER="${MEM_ROOT}/src/tool/mcp_servers/ashare_open_mcp_server.py"
echo "${SERVER_SHA}  ${CURRENT_MEM_SERVER}" | sha256sum -c -

mkdir -p "${ARM_DIR}" "$(dirname "${REPORT}")"
cp --reflink=auto --preserve=mode,timestamps "${SOURCE_DB}" "${FLOW_DB}"
cp --reflink=auto --preserve=mode,timestamps "${SOURCE_DB}" "${MEM_DB}"

python3 - "${SNAPSHOT}" "${MODE}" "${RUN_TAG}" "${FLOW_DB}" "${MEM_DB}" \
  "${FLOW_OUT}" "${MEM_OUT}" "${REPORT}" > "${RUN_MANIFEST}" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

snapshot, mode, tag, flow_db, mem_db, flow_out, mem_out, report = sys.argv[1:]
base = json.loads((Path(snapshot) / "manifest.json").read_text())
payload = {
    "version": 1,
    "created_at": datetime.now(timezone.utc).isoformat(),
    "snapshot": str(Path(snapshot).resolve()),
    "snapshot_manifest_sha256": hashlib.sha256(
        (Path(snapshot) / "manifest.json").read_bytes()
    ).hexdigest(),
    "mode": mode,
    "run_tag": tag,
    "model": base["model"],
    "artifacts": base["artifacts"],
    "arms": {
        "miroflow": {"database": flow_db, "output": flow_out},
        "miromemskill": {"database": mem_db, "output": mem_out},
    },
    "report": report,
}
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY

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

WHITELIST=()
if [[ "${MODE}" == "smoke" ]]; then
  WHITELIST+=(
    "benchmark.data.whitelist=[ashare_open_trader_2024-07-01]"
  )
fi

echo "=== MiroFlow ${MODE} start $(date -Is) ==="
(
  cd "${FLOW_ROOT}"
  export ASHARE_OPEN_DB="${FLOW_DB}"
  export DATA_DIR="${SNAPSHOT}/tasks"
  conda run --no-capture-output -n Miro python main.py common-benchmark \
    --config_file_name=agent_ashare_trader_open_glm \
    "output_dir=${FLOW_OUT}" \
    "${WHITELIST[@]}"
)

echo "=== MiroMemSkill ${MODE} start $(date -Is) ==="
(
  cd "${MEM_ROOT}"
  export ASHARE_OPEN_DB="${MEM_DB}"
  export DATA_DIR="${SNAPSHOT}/tasks"
  conda run --no-capture-output -n Miro python main.py common-benchmark \
    --config_file_name=agent_ashare_trader_open_glm \
    "output_dir=${MEM_OUT}" \
    "${WHITELIST[@]}"
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
    --run "MiroMemSkill(GLM-5.2)=${MEM_OUT}" \
    --random-seeds 100 \
    --out "${REPORT}"
)

echo "=== paired ${MODE} done $(date -Is) ==="
echo "manifest -> ${RUN_MANIFEST}"
echo "report -> ${REPORT}"
