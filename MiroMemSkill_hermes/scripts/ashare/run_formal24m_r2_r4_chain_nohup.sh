#!/usr/bin/env bash
# LEGACY: per-round holdout + intermediate promotion on the fixed 12/6/6 split.
# Prefer the sealed protocol instead:
#   scripts/ashare/run_formal24m_devonly_r2_r4_chain_nohup.sh
#
# Wait for the already-running R2, then run R3 and R4 sequentially.
#
# Start in the background:
#   nohup bash scripts/ashare/run_formal24m_r2_r4_chain_nohup.sh 20260716 \
#     > .evolution/nohup/formal24m_r2_r4_20260716.log 2>&1 &
#
# A round is promoted only when its holdout passes the hard gates AND has a
# positive paired ranking score. Otherwise the next round keeps the current
# active skill.
set -Eeuo pipefail

ROOT="/home/msj_team/Jacob/agent/MiroMemSkill_hermes"
PYTHON="/home/msj_team/.conda/envs/Miro/bin/python"
SNAPSHOT="/home/msj_team/Jacob/agent/shared/ashare_open_stocks_glm52_24m_20260715"

RUN_TAG="${1:-$(date +%Y%m%d)}"
R2_RUN_ID="${R2_RUN_ID:-formal24m_r2_20260715}"
R3_RUN_ID="${R3_RUN_ID:-formal24m_r3_${RUN_TAG}}"
R4_RUN_ID="${R4_RUN_ID:-formal24m_r4_${RUN_TAG}}"
CANDIDATES="${CANDIDATES:-3}"
POLL_SECONDS="${POLL_SECONDS:-60}"

NOHUP_DIR="${ROOT}/.evolution/nohup"
CHAIN_NAME="formal24m_r2_r4_${RUN_TAG}"
PID_FILE="${NOHUP_DIR}/${CHAIN_NAME}.pid"
LOCK_FILE="${NOHUP_DIR}/${CHAIN_NAME}.lock"

mkdir -p "${NOHUP_DIR}"
cd "${ROOT}"

# GLM and Tushare are domestic services; inherited local proxies make them
# less reliable. run_skill_evolution.py loads credentials from the local key
# files itself.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
unset all_proxy ALL_PROXY no_proxy NO_PROXY
export PATH="/home/msj_team/.conda/envs/Miro/bin:${PATH}"
export PYTHONUNBUFFERED=1

log() {
  printf '=== [%s] %s ===\n' "$(date '+%F %T')" "$*"
}

die() {
  log "FATAL: $*"
  exit 1
}

[[ -x "${PYTHON}" ]] || die "missing Python executable: ${PYTHON}"
[[ -d "${SNAPSHOT}" ]] || die "missing frozen snapshot: ${SNAPSHOT}"

# Hold an advisory lock for the full chain so repeated launches cannot create
# duplicate R3/R4 jobs.
exec 9>"${LOCK_FILE}"
flock -n 9 || die "another ${CHAIN_NAME} chain is already running"
printf '%s\n' "$$" >"${PID_FILE}"

cleanup() {
  local rc=$?
  rm -f "${PID_FILE}"
  if [[ ${rc} -eq 0 ]]; then
    log "CHAIN DONE: ${R2_RUN_ID} -> ${R3_RUN_ID} -> ${R4_RUN_ID}"
  else
    log "CHAIN STOPPED with exit ${rc}"
  fi
}
trap cleanup EXIT

round_process_running() {
  local run_id="$1"
  pgrep -f "run_skill_evolution.py .*--run_id=${run_id}([[:space:]]|$)" \
    >/dev/null 2>&1
}

holdout_report() {
  local run_id="$1"
  printf '%s/.evolution/runs/%s/reports/fitness_holdout.json\n' \
    "${ROOT}" "${run_id}"
}

promote_if_qualified() {
  local run_id="$1"
  local report manifest
  report="$(holdout_report "${run_id}")"
  manifest="${ROOT}/.evolution/runs/${run_id}/arms/candidate/arm_manifest.json"

  if [[ ! -f "${report}" ]]; then
    log "${run_id}: no holdout report; keep current active skill"
    return 0
  fi
  [[ -f "${manifest}" ]] || die "${run_id}: missing candidate arm manifest"

  local -a result
  mapfile -t result < <(
    "${PYTHON}" - "${report}" "${manifest}" \
      "${ROOT}/.evolution/registry.json" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
manifest = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
registry = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))

passed = bool(report.get("gates", {}).get("passed", False))
score = float(report.get("score", float("-inf")))
candidate = str(manifest["skill_sha256"])
active = str(registry["active_digest"])

print("true" if passed else "false")
print(f"{score:.6f}")
print(candidate)
print(active)
PY
  )

  [[ ${#result[@]} -eq 4 ]] || die "${run_id}: could not parse promotion state"
  local passed="${result[0]}"
  local score="${result[1]}"
  local candidate="${result[2]}"
  local active="${result[3]}"
  local candidate_short="${candidate:0:12}"

  log "${run_id}: holdout gates=${passed}, score=${score}, candidate=${candidate_short}"

  if [[ "${candidate}" == "${active}" ]]; then
    log "${run_id}: candidate is already active"
    return 0
  fi

  if [[ "${passed}" == "true" ]] && \
     "${PYTHON}" - "${score}" <<'PY'
import sys
raise SystemExit(0 if float(sys.argv[1]) > 0.0 else 1)
PY
  then
    log "${run_id}: promoting ${candidate_short}"
    "${PYTHON}" scripts/ashare/run_skill_evolution.py promote \
      --candidate="${candidate_short}" --run_id="${run_id}"
  else
    log "${run_id}: promotion rejected; active skill unchanged"
  fi
}

wait_for_r2() {
  local report
  report="$(holdout_report "${R2_RUN_ID}")"

  if round_process_running "${R2_RUN_ID}"; then
    log "waiting for existing ${R2_RUN_ID}"
    while round_process_running "${R2_RUN_ID}"; do
      sleep "${POLL_SECONDS}"
    done
  else
    log "${R2_RUN_ID} is not running; validating existing result"
  fi

  [[ -f "${report}" ]] || die "${R2_RUN_ID} ended without fitness_holdout.json"
  promote_if_qualified "${R2_RUN_ID}"
}

run_round() {
  local run_id="$1"
  local run_dir="${ROOT}/.evolution/runs/${run_id}"
  local report
  report="$(holdout_report "${run_id}")"

  if round_process_running "${run_id}"; then
    die "${run_id} is already running outside this chain"
  fi

  if [[ -f "${report}" ]]; then
    log "${run_id}: completed result already exists; not rerunning"
    promote_if_qualified "${run_id}"
    return 0
  fi

  if [[ -d "${run_dir}" ]]; then
    die "${run_id}: partial run directory exists without a holdout report"
  fi

  log "${run_id}: START (base=active, candidates=${CANDIDATES})"
  "${PYTHON}" scripts/ashare/run_skill_evolution.py \
    --snapshot="${SNAPSHOT}" \
    --train_months=12 \
    --dev_months=6 \
    --holdout_months=6 \
    full \
    --run_id="${run_id}" \
    --n="${CANDIDATES}" \
    --base=active \
    --cleanup_db=True
  log "${run_id}: process finished"

  # full exits successfully without holdout when every dev candidate fails.
  # That is a completed negative round, so keep active and continue.
  promote_if_qualified "${run_id}"
}

log "CHAIN START: ${R2_RUN_ID} -> ${R3_RUN_ID} -> ${R4_RUN_ID}"
log "snapshot=${SNAPSHOT}"
wait_for_r2
run_round "${R3_RUN_ID}"
run_round "${R4_RUN_ID}"
