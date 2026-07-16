#!/usr/bin/env bash
# Walk-forward evolution chain R2 -> R3 -> R4 on the frozen 24-month snapshot.
#
# Start in the background:
#   nohup bash scripts/ashare/run_wf24m_r2_r4_chain_nohup.sh 20260716 \
#     > .evolution/nohup/wf24m_r2_r4_20260716.log 2>&1 &
#
# Protocol (24 months, 2024-07 .. 2026-06, indexed chronologically):
#   every round: train = 12 months, dev = 6 months, holdout = 2 months
#   R2: skip 0 -> train 2024-07..2025-06, dev 2025-07..12, holdout 2026-01..02
#   R3: skip 2 -> train 2024-09..2025-08, dev 2025-09..2026-02, holdout 2026-03..04
#   R4: skip 4 -> train 2024-11..2025-10, dev 2025-11..2026-04, holdout 2026-05..06
#
# Within this chain no month is used twice for a promotion decision; a
# round's holdout rolls into the next round's train/dev only after the
# decision is made (rolling-origin walk-forward). Caveat for reporting:
# 2026H1 was already consumed once by R1's 6-month holdout, so this chain is
# sequentially clean but not virgin data relative to R1.
#
# Promotion rule (stricter than the retired formal24m chain): hard gates
# pass AND dev score > 0 AND holdout score > 0. Otherwise keep active.
set -Eeuo pipefail

ROOT="/home/msj_team/Jacob/agent/MiroMemSkill_hermes"
PYTHON="/home/msj_team/.conda/envs/Miro/bin/python"
SNAPSHOT="/home/msj_team/Jacob/agent/shared/ashare_open_stocks_glm52_24m_20260715"

RUN_TAG="${1:-$(date +%Y%m%d)}"
CANDIDATES="${CANDIDATES:-3}"
TRAIN_MONTHS=12
DEV_MONTHS=6
HOLDOUT_MONTHS=2
STEP_MONTHS=2

NOHUP_DIR="${ROOT}/.evolution/nohup"
CHAIN_NAME="wf24m_r2_r4_${RUN_TAG}"
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

# One chain at a time; repeated launches must not duplicate rounds.
exec 9>"${LOCK_FILE}"
flock -n 9 || die "another ${CHAIN_NAME} chain is already running"
printf '%s\n' "$$" >"${PID_FILE}"

cleanup() {
  local rc=$?
  rm -f "${PID_FILE}"
  if [[ ${rc} -eq 0 ]]; then
    log "CHAIN DONE: wf24m R2 -> R3 -> R4 (${RUN_TAG})"
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

candidate = str(manifest["skill_sha256"])
rec = registry["candidates"].get(candidate, {})
dev = rec.get("reports", {}).get("fitness_dev", {})

holdout_passed = bool(report.get("gates", {}).get("passed", False))
holdout_score = float(report.get("score", float("-inf")))
dev_score = float(dev.get("score", float("-inf")))

print("true" if holdout_passed else "false")
print(f"{holdout_score:.6f}")
print(f"{dev_score:.6f}")
print(candidate)
print(str(registry["active_digest"]))
PY
  )

  [[ ${#result[@]} -eq 5 ]] || die "${run_id}: could not parse promotion state"
  local passed="${result[0]}"
  local holdout_score="${result[1]}"
  local dev_score="${result[2]}"
  local candidate="${result[3]}"
  local active="${result[4]}"
  local candidate_short="${candidate:0:12}"

  log "${run_id}: gates=${passed} holdout_score=${holdout_score} dev_score=${dev_score} candidate=${candidate_short}"

  if [[ "${candidate}" == "${active}" ]]; then
    log "${run_id}: candidate is already active"
    return 0
  fi

  if [[ "${passed}" == "true" ]] && \
     "${PYTHON}" - "${dev_score}" "${holdout_score}" <<'PY'
import sys
dev, holdout = float(sys.argv[1]), float(sys.argv[2])
raise SystemExit(0 if dev > 0.0 and holdout > 0.0 else 1)
PY
  then
    log "${run_id}: promoting ${candidate_short} (dev>0 and holdout>0)"
    "${PYTHON}" scripts/ashare/run_skill_evolution.py promote \
      --candidate="${candidate_short}" --run_id="${run_id}"
  else
    log "${run_id}: promotion rejected (need gates + dev>0 + holdout>0); active unchanged"
  fi
}

run_round() {
  local round="$1"
  local skip="$2"
  local run_id="wf24m_r${round}_${RUN_TAG}"
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

  log "${run_id}: START (skip=${skip}, base=active, candidates=${CANDIDATES})"
  "${PYTHON}" scripts/ashare/run_skill_evolution.py \
    --snapshot="${SNAPSHOT}" \
    --train_months="${TRAIN_MONTHS}" \
    --dev_months="${DEV_MONTHS}" \
    --holdout_months="${HOLDOUT_MONTHS}" \
    --skip_months="${skip}" \
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

log "CHAIN START: wf24m walk-forward R2 -> R3 -> R4 (tag=${RUN_TAG})"
log "snapshot=${SNAPSHOT} split=${TRAIN_MONTHS}/${DEV_MONTHS}/${HOLDOUT_MONTHS} step=${STEP_MONTHS}"
run_round 2 0
run_round 3 "${STEP_MONTHS}"
run_round 4 $((STEP_MONTHS * 2))
