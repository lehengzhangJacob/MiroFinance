#!/usr/bin/env bash
# Dev-only multi-round evolution R2 -> R3 -> R4, then ONE sealed multi-seed holdout.
#
# Start in the background:
#   nohup bash scripts/ashare/run_formal24m_devonly_r2_r4_chain_nohup.sh 20260716 \
#     > .evolution/nohup/formal24m_devonly_r2_r4_20260716.log 2>&1 &
#
# Protocol (fixed 12/6/6 on the frozen 24-month snapshot):
#   - R1 is NOT re-run. Parent / production active must already be R1
#     (default: 3aebb813bd33). Its first-time holdout remains the primary
#     formal evidence.
#   - R2/R3/R4 only call ``search`` (train + propose + dev). Holdout is never
#     opened during these rounds; the production skill file is never rewritten.
#   - Each round advances a temporary champion pointer only when the round's
#     best candidate passes hard gates AND has positive dev ranking score
#     relative to the current champion. Otherwise the champion is unchanged.
#   - After R4, if the temporary champion differs from R1, open holdout once
#     with SEEDS independent rollouts (default 3). Promote only when every
#     seed passes gates AND the mean ranking score is > 0.
#
# Caveat for reporting: 2026H1 was already used by R1's promotion, so this
# final holdout is an exploratory "did further search beat R1 on the same
# sealed months" check, not a virgin sealed-set verdict.
set -Eeuo pipefail

ROOT="/home/msj_team/Jacob/agent/MiroMemSkill_hermes"
PYTHON="/home/msj_team/.conda/envs/Miro/bin/python"
SNAPSHOT="/home/msj_team/Jacob/agent/shared/ashare_open_stocks_glm52_24m_20260715"

RUN_TAG="${1:-$(date +%Y%m%d)}"
R1_SHORT="${R1_SHORT:-3aebb813bd33}"
CANDIDATES="${CANDIDATES:-3}"
SEEDS="${SEEDS:-3}"
TRAIN_MONTHS=12
DEV_MONTHS=6
HOLDOUT_MONTHS=6

NOHUP_DIR="${ROOT}/.evolution/nohup"
CHAIN_NAME="formal24m_devonly_r2_r4_${RUN_TAG}"
PID_FILE="${NOHUP_DIR}/${CHAIN_NAME}.pid"
LOCK_FILE="${NOHUP_DIR}/${CHAIN_NAME}.lock"
STATE_FILE="${NOHUP_DIR}/${CHAIN_NAME}.state.json"
CHAIN_HOLD_ID="formal24m_devonly_finalhold_${RUN_TAG}"

mkdir -p "${NOHUP_DIR}"
cd "${ROOT}"

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

exec 9>"${LOCK_FILE}"
flock -n 9 || die "another ${CHAIN_NAME} chain is already running"
printf '%s\n' "$$" >"${PID_FILE}"

cleanup() {
  local rc=$?
  rm -f "${PID_FILE}"
  if [[ ${rc} -eq 0 ]]; then
    log "CHAIN DONE: dev-only R2->R3->R4 + final ${SEEDS}-seed holdout (${RUN_TAG})"
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

require_active_is_r1() {
  local active
  active="$("${PYTHON}" - "${ROOT}/.evolution/registry.json" "${R1_SHORT}" <<'PY'
import json, sys
from pathlib import Path
reg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
want = sys.argv[2]
active = reg["active_digest"]
ok = active.startswith(want) or any(
    rec.get("short_id") == want and dig == active
    for dig, rec in reg["candidates"].items()
)
print(active[:12])
raise SystemExit(0 if ok else 1)
PY
)" || die "active skill is ${active:-unknown}, expected R1 ${R1_SHORT}; refuse to start"
  log "active skill confirmed R1: ${active}"
}

write_state() {
  local champion="$1"
  local round="$2"
  "${PYTHON}" - "${STATE_FILE}" "${champion}" "${round}" "${R1_SHORT}" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
prev = {}
if path.exists():
    prev = json.loads(path.read_text(encoding="utf-8"))
prev.update({
    "champion": sys.argv[2],
    "last_round": int(sys.argv[3]),
    "r1": sys.argv[4],
})
path.write_text(json.dumps(prev, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
}

read_champion() {
  if [[ -f "${STATE_FILE}" ]]; then
    "${PYTHON}" -c "import json; print(json.load(open('${STATE_FILE}'))['champion'])"
  else
    printf '%s\n' "${R1_SHORT}"
  fi
}

run_search_round() {
  local round="$1"
  local champion="$2"
  local run_id="formal24m_devonly_r${round}_${RUN_TAG}"
  local run_dir="${ROOT}/.evolution/runs/${run_id}"
  local summary="${run_dir}/reports/search_summary.json"

  if round_process_running "${run_id}"; then
    die "${run_id} is already running outside this chain"
  fi

  if [[ -f "${summary}" ]]; then
    log "${run_id}: search summary exists; not rerunning"
  else
    if [[ -d "${run_dir}" ]]; then
      die "${run_id}: partial run directory exists without search_summary.json"
    fi
    log "${run_id}: START search (base=${champion}, candidates=${CANDIDATES}, no holdout)"
    "${PYTHON}" scripts/ashare/run_skill_evolution.py \
      --snapshot="${SNAPSHOT}" \
      --train_months="${TRAIN_MONTHS}" \
      --dev_months="${DEV_MONTHS}" \
      --holdout_months="${HOLDOUT_MONTHS}" \
      search \
      --run_id="${run_id}" \
      --n="${CANDIDATES}" \
      --base="${champion}" \
      --cleanup_db=True
    log "${run_id}: search finished"
  fi

  [[ -f "${summary}" ]] || die "${run_id}: missing search_summary.json"

  local -a result
  mapfile -t result < <(
    "${PYTHON}" - "${summary}" "${champion}" <<'PY'
import json, sys
from pathlib import Path
summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
prev = sys.argv[2]
status = summary.get("status")
best = summary.get("best")
score = summary.get("dev_score")
if status != "dev_complete" or best is None or score is None:
    print(prev)
    print("keep")
    print("nan")
    print("none")
    raise SystemExit(0)
score_f = float(score)
if score_f > 0.0:
    print(best)
    print("advance")
    print(f"{score_f:.6f}")
    print(best)
else:
    print(prev)
    print("keep")
    print(f"{score_f:.6f}")
    print(best)
PY
  )

  local new_champion="${result[0]}"
  local decision="${result[1]}"
  local score="${result[2]}"
  local best="${result[3]}"
  log "${run_id}: best=${best} dev_score=${score} decision=${decision} champion=${new_champion}"
  write_state "${new_champion}" "${round}"
  printf '%s\n' "${new_champion}"
}

final_holdout_and_maybe_promote() {
  local champion="$1"
  local report="${ROOT}/.evolution/runs/${CHAIN_HOLD_ID}/reports/fitness_holdout_multiseed.json"

  if "${PYTHON}" - "${ROOT}/.evolution/registry.json" "${champion}" "${R1_SHORT}" <<'PY'
import json, sys
from pathlib import Path
reg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
cand, r1 = sys.argv[2], sys.argv[3]

def resolve(x: str) -> str:
    if x in reg["candidates"]:
        return x
    hits = [
        d
        for d, rec in reg["candidates"].items()
        if d.startswith(x) or rec.get("short_id") == x
    ]
    if len(hits) != 1:
        raise SystemExit(2)
    return hits[0]

raise SystemExit(0 if resolve(cand) == resolve(r1) else 1)
PY
  then
    log "temporary champion still R1 ${R1_SHORT}; skip final holdout and keep active"
    return 0
  fi

  if [[ -f "${report}" ]]; then
    log "${CHAIN_HOLD_ID}: multiseed holdout already exists; not rerunning"
  else
    if [[ -d "${ROOT}/.evolution/runs/${CHAIN_HOLD_ID}" ]]; then
      die "${CHAIN_HOLD_ID}: partial holdout dir without aggregate report"
    fi
    log "${CHAIN_HOLD_ID}: START final ${SEEDS}-seed holdout (base=${R1_SHORT}, candidate=${champion})"
    "${PYTHON}" scripts/ashare/run_skill_evolution.py \
      --snapshot="${SNAPSHOT}" \
      --train_months="${TRAIN_MONTHS}" \
      --dev_months="${DEV_MONTHS}" \
      --holdout_months="${HOLDOUT_MONTHS}" \
      holdout_multiseed \
      --candidate="${champion}" \
      --run_id="${CHAIN_HOLD_ID}" \
      --seeds="${SEEDS}" \
      --base="${R1_SHORT}" \
      --cleanup_db=True
    log "${CHAIN_HOLD_ID}: holdout finished"
  fi

  [[ -f "${report}" ]] || die "${CHAIN_HOLD_ID}: missing fitness_holdout_multiseed.json"

  local -a result
  mapfile -t result < <(
    "${PYTHON}" - "${report}" "${ROOT}/.evolution/registry.json" <<'PY'
import json, sys
from pathlib import Path
report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
reg = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
passed = bool(report.get("gates", {}).get("passed", False))
score = float(report.get("score", float("-inf")))
cand = str(report.get("candidate", ""))
print("true" if passed else "false")
print(f"{score:.6f}")
print(cand)
print(str(reg["active_digest"])[:12])
print(str(report.get("mean_paired_pp", 0.0)))
print(",".join(str(x) for x in report.get("seed_scores", [])))
PY
  )

  local passed="${result[0]}"
  local score="${result[1]}"
  local cand="${result[2]}"
  local active="${result[3]}"
  local paired="${result[4]}"
  local seed_scores="${result[5]}"
  log "final holdout: gates=${passed} mean_score=${score} mean_paired=${paired}pp seeds=[${seed_scores}] candidate=${cand} active=${active}"

  if [[ "${passed}" == "true" ]] && \
     "${PYTHON}" - "${score}" <<'PY'
import sys
raise SystemExit(0 if float(sys.argv[1]) > 0.0 else 1)
PY
  then
    log "promoting ${cand} after sealed multi-seed holdout"
    "${PYTHON}" scripts/ashare/run_skill_evolution.py promote \
      --candidate="${cand}" --run_id="${CHAIN_HOLD_ID}"
  else
    log "promotion rejected; production active remains ${active}"
  fi
}

log "CHAIN START: formal24m dev-only R2->R3->R4 + final ${SEEDS}-seed holdout (tag=${RUN_TAG})"
log "snapshot=${SNAPSHOT} split=${TRAIN_MONTHS}/${DEV_MONTHS}/${HOLDOUT_MONTHS}"
log "R1 parent=${R1_SHORT}; production skill will not change until final holdout"
require_active_is_r1

champion="$(read_champion)"
log "starting temporary champion=${champion}"

for round in 2 3 4; do
  champion="$(run_search_round "${round}" "${champion}")"
done

log "search budget exhausted; temporary champion=${champion}"
final_holdout_and_maybe_promote "${champion}"
