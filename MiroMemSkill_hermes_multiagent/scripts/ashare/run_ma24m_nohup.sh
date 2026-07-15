#!/usr/bin/env bash
# Formal 24-month multi-agent run: train(12) -> dev(6) -> holdout(6),
# then paired comparison vs the hermes single-agent formal24m arms.
set -uo pipefail

cd "$(dirname "$0")/../.."
export PATH="/home/msj_team/.conda/envs/Miro/bin:$PATH"

RUN_ID="${1:-ma24m_20260715}"
SPLITS=(--train_months=12 --dev_months=6 --holdout_months=6)
SINGLE_ROOT="/home/msj_team/Jacob/agent/MiroMemSkill_hermes/.evolution/runs/formal24m_20260715"

for level in train dev holdout; do
  echo "=== [$(date '+%F %T')] arm ${level} start ==="
  python scripts/ashare/run_skill_evolution.py "${SPLITS[@]}" run_arm \
    --run_id="${RUN_ID}" --arm="${level}" --candidate=baseline \
    --level="${level}" --cleanup_db=True --resume=True
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "=== [$(date '+%F %T')] arm ${level} FAILED (exit ${rc}) ==="
    exit $rc
  fi
  echo "=== [$(date '+%F %T')] arm ${level} done ==="
done

echo "=== [$(date '+%F %T')] comparison ==="
python scripts/ashare/compare_architecture.py \
  --multi_root=".evolution/runs/${RUN_ID}" \
  --single_root="${SINGLE_ROOT}" \
  --out=".evolution/runs/${RUN_ID}/reports/architecture_compare.md"
echo "=== [$(date '+%F %T')] ALL DONE ==="
