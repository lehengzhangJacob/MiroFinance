#!/usr/bin/env bash
# qlib_skill: full pipeline example. Requires the Qlib conda env
# (deploy/conda/setup_qlib.sh) and the local A-share CSV cache.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${QLIB_PY:-/home/msj_team/.conda/envs/Qlib/bin/python}"
RUN=demo

echo "== convert: CSV cache -> qlib bin =="
"$PY" run.py convert

echo "== train: LGBM + Alpha158 (label horizon 20d) =="
"$PY" run.py train --run-name "$RUN"

echo "== signal: IC / RankIC =="
"$PY" run.py signal --run-name "$RUN"

echo "== backtest: TopkDropout vs SH000300 =="
"$PY" run.py backtest --run-name "$RUN"

echo "== report =="
"$PY" run.py report --run-name "$RUN"
echo
echo "Report: outputs/$RUN/report.md"
