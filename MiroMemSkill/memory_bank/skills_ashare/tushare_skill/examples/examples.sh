#!/usr/bin/env bash
# tushare_skill: one runnable example per subcommand.
# Requires a valid token (env TUSHARE_TOKEN, config.yaml token.file, or an
# ancestor tushare_token file). Run from this skill's directory:
#   cd "$(dirname "$0")/.." && bash examples/examples.sh
set -euo pipefail
cd "$(dirname "$0")/.."

AS_OF=20240701

echo "== daily: Moutai 2024H1, qfq, point-in-time $AS_OF =="
python run.py daily --ts-code 600519.SH --start 20240101 --as-of $AS_OF | head -c 600; echo; echo

echo "== index: CSI300 same window =="
python run.py index --start 20240101 --as-of $AS_OF | head -c 400; echo; echo

echo "== valuation: PE/PB/turnover =="
python run.py valuation --ts-code 600519.SH --start 20240601 --as-of $AS_OF | head -c 400; echo; echo

echo "== financials: only reports ANNOUNCED on/before $AS_OF =="
python run.py financials --ts-code 600519.SH --as-of $AS_OF | head -c 400; echo; echo

echo "== stock-info: current snapshot (not point-in-time) =="
python run.py stock-info --ts-code 600519.SH; echo

echo "== trade-cal: write CSV to /tmp =="
python run.py trade-cal --start 20240101 --end 20240131 --out /tmp/tushare_skill_cal.csv
echo
echo "All examples done."
