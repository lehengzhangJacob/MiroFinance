#!/usr/bin/env bash
set -euo pipefail

SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python "$SKILL_DIR/run.py" --as-of 2024-07-01
python "$SKILL_DIR/run.py" --as-of 2024-07-01 --window 20 --top-k 4 --format csv
