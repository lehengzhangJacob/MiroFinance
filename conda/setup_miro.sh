#!/usr/bin/env bash
# Create/update conda env "Miro" for MiroFlow + MiroMemSkill.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONDA_DIR="$ROOT/conda"

echo "==> Exporting locked deps from MiroFlow/uv.lock"
cd "$ROOT/MiroFlow"
uv export --frozen --no-dev --no-emit-project -o "$CONDA_DIR/requirements-miro.txt"

echo "==> Creating/updating conda env: Miro"
if conda env list | awk '{print $1}' | grep -qx Miro; then
  conda env update -f "$CONDA_DIR/miro.yml" --prune
else
  conda env create -f "$CONDA_DIR/miro.yml"
fi

echo "==> Installing pip dependencies"
conda run -n Miro pip install -r "$CONDA_DIR/requirements-miro.txt"
conda run -n Miro pip install -e "$ROOT/MiroFlow"
conda run -n Miro pip install -e "$ROOT/MiroMemSkill"

echo "==> Miro env ready"
conda run -n Miro python -c "import hydra, fastmcp, openai; print('imports ok')"
