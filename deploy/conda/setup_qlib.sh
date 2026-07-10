#!/usr/bin/env bash
# Create/update conda env "Qlib" for the qlib_skill package
# (MiroMemSkill/memory_bank/skills_ashare/qlib_skill).
# Separate env because pyqlib pins numpy/pandas versions that conflict
# with the Miro env (numpy 2.3 / pandas 2.3).
set -euo pipefail

CONDA_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "==> Writing Qlib requirements"
cat > "$CONDA_DIR/requirements-qlib.txt" <<'EOF'
pyqlib
lightgbm
pyyaml>=6.0
EOF

echo "==> Creating/updating conda env: Qlib"
if conda env list | awk '{print $1}' | grep -qx Qlib; then
  conda env update -f "$CONDA_DIR/qlib.yml" --prune
else
  conda env create -f "$CONDA_DIR/qlib.yml"
fi

echo "==> Installing pip dependencies"
conda run -n Qlib pip install -r "$CONDA_DIR/requirements-qlib.txt"

echo "==> Qlib env ready"
conda run -n Qlib python -c "import qlib, lightgbm; print('qlib', qlib.__version__, '| lightgbm', lightgbm.__version__)"
