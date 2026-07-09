#!/usr/bin/env bash
# Create/update conda env "Hermes" for hermes-agent-self-evolution.
set -euo pipefail

AGENT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CONDA_DIR="$(cd "$(dirname "$0")" && pwd)"
HERMES="$AGENT_ROOT/hermes-agent-self-evolution"

echo "==> Writing Hermes requirements (no uv.lock in project)"
cat > "$CONDA_DIR/requirements-hermes.txt" <<'EOF'
dspy>=3.0.0
openai>=1.0.0
pyyaml>=6.0
click>=8.0
rich>=13.0
pytest>=7.0
pytest-asyncio>=0.21
optuna>=3.0
EOF

echo "==> Creating/updating conda env: Hermes"
if conda env list | awk '{print $1}' | grep -qx Hermes; then
  conda env update -f "$CONDA_DIR/hermes.yml" --prune
else
  conda env create -f "$CONDA_DIR/hermes.yml"
fi

echo "==> Installing pip dependencies"
conda run -n Hermes pip install -r "$CONDA_DIR/requirements-hermes.txt"
conda run -n Hermes pip install -e "$HERMES"

echo "==> Hermes env ready"
conda run -n Hermes python -c "import dspy, click, optuna; print('imports ok')"
