#!/usr/bin/env bash
# Launch pool3 MiroFlow baseline with conda env Miro (nohup).
# Uses a separate output dir so the earlier uv-run baseline stays intact.
set -u
AGENT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
KEY=$(grep -oP '(?<=KIMI_API_KEY=).*' "$AGENT_ROOT/llm_key")

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY

export KIMI_API_KEY="$KEY"
export EVAL_LLM_API_KEY="$KEY"
export EVAL_LLM_BASE_URL="https://api.moonshot.cn/v1"
export EVAL_LLM_MODEL_NAME="moonshot-v1-32k"

LOG_DIR="$AGENT_ROOT/deploy/logs/nohup_pool3"
mkdir -p "$LOG_DIR"

PYTHON="/home/msj_team/.conda/envs/Miro/bin/python"
if [ ! -x "$PYTHON" ]; then
  echo "Miro conda env not found. Run: $AGENT_ROOT/deploy/conda/setup_miro.sh" >&2
  exit 1
fi

cd "$AGENT_ROOT/MiroFlow"
nohup "$PYTHON" -u main.py common-benchmark \
    --config_file_name=agent_ashare_kimi \
    benchmark=ashare-pred \
    output_dir=logs/ashare_pool3_baseline_conda \
    benchmark.execution.max_concurrent=3 \
    > "$LOG_DIR/baseline_conda.log" 2>&1 &
echo "baseline-conda pid=$! log=$LOG_DIR/baseline_conda.log"
echo "output_dir=MiroFlow/logs/ashare_pool3_baseline_conda"
echo "tail -f $LOG_DIR/baseline_conda.log"
