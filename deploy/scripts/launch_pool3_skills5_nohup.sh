#!/usr/bin/env bash
# Parallel pool3: MiroFlow baseline + MiroMemSkill v5 (tushare/qlib skills, ashare_v5 bank).
set -u
AGENT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
KEY=$(grep -oP '(?<=KIMI_API_KEY=).*' "$AGENT_ROOT/llm_key")

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY

export KIMI_API_KEY="$KEY"
export EVAL_LLM_API_KEY="$KEY"
export EVAL_LLM_BASE_URL="https://api.moonshot.cn/v1"
export EVAL_LLM_MODEL_NAME="moonshot-v1-32k"
export REFLECTION_LLM_API_KEY="$KEY"
export REFLECTION_LLM_BASE_URL="https://api.moonshot.cn/v1"
export REFLECTION_LLM_MODEL_NAME="moonshot-v1-32k"

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
    output_dir=logs/ashare_pool3_baseline_skills5 \
    benchmark.execution.max_concurrent=3 \
    > "$LOG_DIR/baseline_skills5.log" 2>&1 &
BASELINE_PID=$!
echo "baseline-skills5 pid=$BASELINE_PID log=$LOG_DIR/baseline_skills5.log"
echo "output_dir=MiroFlow/logs/ashare_pool3_baseline_skills5"

cd "$AGENT_ROOT/MiroMemSkill"
nohup "$PYTHON" -u main.py common-benchmark \
    --config_file_name=agent_ashare_memskill_v5_kimi \
    benchmark=ashare-pred \
    output_dir=logs/ashare_pool3_v5_kimi \
    benchmark.execution.max_concurrent=3 \
    > "$LOG_DIR/memskill_v5.log" 2>&1 &
MEMSKILL_PID=$!
echo "memskill-v5 pid=$MEMSKILL_PID log=$LOG_DIR/memskill_v5.log"
echo "output_dir=MiroMemSkill/logs/ashare_pool3_v5_kimi"

echo ""
echo "Monitor:"
echo "  tail -f $LOG_DIR/baseline_skills5.log"
echo "  tail -f $LOG_DIR/memskill_v5.log"
