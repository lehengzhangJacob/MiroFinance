#!/usr/bin/env bash
# Rerun pool3 MemSkill arm with direction-collapse fixes (ashare_v4 namespace).
# Output goes to logs/ashare_pool3_v4_kimi so the v3 run stays intact.
set -u
AGENT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
KEY=$(grep -oP '(?<=KIMI_API_KEY=).*' "$AGENT_ROOT/llm_key")

# Kimi (moonshot.cn) is domestic; SOCKS proxy from BASH_ENV breaks httpx.
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

cd "$AGENT_ROOT/MiroMemSkill"
nohup "$PYTHON" -u main.py common-benchmark \
    --config_file_name=agent_ashare_memskill_v4_kimi \
    benchmark=ashare-pred \
    output_dir=logs/ashare_pool3_v4_kimi \
    benchmark.execution.max_concurrent=3 \
    > "$LOG_DIR/memskill_v4.log" 2>&1 &
echo "memskill-v4 pid=$! log=$LOG_DIR/memskill_v4.log"

# Report finalizer: wait for the v4 run, then rebuild 3-arm comparison
nohup "$AGENT_ROOT/MiroFlow/scripts/ashare/finalize_pool3_v4_report.sh" \
    > /tmp/ashare_pool3_v4_finalize.log 2>&1 &
echo "finalize pid=$! log=/tmp/ashare_pool3_v4_finalize.log"
