#!/usr/bin/env bash
# Launch pool3 Kimi baseline + MemSkill with nohup (survives terminal/IDE exit).
set -u
AGENT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
KEY=$(grep -oP '(?<=KIMI_API_KEY=).*' "$AGENT_ROOT/llm_key")

# A-share + Kimi (moonshot.cn) are domestic; proxy from BASH_ENV breaks httpx
# (socks5 without socksio). Unset before launch.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY

export KIMI_API_KEY="$KEY"
export EVAL_LLM_API_KEY="$KEY"
export EVAL_LLM_BASE_URL="https://api.moonshot.cn/v1"
export EVAL_LLM_MODEL_NAME="moonshot-v1-32k"
export REFLECTION_LLM_API_KEY="$KEY"
export REFLECTION_LLM_BASE_URL="https://api.moonshot.cn/v1"
export REFLECTION_LLM_MODEL_NAME="moonshot-v1-32k"

LOG_DIR="$AGENT_ROOT/logs/nohup_pool3"
mkdir -p "$LOG_DIR"

# Baseline (MiroFlow)
cd "$AGENT_ROOT/MiroFlow"
nohup conda run -n Miro python main.py common-benchmark \
    --config_file_name=agent_ashare_kimi \
    benchmark=ashare-pred \
    output_dir=logs/ashare_pool3_kimi \
    benchmark.execution.max_concurrent=3 \
    > "$LOG_DIR/baseline.log" 2>&1 &
echo "baseline pid=$! log=$LOG_DIR/baseline.log"

# MemSkill v3 (MiroMemSkill)
cd "$AGENT_ROOT/MiroMemSkill"
nohup conda run -n Miro python main.py common-benchmark \
    --config_file_name=agent_ashare_memskill_v3_kimi \
    benchmark=ashare-pred \
    output_dir=logs/ashare_pool3_kimi \
    benchmark.execution.max_concurrent=3 \
    > "$LOG_DIR/memskill.log" 2>&1 &
echo "memskill pid=$! log=$LOG_DIR/memskill.log"

# Report finalizer
nohup "$AGENT_ROOT/MiroFlow/scripts/ashare/finalize_pool3_report.sh" \
    > /tmp/ashare_pool3_finalize.log 2>&1 &
echo "finalize pid=$! log=/tmp/ashare_pool3_finalize.log"
