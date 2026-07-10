#!/usr/bin/env bash
# Launch GEPA skill evolution with nohup (survives terminal/IDE exit).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SKILL="${1:-ashare_prediction_protocol}"
ITER="${2:-3}"
EXTRA="${3:-}"

LOG_DIR="$ROOT/logs/nohup_evolve"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${SKILL}_${ITER}iter.log"

# Kimi via OpenAI-compatible API
LLM_KEY="$ROOT/../llm_key"
if [ -f "$LLM_KEY" ]; then
  export OPENAI_API_KEY="$(grep -oP '(?<=KIMI_API_KEY=).*' "$LLM_KEY")"
fi
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.moonshot.cn/v1}"
export HERMES_AGENT_REPO="$ROOT/hermes-agent"

# Proxy from BASH_ENV breaks httpx (socks5 without socksio)
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY

PYTHON="${CONDA_PREFIX:-/home/msj_team/.conda/envs/Hermes}/bin/python"
if [ ! -x "$PYTHON" ]; then
  PYTHON="$(conda run -n Hermes which python 2>/dev/null || true)"
fi
if [ ! -x "$PYTHON" ]; then
  echo "Hermes conda env not found. Run: $(cd "$ROOT/.." && pwd)/deploy/conda/setup_hermes.sh" >&2
  exit 1
fi

nohup env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY \
  OPENAI_API_KEY="$OPENAI_API_KEY" \
  OPENAI_BASE_URL="$OPENAI_BASE_URL" \
  HERMES_AGENT_REPO="$HERMES_AGENT_REPO" \
  "$PYTHON" -u -m evolution.skills.evolve_skill \
    --skill "$SKILL" \
    --iterations "$ITER" \
    --eval-source synthetic \
    --hermes-repo "$HERMES_AGENT_REPO" \
    --optimizer-model "${OPTIMIZER_MODEL:-openai/moonshot-v1-32k}" \
    --eval-model "${EVAL_MODEL:-openai/moonshot-v1-32k}" \
    $EXTRA \
  > "$LOG_FILE" 2>&1 &

echo "evolve pid=$! skill=$SKILL iterations=$ITER"
echo "log=$LOG_FILE"
echo "tail -f $LOG_FILE"
