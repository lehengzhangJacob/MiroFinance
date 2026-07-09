#!/usr/bin/env bash
# Run skill evolution; all artifacts stay under hermes-agent-self-evolution/.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Local hermes-agent stub (never ../hermes-agent at repo root)
export HERMES_AGENT_REPO="$ROOT/hermes-agent"

# Kimi via OpenAI-compatible API (override with OPENAI_* env if needed)
LLM_KEY="$ROOT/../llm_key"
if [ -z "${OPENAI_API_KEY:-}" ] && [ -f "$LLM_KEY" ]; then
  export OPENAI_API_KEY="$(grep -oP '(?<=KIMI_API_KEY=).*' "$LLM_KEY")"
fi
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.moonshot.cn/v1}"

# No SOCKS proxy bleed from BASH_ENV (conda run spawns a fresh shell)
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY

SKILL="${1:-ashare_prediction_protocol}"
ITER="${2:-3}"
EXTRA="${3:-}"

exec env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY \
  conda run -n Hermes python -m evolution.skills.evolve_skill \
  --skill "$SKILL" \
  --iterations "$ITER" \
  --eval-source synthetic \
  --hermes-repo "$HERMES_AGENT_REPO" \
  --optimizer-model "${OPTIMIZER_MODEL:-openai/moonshot-v1-32k}" \
  --eval-model "${EVAL_MODEL:-openai/moonshot-v1-32k}" \
  $EXTRA
