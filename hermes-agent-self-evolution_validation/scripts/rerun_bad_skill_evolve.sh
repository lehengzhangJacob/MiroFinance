#!/usr/bin/env bash
# Re-seed bad skill fixture and run Phase 1 skill evolution only.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export HERMES_AGENT_REPO="$ROOT/hermes-agent"
LLM_KEY="$ROOT/../llm_key"
[ -z "${OPENAI_API_KEY:-}" ] && [ -f "$LLM_KEY" ] && export OPENAI_API_KEY="$(grep -oP '(?<=KIMI_API_KEY=).*' "$LLM_KEY")"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.moonshot.cn/v1}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

ITER="${1:-3}"
NOHUP="${2:-}"

# Only re-copy skill seed (keep code/prompt/tools as-is)
mkdir -p hermes-agent/skills/dev/commit_message_writer
cp fixtures/dev_assistant/skills/dev/commit_message_writer/SKILL.md \
   hermes-agent/skills/dev/commit_message_writer/SKILL.md
echo "Seeded BAD commit_message_writer skill ($(wc -c < hermes-agent/skills/dev/commit_message_writer/SKILL.md) bytes)"

PYTHON="${CONDA_PREFIX:-/home/msj_team/.conda/envs/Hermes}/bin/python"
CMD=( "$PYTHON" -u -m evolution.skills.evolve_skill
  --skill commit_message_writer
  --iterations "$ITER"
  --eval-source golden
  --dataset-path datasets/skills/commit_message_writer
  --hermes-repo "$HERMES_AGENT_REPO"
  --optimizer-model "${OPTIMIZER_MODEL:-openai/moonshot-v1-32k}"
  --eval-model "${EVAL_MODEL:-openai/moonshot-v1-32k}"
)

if [ "$NOHUP" = "--nohup" ]; then
  LOG_DIR="$ROOT/logs/nohup_validation"
  mkdir -p "$LOG_DIR"
  LOG="$LOG_DIR/skill_bad_rerun_${ITER}iter.log"
  nohup env -u http_proxy -u https_proxy "${CMD[@]}" > "$LOG" 2>&1 &
  echo "pid=$! log=$LOG"
else
  "${CMD[@]}"
fi
