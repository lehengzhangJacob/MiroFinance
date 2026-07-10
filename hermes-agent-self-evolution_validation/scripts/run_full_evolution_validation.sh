#!/usr/bin/env bash
# Full four-phase Hermes self-evolution validation (dev-assistant scenario).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DRY_RUN=""
NOHUP=""
ITER="${EVOL_ITERATIONS:-5}"
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN="--dry-run" ;;
    --nohup) NOHUP=1 ;;
  esac
done

export HERMES_AGENT_REPO="$ROOT/hermes-agent"
LLM_KEY="$ROOT/../llm_key"
if [ -z "${OPENAI_API_KEY:-}" ] && [ -f "$LLM_KEY" ]; then
  export OPENAI_API_KEY="$(grep -oP '(?<=KIMI_API_KEY=).*' "$LLM_KEY")"
fi
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.moonshot.cn/v1}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

PYTHON="${CONDA_PREFIX:-/home/msj_team/.conda/envs/Hermes}/bin/python"
[ -x "$PYTHON" ] || PYTHON="$(conda run -n Hermes which python 2>/dev/null || true)"

chmod +x "$ROOT/scripts/seed_validation_fixtures.sh"
"$ROOT/scripts/seed_validation_fixtures.sh"

OPT_MODEL="${OPTIMIZER_MODEL:-openai/moonshot-v1-32k}"
EVAL_MODEL="${EVAL_MODEL:-openai/moonshot-v1-32k}"

_run() {
  echo "=== $1 ==="
  shift
  env -u http_proxy -u https_proxy \
    OPENAI_API_KEY="$OPENAI_API_KEY" \
    OPENAI_BASE_URL="$OPENAI_BASE_URL" \
    HERMES_AGENT_REPO="$HERMES_AGENT_REPO" \
    "$PYTHON" "$@"
}

phases() {
  _run "Phase 1: Skill" -m evolution.skills.evolve_skill \
    --skill commit_message_writer \
    --iterations "$ITER" \
    --eval-source golden \
    --dataset-path datasets/skills/commit_message_writer \
    --hermes-repo "$HERMES_AGENT_REPO" \
    --optimizer-model "$OPT_MODEL" \
    --eval-model "$EVAL_MODEL" \
    $DRY_RUN

  _run "Phase 2: Tools" -m evolution.tools.evolve_tool_descriptions \
    --tool-set dev \
    --iterations "$ITER" \
    --dataset-path datasets/tools/dev_tool_selection \
    --hermes-repo "$HERMES_AGENT_REPO" \
    --optimizer-model "$OPT_MODEL" \
    --eval-model "$EVAL_MODEL" \
    $DRY_RUN

  _run "Phase 3: Prompt" -m evolution.prompts.evolve_prompt_section \
    --section coding_agent_guidelines \
    --iterations "$ITER" \
    --dataset-path datasets/prompts/coding_agent_guidelines \
    --hermes-repo "$HERMES_AGENT_REPO" \
    --optimizer-model "$OPT_MODEL" \
    --eval-model "$EVAL_MODEL" \
    $DRY_RUN

  _run "Phase 4: Code" -m evolution.code.evolve_tool_code \
    --target text_helpers \
    --iterations "$ITER" \
    --hermes-repo "$HERMES_AGENT_REPO" \
    --optimizer-model "$OPT_MODEL" \
    --eval-model "$EVAL_MODEL" \
    $DRY_RUN

  if [ -z "$DRY_RUN" ]; then
    _run "Summarize" "$ROOT/scripts/summarize_evolution_validation.py"
  fi
}

if [ -n "$NOHUP" ] && [ -z "$DRY_RUN" ]; then
  LOG_DIR="$ROOT/logs/nohup_validation"
  mkdir -p "$LOG_DIR"
  LOG="$LOG_DIR/full_validation.log"
  nohup bash "$0" > "$LOG" 2>&1 &
  echo "pid=$! log=$LOG"
  exit 0
fi

phases
