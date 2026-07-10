#!/usr/bin/env bash
# Seed hermes-agent stub from fixtures/dev_assistant (validation-only, no Miro deps).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FIX="$ROOT/fixtures/dev_assistant"
HA="$ROOT/hermes-agent"

mkdir -p "$HA/skills/dev/commit_message_writer"
mkdir -p "$HA/tools" "$HA/prompts" "$HA/code"

cp "$FIX/skills/dev/commit_message_writer/SKILL.md" "$HA/skills/dev/commit_message_writer/SKILL.md"
cp "$FIX/tools/dev_tools.json" "$HA/tools/dev_tools.json"
cp "$FIX/prompts/coding_agent_guidelines.md" "$HA/prompts/coding_agent_guidelines.md"
cp "$FIX/code/text_helpers.py" "$HA/code/text_helpers.py"

for split_dir in \
  skills/commit_message_writer \
  tools/dev_tool_selection \
  prompts/coding_agent_guidelines; do
  dst="$ROOT/datasets/$split_dir"
  src="$FIX/datasets/$split_dir"
  mkdir -p "$dst"
  cp "$src"/*.jsonl "$dst/"
done

echo "Seeded dev-assistant fixtures into hermes-agent/ and datasets/"
