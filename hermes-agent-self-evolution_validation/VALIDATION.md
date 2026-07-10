# Hermes Self-Evolution Validation (dev-assistant)

Independent mini-scenario to validate Phase 1–4 GEPA evolution. **No Miro dependencies.**

## Scenario

Agent helps with developer workflow: Conventional Commits, tool selection, coding guidelines, text helpers.

| Phase | Target | Artifact |
|-------|--------|----------|
| 1 | Skill | `hermes-agent/skills/dev/commit_message_writer/SKILL.md` |
| 2 | Tool descriptions | `hermes-agent/tools/dev_tools.json` |
| 3 | Prompt section | `hermes-agent/prompts/coding_agent_guidelines.md` |
| 4 | Code | `hermes-agent/code/text_helpers.py` |

Baseline seeds contain intentional defects for GEPA to fix.

## Setup

```bash
conda activate Hermes
cd hermes-agent-self-evolution_validation
pip install -e ".[dev]"
export HERMES_AGENT_REPO="$PWD/hermes-agent"
./scripts/seed_validation_fixtures.sh
```

## Run

```bash
# Dry-run all phases
./scripts/run_full_evolution_validation.sh --dry-run

# Full validation (serial, uses Kimi from ../llm_key)
./scripts/run_full_evolution_validation.sh

# Background
./scripts/run_full_evolution_validation.sh --nohup
tail -f logs/nohup_validation/full_validation.log
```

## What to observe

- `output/{target}/{timestamp}/gepa_trace.jsonl` — proposed mutations, accept/reject
- `metrics.json` — baseline vs evolved holdout scores
- `diff baseline_* evolved_*` — text/code changes
- `validation_report.md` — summary after full run

## Success criteria

1. All four phases dry-run pass; `pytest tests/ -q` green
2. ≥2/4 phases show holdout improvement > 0
3. ≥1 accepted GEPA mutation with non-empty diff
4. Phase 4 evolved code passes all helper tests
