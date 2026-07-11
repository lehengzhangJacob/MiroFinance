# MiroMemSkill

MiroFlow fork with **memory** and **skill** mechanisms for financial agent reasoning.

## Architecture

Current linkage paths between tools, memory, and skills:

1. **Passive injection** — Before each task, inject matched skills and only temporally visible memory (`orchestrator.py`).
2. **Active tools** — MCP server `tool-memskill`: `memory_search`, `memory_save`, `skill_list`, `skill_load`.
3. **Post-tool feature evidence** — For A-share direction tasks, condition matured samples/rules on the current stock only after point-in-time market tools have run.
4. **Barrier reflection** — Direction tasks use strict expanding-window/FDR rules; ranking tasks expose matured monthly factor RankIC only after exact sign-flip tests and FDR control. A direction-free abstention status is available as an opt-in ablation (`common_benchmark.py`).

## Quick Start

```bash
cd MiroMemSkill
uv sync

# Unit smoke tests
uv run python scripts/memory_smoke_test.py
uv run python scripts/ashare/rank_smoke_test.py

# FinSearchComp with memory+skill (5-task smoke)
uv run main.py common-benchmark \
  --config_file_name=agent_finsearchcomp_memskill \
  benchmark=finsearchcomp-memskill-smoke5 \
  output_dir=logs/memskill_smoke5 \
  benchmark.execution.max_concurrent=1

# Verify linkage paths in logs
uv run python scripts/verify_memskill_integration.py logs/memskill_smoke5

# Generate the 12-month, 16-stock cross-sectional ranking benchmark
uv run python scripts/ashare/gen_rank_tasks.py

# Ranking baseline / factor-memory arms
uv run main.py common-benchmark \
  --config_file_name=agent_ashare_rank_kimi \
  output_dir=logs/ashare_rank_baseline
uv run main.py common-benchmark \
  --config_file_name=agent_ashare_rank_skill_only_kimi \
  output_dir=logs/ashare_rank_skill_only
uv run main.py common-benchmark \
  --config_file_name=agent_ashare_rank_mem0_kimi \
  output_dir=logs/ashare_rank_mem0

# Deterministic metrics plus paired monthly sign-flip tests
# (framework pass@1 is parse success only)
uv run python scripts/ashare/eval_rank.py \
  --run baseline=logs/ashare_rank_baseline \
  --run skill_only=logs/ashare_rank_skill_only \
  --run rank_memory=logs/ashare_rank_mem0 \
  --out logs/ashare_rank_report.md
```

## Ablation Switches

Override via Hydra CLI:

```bash
memory.enabled=false              # baseline (no memory/skill injection)
memory.skill_enabled=false        # memory only
memory.reflection_enabled=false   # no post-task writeback
```

Ready-made A-share direction ablations:

- `agent_ashare_clean_baseline_kimi`
- `agent_ashare_skill_only_kimi`
- `agent_ashare_calibration_only_kimi`
- `agent_ashare_mem0v4_kimi`

Ready-made A-share ranking arms:

- `agent_ashare_rank_kimi`
- `agent_ashare_rank_skill_only_kimi`
- `agent_ashare_rank_mem0_kimi`

Kimi K2.6 uses fixed `temperature=1` in thinking mode. The ranking arms
uniformly disable thinking and use the model's required `temperature=0.6`,
preventing hidden reasoning from consuming the full completion budget. Compare
only arms with the same mode, and use paired tasks plus repeated runs when
estimating treatment effects.

The ranking report audits both actual memory-block activation and effective
prompt changes. If no factor passes FDR, the strict memory arm intentionally
abstains; stochastic output differences must not be reported as memory gains.

## Key Paths

| Component | Path |
|-----------|------|
| Memory store | `src/memory/vector_store.py` + `src/memory/memory.py` |
| Skill library | `src/memory/skills.py` + `memory_bank/skills_ashare/` |
| Direction reflection | `src/memory/rolling_reflection.py` |
| Ranking reflection | `src/memory/rank_reflection.py` |
| Post-tool evidence | `src/memory/feature_evidence.py` |
| MCP tools | `src/tool/mcp_servers/memskill_mcp_server.py` |
| A-share rank evaluator | `scripts/ashare/eval_rank.py` |
| Agent configs | `config/agent_ashare_*.yaml` |
