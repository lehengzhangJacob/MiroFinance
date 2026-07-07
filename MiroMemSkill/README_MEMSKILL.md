# MiroMemSkill

MiroFlow fork with **memory** and **skill** mechanisms for financial agent reasoning.

## Architecture

Three linkage paths between tools, memory, and skills:

1. **Passive injection** — Before each task, retrieve top-k memories + matched skills into the initial prompt (`orchestrator.py`).
2. **Active tools** — MCP server `tool-memskill`: `memory_search`, `memory_save`, `skill_list`, `skill_load`.
3. **Post-task reflection** — After judging, distill strategy lessons into episodic memory (`common_benchmark.py`).

## Quick Start

```bash
cd MiroMemSkill
uv sync

# Unit smoke tests
uv run python scripts/memory_smoke_test.py

# FinSearchComp with memory+skill (5-task smoke)
uv run main.py common-benchmark \
  --config_file_name=agent_finsearchcomp_memskill \
  benchmark=finsearchcomp-memskill-smoke5 \
  output_dir=logs/memskill_smoke5 \
  benchmark.execution.max_concurrent=1

# Verify linkage paths in logs
uv run python scripts/verify_memskill_integration.py logs/memskill_smoke5
```

## Ablation Switches

Override via Hydra CLI:

```bash
memory.enabled=false              # baseline (no memory/skill injection)
memory.skill_enabled=false        # memory only
memory.reflection_enabled=false   # no post-task writeback
```

## Key Paths

| Component | Path |
|-----------|------|
| Memory store | `src/memory/store.py` |
| Skill library | `src/memory/skills.py` + `memory_bank/skills/*.md` |
| Reflection | `src/memory/reflection.py` |
| MCP tools | `src/tool/mcp_servers/memskill_mcp_server.py` |
| Agent config | `config/agent_finsearchcomp_memskill.yaml` |
| Memory config | `config/memory/default.yaml` |
