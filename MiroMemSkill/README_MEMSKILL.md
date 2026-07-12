# MiroMemSkill

MiroFlow fork with **memory** and **skill** mechanisms for financial agent reasoning.

## Architecture

Current linkage paths between tools, memory, and skills:

1. **Passive injection** — Before each task, inject matched skills and only temporally visible memory (`orchestrator.py`).
2. **Active tools** — MCP server `tool-memskill`: `memory_search`, `memory_save`, `skill_list`, `skill_load`.
3. **Post-tool feature evidence** — For A-share direction tasks, condition matured samples/rules on the current stock only after point-in-time market tools have run.
4. **Barrier reflection** — Direction tasks use strict expanding-window/FDR rules; ranking tasks expose matured monthly factor RankIC only after exact sign-flip tests and FDR control. Unified trader tasks additionally persist factual portfolio episodes and inject them only after the 20-session exit date. A direction-free abstention status is available as an opt-in ablation (`common_benchmark.py`).
5. **Official persistence** — `mem0ai.Memory` performs vector CRUD, extraction and consolidation against a shared Qdrant service. Mem0 operation history is stored in SQLite.

All enabled `config/memory/*.yaml` profiles use `backend: mem0_qdrant`.
The profile `namespace` is passed to official Mem0 as `user_id`, so experiments
share the single `miromemskill` collection without sharing memories. Per-task
reflection uses `infer=True`; validated rolling rules use `infer=False` so an
LLM cannot rewrite deterministic statistical output.

The fresh-store invariant is that every retrievable memory has an ISO
`available_after` timestamp. Qdrant applies the cutoff before semantic search.
Records with unknown availability fail closed under a time-filtered query.

## Quick Start

```bash
# From the repository root: create/update and activate the Conda runtime.
./deploy/conda/setup_miro.sh
source deploy/conda/activate_miro.sh
cd MiroMemSkill

# Start the shared, persistent Qdrant service
docker compose -f deploy/qdrant/compose.yaml up -d
curl --fail http://127.0.0.1:6333/healthz

# Unit smoke tests
python scripts/memory_smoke_test.py
python scripts/ashare/rank_smoke_test.py
python scripts/ashare/trader_smoke_test.py

# Live official-Mem0/Qdrant integration checks
set -a; source ../llm_key; set +a
MEM0_QDRANT_INTEGRATION=1 MEM0_QDRANT_INFERENCE_TEST=1 \
  python scripts/memory_smoke_test.py
python scripts/qdrant_concurrency_test.py
python scripts/agent_memory_path_smoke_test.py

# FinSearchComp with memory+skill (5-task smoke)
python main.py common-benchmark \
  --config_file_name=agent_finsearchcomp_memskill \
  benchmark=finsearchcomp-memskill-smoke5 \
  output_dir=logs/memskill_smoke5 \
  benchmark.execution.max_concurrent=1

# Verify linkage paths in logs
python scripts/verify_memskill_integration.py logs/memskill_smoke5

# Generate the 12-month, 16-stock cross-sectional ranking benchmark
python scripts/ashare/gen_rank_tasks.py

# Ranking baseline / factor-memory arms
python main.py common-benchmark \
  --config_file_name=agent_ashare_rank_kimi \
  output_dir=logs/ashare_rank_baseline
python main.py common-benchmark \
  --config_file_name=agent_ashare_rank_skill_only_kimi \
  output_dir=logs/ashare_rank_skill_only
python main.py common-benchmark \
  --config_file_name=agent_ashare_rank_mem0_kimi \
  output_dir=logs/ashare_rank_mem0

# Deterministic metrics plus paired monthly sign-flip tests
# (framework pass@1 is parse success only)
python scripts/ashare/eval_rank.py \
  --run baseline=logs/ashare_rank_baseline \
  --run skill_only=logs/ashare_rank_skill_only \
  --run rank_memory=logs/ashare_rank_mem0 \
  --out logs/ashare_rank_report.md

# Generate the unified monthly trader benchmark (12 decisions, 16 stocks each)
python scripts/ashare/gen_trader_tasks.py

# Clean trader: one 16-stock decision per month, no cross-month memory
python main.py common-benchmark \
  --config_file_name=agent_ashare_trader_kimi \
  output_dir=logs/ashare_trader_clean

# Memory trader: one explicit ID is shared inside this run only.
# Use a new ID for every independent experiment/repeat.
ASHARE_TRADER_RUN_ID=kimi_r1 python main.py common-benchmark \
  --config_file_name=agent_ashare_trader_mem0_kimi \
  output_dir=logs/ashare_trader_mem0_r1

# Portfolio backtest: ¥1m default capital, 5bp buy / 15bp sell, ¥5 minimum
python scripts/ashare/eval_trader.py \
  --run clean=logs/ashare_trader_clean \
  --run memory=logs/ashare_trader_mem0_r1 \
  --out logs/ashare_trader_report.md
```

`ashare-trader` is a single logical trader, not one permanent chat transcript.
Each monthly task starts a fresh auditable LLM session, while the run-scoped
Mem0 namespace carries only matured factual episodes and statistically
validated factor reliability forward. `max_concurrent=1` serializes execution;
`task_order=monthly` plus `available_after=exit_date` provides the walk-forward
state boundary.

The generator also enforces a single-capital-account invariant. If a calendar
month's first trading day falls before the previous 20-session position has
liquidated (common around long holidays), that month's decision shifts to the
previous liquidation close. There is still one decision in each calendar month,
but no two portfolios reuse the same capital at the same time.

The trader may freely inspect all point-in-time data available in the local
Tushare cache via any ashare-market tool.  Omit `lookback_days` or pass `0`
to return every row on or before the decision date; there is no artificial
lookback cap beyond what the cache contains.

## Qdrant and Memory Operations

Connection settings are read from the environment and contain no secrets:

- `MEM0_QDRANT_HOST` (default `127.0.0.1`)
- `MEM0_QDRANT_PORT` (default `6333`)
- `MEM0_QDRANT_COLLECTION` (default `miromemskill`)
- `MEM0_HISTORY_DB_PATH` (default `memory_bank/mem0_history.db`)
- `MEM0_EMBEDDING_MODEL` (default `embedding-3`)
- `MEM0_EMBEDDING_DIMS` (default `2048`)
- `ASHARE_TRADER_RUN_ID` (required by the trader-memory profile; unique per run)

The embedding endpoint uses `GLM_API_KEY`/`GLM_BASE_URL`. Mem0 reflection uses
`REFLECTION_LLM_*` when set, otherwise the existing `DEEPSEEK_*` settings.

Runtime data locations:

- Qdrant vectors: Docker volume `miromem_qdrant_data`
- Mem0 history: `memory_bank/mem0_history.db`
- Structured statistical ledgers: `memory_bank/*_samples.jsonl` and
  `memory_bank/*_outcomes.jsonl`
- Static, versioned skills: `memory_bank/skills/` and
  `memory_bank/skills_ashare/`
- Verified local archives: `.memory_archives/<UTC timestamp>/`

Useful lifecycle commands:

```bash
# Inspect service and application-side runtime files
docker compose -f deploy/qdrant/compose.yaml ps
python scripts/memory_admin.py status

# Stop Qdrant while retaining its volume
docker compose -f deploy/qdrant/compose.yaml stop

# Restart after a host/container failure; persisted data is reused
docker compose -f deploy/qdrant/compose.yaml up -d

# Cold reset: first stop all benchmark/MCP writers, then archive sidecars/history
python scripts/memory_admin.py archive-reset --dry-run
python scripts/memory_admin.py archive-reset
python scripts/memory_admin.py qdrant-reset --yes
```

`archive-reset` verifies every archived file by SHA256 before deleting it and
never touches either skills directory. `qdrant-reset` removes the configured
collection and Mem0 SQLite history; the next application start recreates them
empty. Do not use `docker compose down -v` unless destruction of the persistent
Qdrant volume is explicitly intended.

The old `src/memory/vector_store.py` JSONL store remains only for offline
compatibility tests. Production profiles and the MCP server do not read
`{namespace}_memories.jsonl`.

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

Ready-made unified trader arms:

- `agent_ashare_trader_kimi`
- `agent_ashare_trader_mem0_kimi` (requires `ASHARE_TRADER_RUN_ID`)

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
| Memory store | `src/memory/official_mem0_store.py` + `src/memory/memory.py` |
| Qdrant deployment | `deploy/qdrant/compose.yaml` |
| Memory administration | `scripts/memory_admin.py` |
| Skill library | `src/memory/skills.py` + `memory_bank/skills_ashare/` |
| Direction reflection | `src/memory/rolling_reflection.py` |
| Ranking reflection | `src/memory/rank_reflection.py` |
| Trader parser/math | `src/utils/ashare_trader.py` |
| Trader history panel | `src/utils/ashare_trader_features.py` |
| Momentum soft anchor | `src/utils/ashare_momentum.py` + `memory_bank/skills_ashare/ashare_momentum_relative_strength/` |
| Post-tool evidence | `src/memory/feature_evidence.py` |
| MCP tools | `src/tool/mcp_servers/memskill_mcp_server.py` + `src/tool/mcp_servers/ashare_mcp_server.py` |
| A-share rank evaluator | `scripts/ashare/eval_rank.py` |
| A-share trader evaluator | `scripts/ashare/eval_trader.py` |
| Agent configs | `config/agent_ashare_*.yaml` |
