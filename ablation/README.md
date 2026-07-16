# Ablation experiments (separate keys, never the main evolution key)

This directory is for **controlled ablations**, isolated from the Hermes
R2–R4 evolution chain.

| Job | API key | Purpose |
|-----|---------|---------|
| Hermes evolution (`MiroMemSkill_hermes`) | `own_glm` | ongoing skill search |
| Full (R1+memory) arm (`run_skill_ablation.py` → `r1_best`) | `own_glm3` | FINAL system cell |
| Memory-only (`run_memonly_ablation.py`) | `own_glm4` | leave-one-out: drop skill |
| Skill-only R1 (`run_skillonly_ablation.py`) | `own_glm5` | leave-one-out: drop memory |
| Plain control (`run_plain_ablation.py`) | `own_glm6` | leave-one-out: drop both |

**Final skill (temporary champion):** `3aebb813bd33`

Do not commit key files (`own_*` is gitignored).

## Artifacts

| Path | Role |
|------|------|
| `baseline_0a931278001c/` | Pre-R1 open-universe skill |
| `r1_best_3aebb813bd33/` | R1 promoted skill + original formal fitness |
| `skill_r1_3aebb813bd33/` | Clean single-`.md` mount of FINAL R1 (for controller) |
| `no_skill_placeholder/` | Mounted for no-skill arms; never read or injected |
| `run_skill_ablation.py` | Memory-ON re-run: baseline vs R1 on the 24m snapshot |
| `run_skill_ablation_nohup.sh` | nohup launcher (forces `own_glm3`) |
| `run_memonly_ablation.py` | Memory-only arm (mem_only) on the 24m snapshot |
| `run_memonly_ablation_nohup.sh` | nohup launcher (forces `own_glm4`) |
| `run_plain_ablation.py` | Plain control arm: no skill, no memory |
| `run_plain_ablation_nohup.sh` | nohup launcher (forces `own_glm6`) |
| `run_skillonly_ablation.py` | Skill-only of FINAL R1 (`3aebb813bd33`), memory OFF |
| `run_skillonly_ablation_nohup.sh` | nohup launcher (forces `own_glm5`) |
| `build_ablation_matrix.py` | Leave-one-out 2×2 around FINAL R1 -> `reports/matrix_24m.*` |
| `runs/` | Ablation rollout outputs |
| `reports/` | Cross-run matrix reports |
| `logs/` | nohup logs / locks |

## Ablation matrix (leave-one-out around FINAL R1 `3aebb813bd33`)

| memory \ skill | none | Skill `0a931278001c`（进化前） | R1 `3aebb813bd33` |
|---|---|---|---|
| **off** | `plain` (own_glm6) | — | `skill_only_r1` (own_glm5) |
| **on** | **w/o skill** / `mem_only` (own_glm4) | **w/o self-evolve** (own_glm3 baseline) | `full_r1` (own_glm3) = **FINAL** |

`build_ablation_matrix.py` pairs against `full_r1` when ready (else `plain`).
Incomplete arms are skipped; re-run any time.

Headline numbers also live in the repo-root [`README.md`](../README.md) 消融对照表.

### Completed: w/o self-evolve (24 months)

`mem_ablation_24m` / arm `baseline` — Memory ON, Skill=`0a931278001c`, no Hermes evolution.

| Segment | Window | Total | Index | Excess | MaxDD |
|---------|--------|------:|------:|-------:|------:|
| **full_24m** | 2024-07..2026-06 | **+34.48%** | +33.76% | **+0.73%** | -16.62% |
| formal_12m | 2024-07..2025-06 | +18.00% | +13.98% | +4.03% | -16.62% |
| dev_6m | 2025-07..2025-12 | +22.04% | +15.53% | +6.51% | -11.18% |
| holdout_6m | 2026-01..2026-06 | -6.18% | +1.58% | -7.76% | -10.99% |

### Completed: w/o skill (24 months)

`memonly_ablation_24m` / arm `mem_only` — Memory ON, **no Skill** (`own_glm4`).

| Segment | Window | Total | Index | Excess | MaxDD |
|---------|--------|------:|------:|-------:|------:|
| **full_24m** | 2024-07..2026-06 | **+22.30%** | +33.76% | **-11.46%** | -26.92% |
| formal_12m | 2024-07..2025-06 | -11.88% | +13.98% | -25.86% | -26.92% |
| dev_6m | 2025-07..2025-12 | +19.16% | +15.53% | +3.63% | -10.16% |
| holdout_6m | 2026-01..2026-06 | +16.66% | +1.58% | +15.08% | -2.20% |

vs w/o self-evolve on full_24m: total **-12.18pp**, excess **-12.19pp**.

### vs main experiment R1 (`r1_best_3aebb813bd33/`)

Sealed skill-only protocol from `formal24m_20260715` (Memory OFF). Ablation cells are Memory ON — same snapshot/evaluator, different protocol.

| Cell | Protocol | Dev total | Holdout total | Holdout excess |
|------|----------|----------:|--------------:|---------------:|
| **Main R1** `3aebb813bd33` | skill-only | **+83.53%** | **+38.95%** | **+37.37%** |
| Main baseline skill | skill-only | +29.35% | -1.89% | -3.47% |
| w/o self-evolve | memory + `0a931…` | +22.04% | -6.18% | -7.76% |
| w/o skill | memory, no skill | +19.16% | +16.66% | +15.08% |

Source: [`r1_best_3aebb813bd33/fitness_dev.*`](r1_best_3aebb813bd33/) / `fitness_holdout.*`. Full narrative in repo-root README.

## Memory-ON skill ablation (24 months)

Question answered: **does the R1-evolved skill text still help when the full
MemSkill memory system is running?** Both arms are the complete MemSkill
trader runtime — episodic trader memory recorded at every month barrier and
injected only after each episode's exit date matures (no lookahead). There is
no Hermes evolution loop anywhere in this run. The ONLY difference between
the two arms is the injected skill Markdown:

- **Baseline arm**: skill `0a931278001c` (pre-evolution)
- **Treatment arm**: skill `3aebb813bd33` (R1 best)
- **Window**: 2024-07 … 2026-06 (24 months, `ashare_open_stocks_glm52_24m_20260715`)
- **Config**: `agent_ashare_trader_open_hermes_memfull_glm`
  (memory = `ashare_trader_hermes_memfull`: trader episodes ON, per-arm
  store/namespace isolation, full-body skill injection)
- **Seeds**: 1 per arm (sequential monthly run; memory accumulates within the arm)

```bash
cd /home/msj_team/Jacob/agent
nohup bash ablation/run_skill_ablation_nohup.sh mem_ablation_24m \
  > ablation/logs/mem_ablation_24m.log 2>&1 &
tail -f ablation/logs/mem_ablation_24m.log
```

Reports land in `ablation/runs/<run_id>/reports/`:

| Report | Months | Meaning |
|--------|--------|---------|
| `fitness_full_24m` | 2024-07..2026-06 | headline paired comparison |
| `fitness_formal_12m` | 2024-07..2025-06 | original MemSkill-Full window |
| `fitness_dev_6m` | 2025-07..2025-12 | R1 selection window (cross-reference) |
| `fitness_holdout_6m` | 2026-01..2026-06 | R1 sealed holdout window (cross-reference) |

`summary.json` aggregates all segments plus per-arm episode-write counts
(sanity check that memory was actually ON).

Caveat: months up to 2025-12 overlap R1's train/dev search data, so treat
`full_24m` as an in-sample interaction check; `holdout_6m` is the only
segment that is out-of-sample for the R1 skill.

## Memory-only ablation (24 months, `own_glm4`)

Leave-one-out: trader-episode memory ON (exit-date embargo), NO skill.
Single arm under `agent_ashare_trader_open_hermes_memonly_glm`.

```bash
cd /home/msj_team/Jacob/agent
nohup bash ablation/run_memonly_ablation_nohup.sh memonly_ablation_24m \
  > ablation/logs/memonly_ablation_24m.log 2>&1 &
tail -f ablation/logs/memonly_ablation_24m.log
```

Post-verified: no `Top Skill Preview` in any prompt; episode writes and
matured-episode injections present.

Note: the plain control arm originally queued inside this run was split out
to `plain_ablation_24m` (own_glm6) to run in parallel; a placeholder at
`runs/memonly_ablation_24m/arms/plain/out/BLOCKED.md` makes the in-flight
process skip it (expected fast-fail after mem_only completes).

## Plain control (24 months, `own_glm6`)

Leave-one-out: drop BOTH skill and memory (`agent_ashare_trader_open_glm`,
no memory section). Pairing baseline for the whole matrix when Full is
unfinished.

```bash
cd /home/msj_team/Jacob/agent
nohup bash ablation/run_plain_ablation_nohup.sh plain_ablation_24m \
  > ablation/logs/plain_ablation_24m.log 2>&1 &
tail -f ablation/logs/plain_ablation_24m.log
```

Post-verified: zero skill previews, zero episode writes, zero memory blocks.

Interpretation notes:

- Each cell is an independent single-seed run on the same frozen snapshot
  and model config; cross-cell comparisons are descriptive.
- Months through 2025-12 were seen during R1's evolution search, so for any
  cell involving the R1 skill only `holdout_6m` (2026-01..06) is strictly
  out-of-sample.

## Skill-only ablation of FINAL R1 (24 months, `own_glm5`)

Leave-one-out: keep skill `3aebb813bd33`, turn memory OFF.
Single arm under `agent_ashare_trader_open_hermes_glm`.

```bash
cd /home/msj_team/Jacob/agent
nohup bash ablation/run_skillonly_ablation_nohup.sh skillonly_r1_24m \
  > ablation/logs/skillonly_r1_24m.log 2>&1 &
tail -f ablation/logs/skillonly_r1_24m.log
```

Post-verified: skill preview every month, zero episode writes/injections.

## Formal R1 sealed numbers (skill-only protocol, already recorded)

From `formal24m_20260715` (not re-run here):

| Split | Baseline | R1 best | Paired | W-L |
|-------|---------:|--------:|-------:|----:|
| Dev 2025-07..12 | +29.35% | +83.53% | +6.78pp | 5-1 |
| Holdout 2026-01..06 | -1.89% | +38.95% | +5.94pp | 5-1 |
