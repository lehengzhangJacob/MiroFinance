# R1 best evolved skill (formal ablation artifact)

- **short_id**: `3aebb813bd33`
- **run**: `formal24m_20260715`
- **parent**: baseline open-universe skill (`0a931278001c`)
- **source**: `MiroMemSkill_hermes/.evolution/candidates/3aebb813bd33/ashare_open_portfolio.md`
- **protocol**: fixed 12/6/6 train/dev/holdout on snapshot `ashare_open_stocks_glm52_24m_20260715`

## Headline fitness (vs pre-R1 baseline skill)

| Split | Baseline cum | Candidate cum | Mean paired | W-L | Gates |
|-------|-------------:|--------------:|------------:|----:|:-----:|
| Dev (2025-07..12) | +29.35% | +83.53% | +6.78pp | 5-1 | PASS |
| Holdout (2026-01..06) | -1.89% | +38.95% | +5.94pp | 5-1 | PASS |

This is the **primary formal Hermes R1 selection result**: sealed holdout for promotion.
Later R2+ rounds are exploratory. **Post-evolution evaluation** uses the full
24-month window (see `fitness_full_24m.*`), not the selection splits.

## Files

- `ashare_open_portfolio.md` — promoted R1 skill text
- `fitness_dev.json` / `.md` — selection: paired evaluation on 2025-07..12
- `fitness_holdout.json` / `.md` — selection: sealed evaluation on 2026-01..06
- `fitness_full_24m.json` / `.md` — **evaluation**: continuous skill-only 24m (`skillonly_r1_24m`, +112.88%)
