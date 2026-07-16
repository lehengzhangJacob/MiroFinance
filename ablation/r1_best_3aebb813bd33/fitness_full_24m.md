# Main experiment R1 — full_24m evaluation (skill-only)

Skill `3aebb813bd33`, memory OFF, continuous 24 months.
Source run: `ablation/runs/skillonly_r1_24m`.
Sharpe = sqrt(12) x mean(monthly net) / stdev(monthly net), rf=0.

| segment | total | index | excess | maxDD | sharpe | win |
|---|---:|---:|---:|---:|---:|---:|
| full_24m | +112.88% | +33.76% | +79.12% | -16.54% | 1.44 | 71% |
| formal_12m | +0.50% | +13.98% | -13.47% | -16.54% | 0.13 | 58% |
| dev_6m | +44.03% | +15.53% | +28.50% | -5.09% | 1.94 | 67% |
| holdout_6m | +46.55% | +1.58% | +44.98% | 0.00% | 4.88 | 100% |
