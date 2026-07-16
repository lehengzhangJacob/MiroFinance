# Experiment figures

The figures use the frozen 12-month results in:

`shared/ashare_open_stocks_glm52_20260714/reports/ashare_open_flow_vs_memskill_20260714_memfix02_full.md`

Regenerate them with the isolated plotting environment:

```bash
MPLCONFIGDIR=/tmp/miroplot-mpl \
  /home/msj_team/.conda/envs/MiroPlot/bin/python generate_experiment_figures.py
```

No benchmark, market API, or database query is run by the script.
