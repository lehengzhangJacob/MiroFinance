# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

"""Aggregate agent predictions into a portfolio backtest and comparison report.

Reads one or more benchmark run dirs (task_*_attempt_1.json), maps each
prediction to a monthly long-only position (hold stocks predicted to
outperform, equal weight; excess return vs CSI 300), and reports:
hit rate, mean excess per pick, cumulative long-short curve, annualized
Sharpe of monthly excess, max drawdown. Rule/random baselines included.

Usage:
    uv run python scripts/ashare/backtest.py \
        --run baseline=../MiroFlow/logs/ashare_full \
        --run memskill=../MiroMemSkill/logs/ashare_full \
        --out logs/ashare_report.md
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "ashare"

OUTPERFORM, UNDERPERFORM = "跑赢", "跑输"


def load_tasks() -> pd.DataFrame:
    """Ground-truth panel from the generated benchmark JSONL."""
    rows = []
    task_file = REPO_ROOT / "data" / "ashare_pred" / "standardized_data.jsonl"
    with open(task_file, encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            m = t["metadata"]
            rows.append(
                {
                    "task_id": t["task_id"],
                    "ts_code": m["ts_code"],
                    "stock_name": m["stock_name"],
                    "entry_date": m["entry_date"],
                    "excess_return": m["excess_return"],
                    "label": t["ground_truth"],
                }
            )
    return pd.DataFrame(rows)


def extract_prediction(boxed: str | None) -> str | None:
    if not boxed:
        return None
    text = str(boxed).strip()
    if OUTPERFORM in text and UNDERPERFORM not in text:
        return OUTPERFORM
    if UNDERPERFORM in text and OUTPERFORM not in text:
        return UNDERPERFORM
    return None


def load_run(run_dir: str) -> pd.DataFrame:
    rows = []
    for f in glob.glob(f"{run_dir}/task_*_attempt_1.json"):
        d = json.load(open(f, encoding="utf-8"))
        if d.get("status") != "completed":
            continue
        rows.append(
            {
                "task_id": d.get("task_id") or d.get("task_name"),
                "prediction": extract_prediction(d.get("final_boxed_answer")),
                "judge_result": d.get("judge_result"),
            }
        )
    return pd.DataFrame(rows)


def arima_rule(tasks: pd.DataFrame) -> pd.Series:
    """Course-ch.10-style ARIMA baseline (course notebook: log returns, ADF,
    AIC order selection). Point-in-time: fit on data <= entry date only,
    forecast the 20d horizon for stock and index, predict outperform when the
    stock's cumulative forecast log return exceeds the index's."""
    import warnings

    import numpy as np
    from statsmodels.tsa.arima.model import ARIMA

    idx = pd.read_csv(DATA_DIR / "index_000300.SH.csv", dtype={"trade_date": str})
    idx = idx.sort_values("trade_date").reset_index(drop=True)

    def forecast_sum(closes: pd.Series, horizon: int = 20, train: int = 250) -> float:
        rets = np.log(closes.astype(float)).diff().dropna().iloc[-train:]
        best_aic, best_sum = float("inf"), 0.0
        for order in [(1, 0, 0), (0, 0, 1), (1, 0, 1), (2, 0, 1), (1, 0, 2)]:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fit = ARIMA(rets.values, order=order).fit()
                if fit.aic < best_aic:
                    best_aic = fit.aic
                    best_sum = float(fit.forecast(horizon).sum())
            except Exception:
                continue
        return best_sum

    preds = {}
    cache: dict[tuple[str, str], float] = {}
    for _, row in tasks.iterrows():
        d = row.entry_date
        key_s = (row.ts_code, d)
        if key_s not in cache:
            df = pd.read_csv(DATA_DIR / f"daily_{row.ts_code}.csv", dtype={"trade_date": str})
            df = df.sort_values("trade_date").reset_index(drop=True)
            cache[key_s] = forecast_sum(df[df.trade_date <= d].close_qfq)
        key_i = ("index", d)
        if key_i not in cache:
            cache[key_i] = forecast_sum(idx[idx.trade_date <= d].close)
        preds[row.task_id] = OUTPERFORM if cache[key_s] > cache[key_i] else UNDERPERFORM
    return pd.Series(preds)


def momentum_rule(tasks: pd.DataFrame) -> pd.Series:
    """Simple 20d relative-momentum baseline computed from the local cache."""
    idx = pd.read_csv(DATA_DIR / "index_000300.SH.csv", dtype={"trade_date": str})
    idx = idx.sort_values("trade_date").reset_index(drop=True)
    preds = {}
    for _, row in tasks.iterrows():
        df = pd.read_csv(DATA_DIR / f"daily_{row.ts_code}.csv", dtype={"trade_date": str})
        df = df.sort_values("trade_date").reset_index(drop=True)
        d = row.entry_date
        try:
            p = df.index[df.trade_date == d][0]
            q = idx.index[idx.trade_date == d][0]
            if p < 20 or q < 20:
                raise IndexError
            s_mom = df.close_qfq.iloc[p] / df.close_qfq.iloc[p - 20] - 1
            i_mom = idx.close.iloc[q] / idx.close.iloc[q - 20] - 1
            preds[row.task_id] = OUTPERFORM if s_mom > i_mom else UNDERPERFORM
        except IndexError:
            preds[row.task_id] = OUTPERFORM
    return pd.Series(preds)


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% score interval for a binomial proportion (as percentages)."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z / denom * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return round((center - half) * 100, 1), round((center + half) * 100, 1)


def _cum(series: pd.Series) -> float:
    return (1 + series).prod() - 1


def _sharpe(series: pd.Series) -> float:
    if len(series) > 1 and series.std() > 0:
        return series.mean() / series.std() * math.sqrt(12)
    return float("nan")


def evaluate(name: str, merged: pd.DataFrame) -> dict:
    """merged: columns [prediction, label, excess_return, entry_date]."""
    n = len(merged)
    valid = merged[merged.prediction.notna()]
    hit = (valid.prediction == valid.label).mean() if len(valid) else float("nan")

    # Long book: hold predicted outperformers, earn their excess return.
    # Long-short spread: long predicted outperformers, short predicted
    # underperformers (equal weight per leg). Selection-free strategies
    # (e.g. always-long) have an empty short leg -> spread 0: the LS metric
    # only rewards actual discrimination between stocks.
    monthly, monthly_ls = [], []
    for date, g in merged.groupby("entry_date"):
        long_leg = g[g.prediction == OUTPERFORM].excess_return
        short_leg = g[g.prediction == UNDERPERFORM].excess_return
        monthly.append(long_leg.mean() if len(long_leg) else 0.0)
        if len(long_leg) and len(short_leg):
            monthly_ls.append(long_leg.mean() - short_leg.mean())
        else:
            monthly_ls.append(0.0)
    monthly = pd.Series(monthly, dtype=float)
    monthly_ls = pd.Series(monthly_ls, dtype=float)

    nav = (1 + monthly).cumprod()
    dd = ((nav - nav.cummax()) / nav.cummax()).min() if len(nav) else float("nan")

    picked = merged[merged.prediction == OUTPERFORM]
    n_hits = int((valid.prediction == valid.label).sum()) if len(valid) else 0
    ci_lo, ci_hi = _wilson_ci(n_hits, len(valid))
    return {
        "run": name,
        "n_tasks": n,
        "n_valid_pred": len(valid),
        "hit_rate": round(hit * 100, 1),
        "hit_ci": f"[{ci_lo},{ci_hi}]" if len(valid) else "-",
        "mean_excess_of_picks": round(picked.excess_return.mean() * 100, 2) if len(picked) else float("nan"),
        "cum_excess": round(_cum(monthly) * 100, 2),
        "ann_sharpe": round(_sharpe(monthly), 2) if not math.isnan(_sharpe(monthly)) else float("nan"),
        "max_dd": round(dd * 100, 2) if not math.isnan(dd) else float("nan"),
        "cum_ls": round(_cum(monthly_ls) * 100, 2),
        "ls_sharpe": round(_sharpe(monthly_ls), 2) if not math.isnan(_sharpe(monthly_ls)) else float("nan"),
        "monthly": monthly,
        "monthly_ls": monthly_ls,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", default=[], help="name=path/to/logs")
    ap.add_argument("--out", default="logs/ashare_report.md")
    args = ap.parse_args()

    tasks = load_tasks()
    results = []

    for spec in args.run:
        name, _, path = spec.partition("=")
        run = load_run(path)
        if run.empty:
            print(f"[skip] {name}: no completed tasks in {path}")
            continue
        merged = tasks.merge(run, on="task_id", how="inner")
        print(f"{name}: {len(merged)} predictions matched")
        results.append(evaluate(name, merged))

    # Reference baselines on the full task panel.
    always = tasks.copy()
    always["prediction"] = OUTPERFORM
    results.append(evaluate("always-long(全做多)", always))

    # Random: average over many draws, not one lucky seed.
    n_sims = 200
    sims = []
    for seed in range(n_sims):
        rnd = tasks.copy()
        rng = random.Random(seed)
        rnd["prediction"] = [rng.choice([OUTPERFORM, UNDERPERFORM]) for _ in range(len(rnd))]
        sims.append(evaluate("", rnd))
    avg_hit = round(sum(s["hit_rate"] for s in sims) / n_sims, 1)
    avg_n = sims[0]["n_valid_pred"]
    avg_ci = _wilson_ci(round(avg_hit / 100 * avg_n), avg_n)
    avg = {
        "run": f"random(随机,{n_sims}次均值)",
        "n_tasks": sims[0]["n_tasks"],
        "n_valid_pred": avg_n,
        "hit_rate": avg_hit,
        "hit_ci": f"[{avg_ci[0]},{avg_ci[1]}]",
        "mean_excess_of_picks": round(sum(s["mean_excess_of_picks"] for s in sims) / n_sims, 2),
        "cum_excess": round(sum(s["cum_excess"] for s in sims) / n_sims, 2),
        "ann_sharpe": round(sum(s["ann_sharpe"] for s in sims) / n_sims, 2),
        "max_dd": round(sum(s["max_dd"] for s in sims) / n_sims, 2),
        "cum_ls": round(sum(s["cum_ls"] for s in sims) / n_sims, 2),
        "ls_sharpe": round(sum(s["ls_sharpe"] for s in sims) / n_sims, 2),
        "monthly": sum(s["monthly"] for s in sims) / n_sims,
        "monthly_ls": sum(s["monthly_ls"] for s in sims) / n_sims,
    }
    results.append(avg)

    mom = tasks.copy()
    mom["prediction"] = mom.task_id.map(momentum_rule(tasks))
    results.append(evaluate("momentum-rule(20日动量)", mom))

    # Reference-only row; never let it block the main report (needs statsmodels).
    try:
        ari = tasks.copy()
        ari["prediction"] = ari.task_id.map(arima_rule(tasks))
        results.append(evaluate("arima(统计基线,仅参考)", ari))
    except Exception as exc:
        print(f"[skip] arima baseline: {exc}")

    lines = [
        "# A股超额收益方向预测回测报告",
        "",
        f"任务: {len(tasks)} 条 (6 股 x 12 月, 20 交易日持有, 相对沪深300)",
        "",
        "多空对冲(LS) = 做多预测「跑赢」腿 - 做空预测「跑输」腿(等权)。",
        "无选股信息的策略(如全做多)缺少空头腿, LS 记 0, 因此该列只衡量真实的个股区分能力。",
        "",
        "| 运行 | 预测数 | 方向命中率% | 命中率95%CI | 入选组合平均超额% | 多头累计超额% | 多头夏普 | 最大回撤% | LS累计% | LS夏普 |",
        "|------|--------|------------|-------------|------------------|--------------|----------|-----------|---------|--------|",
    ]
    for r in results:
        lines.append(
            f"| {r['run']} | {r['n_valid_pred']}/{r['n_tasks']} | {r['hit_rate']} | {r.get('hit_ci', '-')} "
            f"| {r['mean_excess_of_picks']} | {r['cum_excess']} | {r['ann_sharpe']} | {r['max_dd']} "
            f"| {r['cum_ls']} | {r['ls_sharpe']} |"
        )

    months = sorted(tasks.entry_date.unique())
    for key, title in [("monthly", "逐月超额收益(做多「跑赢」组合)"), ("monthly_ls", "逐月多空对冲收益(LS)")]:
        lines += ["", f"## {title}", ""]
        header = "| 运行 | " + " | ".join(m[:6] for m in months) + " |"
        lines.append(header)
        lines.append("|" + "---|" * (len(months) + 1))
        for r in results:
            vals = " | ".join(f"{v*100:+.2f}" for v in r[key])
            lines.append(f"| {r['run']} | {vals} |")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nreport -> {out}")
    for line in lines[7 : 9 + len(results)]:
        print(line)


if __name__ == "__main__":
    main()
