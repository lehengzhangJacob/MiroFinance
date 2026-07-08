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


def evaluate(name: str, merged: pd.DataFrame) -> dict:
    """merged: columns [prediction, label, excess_return, entry_date]."""
    n = len(merged)
    valid = merged[merged.prediction.notna()]
    hit = (valid.prediction == valid.label).mean() if len(valid) else float("nan")

    # Long book: hold predicted outperformers, earn their excess return.
    monthly = []
    for date, g in merged.groupby("entry_date"):
        picks = g[g.prediction == OUTPERFORM]
        monthly.append(picks.excess_return.mean() if len(picks) else 0.0)
    monthly = pd.Series(monthly, dtype=float)

    cum = (1 + monthly).prod() - 1
    ann_sharpe = (
        monthly.mean() / monthly.std() * math.sqrt(12)
        if len(monthly) > 1 and monthly.std() > 0
        else float("nan")
    )
    nav = (1 + monthly).cumprod()
    dd = ((nav - nav.cummax()) / nav.cummax()).min() if len(nav) else float("nan")

    picked = merged[merged.prediction == OUTPERFORM]
    return {
        "run": name,
        "n_tasks": n,
        "n_valid_pred": len(valid),
        "hit_rate": round(hit * 100, 1),
        "mean_excess_of_picks": round(picked.excess_return.mean() * 100, 2) if len(picked) else float("nan"),
        "cum_excess": round(cum * 100, 2),
        "ann_sharpe": round(ann_sharpe, 2) if not math.isnan(ann_sharpe) else float("nan"),
        "max_dd": round(dd * 100, 2) if not math.isnan(dd) else float("nan"),
        "monthly": monthly,
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

    rnd = tasks.copy()
    random.seed(42)
    rnd["prediction"] = [random.choice([OUTPERFORM, UNDERPERFORM]) for _ in range(len(rnd))]
    results.append(evaluate("random(随机)", rnd))

    mom = tasks.copy()
    mom["prediction"] = mom.task_id.map(momentum_rule(tasks))
    results.append(evaluate("momentum-rule(20日动量)", mom))

    lines = [
        "# A股超额收益方向预测回测报告",
        "",
        f"任务: {len(tasks)} 条 (6 股 x 12 月, 20 交易日持有, 相对沪深300)",
        "",
        "| 运行 | 预测数 | 方向命中率% | 入选组合平均超额% | 累计超额% | 年化夏普 | 最大回撤% |",
        "|------|--------|------------|------------------|-----------|----------|-----------|",
    ]
    for r in results:
        lines.append(
            f"| {r['run']} | {r['n_valid_pred']}/{r['n_tasks']} | {r['hit_rate']} "
            f"| {r['mean_excess_of_picks']} | {r['cum_excess']} | {r['ann_sharpe']} | {r['max_dd']} |"
        )

    lines += ["", "## 逐月超额收益(做多「跑赢」组合)", ""]
    months = sorted(tasks.entry_date.unique())
    header = "| 运行 | " + " | ".join(m[:6] for m in months) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(months) + 1))
    for r in results:
        vals = " | ".join(f"{v*100:+.2f}" for v in r["monthly"])
        lines.append(f"| {r['run']} | {vals} |")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nreport -> {out}")
    for line in lines[4:12]:
        print(line)


if __name__ == "__main__":
    main()
