# SPDX-FileCopyrightText: 2026 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

"""Paired single-agent vs multi-agent comparison on the frozen open snapshot.

Both architectures are replayed with the deterministic evaluator
(``eval_open_trader``: lot rounding, real fees, sequential compounding) on the
pristine snapshot DB. The single-agent arms come from the hermes fork's formal
run (baseline_train / baseline_dev / baseline); the multi-agent arms come from
this fork's run (train / dev / holdout). Positive paired diffs mean the
multi-agent arm beat the single-agent arm that month.

Usage::

    python scripts/ashare/compare_architecture.py \
      --multi_root=.evolution/runs/<run_id> \
      --single_root=/home/msj_team/Jacob/agent/MiroMemSkill_hermes/.evolution/runs/formal24m_20260715 \
      --out=.evolution/runs/<run_id>/reports/architecture_compare.md
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.evolution import fitness as fitness_mod  # noqa: E402
from src.evolution.controller import (  # noqa: E402
    DEFAULT_SNAPSHOT,
    render_report_markdown,
)
from src.evolution.splits import filter_tasks, load_tasks, make_splits  # noqa: E402

LEVELS = ("train", "dev", "holdout")
SINGLE_ARMS = {"train": "baseline_train", "dev": "baseline_dev", "holdout": "baseline"}
MULTI_ARMS = {"train": "train", "dev": "dev", "holdout": "holdout"}


def _evaluate_merged(
    out_dirs: list[Path], tasks: list[dict], db_path: Path
) -> dict:
    """Replay one architecture's merged monthly allocations across all months."""
    ev = fitness_mod.evaluator()
    allocations: dict = {}
    for out_dir in out_dirs:
        allocations.update(ev.extract_run_allocations(Path(out_dir)))
    conn = sqlite3.connect(str(db_path))
    try:
        market = ev.Market(conn)
        total, months = ev.replay(market, tasks, allocations)
    finally:
        conn.close()
    metrics = ev.replay_metrics(total, months)
    index_total = 1.0
    for month in months:
        index_total *= 1.0 + float(month.get("index", 0.0))
    return {
        "months": months,
        "total_return": total,
        "index_return": index_total - 1.0,
        "excess_return": total - (index_total - 1.0),
        "max_drawdown": metrics["max_drawdown"],
        "worst_month": metrics["worst_month"],
        "win_rate": metrics["win_rate"],
        "fees": metrics["fees"],
        "invalid_months": [m["as_of"] for m in months if "note" in m],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--multi_root", required=True, help="multi-agent run dir")
    parser.add_argument("--single_root", required=True, help="single-agent run dir")
    parser.add_argument("--snapshot", default=str(DEFAULT_SNAPSHOT))
    parser.add_argument("--train_months", type=int, default=12)
    parser.add_argument("--dev_months", type=int, default=6)
    parser.add_argument("--holdout_months", type=int, default=6)
    parser.add_argument("--out", default="", help="markdown report path")
    args = parser.parse_args()

    snapshot = Path(args.snapshot)
    db_path = snapshot / "ashare_pools_snapshot.db"
    tasks = load_tasks(
        snapshot / "tasks" / "ashare_trader_open" / "standardized_data.jsonl"
    )
    splits = make_splits(
        tasks,
        train_months=args.train_months,
        dev_months=args.dev_months,
        holdout_months=args.holdout_months,
    )

    multi_root = Path(args.multi_root)
    single_root = Path(args.single_root)

    sections: list[str] = ["# 单智能体 vs 多智能体 架构对比\n"]
    sections.append(
        f"- baseline = 单智能体（{single_root}）\n"
        f"- candidate = 多智能体（{multi_root}）\n"
        f"- 快照: {snapshot}\n"
        "- 正的配对差值表示多智能体当月更优。\n"
    )
    summary: dict = {"levels": {}}

    single_outs: list[Path] = []
    multi_outs: list[Path] = []
    for level in LEVELS:
        months = splits.level_months(level)
        subset = filter_tasks(tasks, months)
        single_out = single_root / "arms" / SINGLE_ARMS[level] / "out"
        multi_out = multi_root / "arms" / MULTI_ARMS[level] / "out"
        if not single_out.exists() or not multi_out.exists():
            sections.append(f"## {level}\n\n缺少输出目录，跳过（single={single_out.exists()} multi={multi_out.exists()}）。\n")
            continue
        single_outs.append(single_out)
        multi_outs.append(multi_out)
        single_arm = fitness_mod.evaluate_arm(single_out, subset, db_path)
        multi_arm = fitness_mod.evaluate_arm(multi_out, subset, db_path)
        report = fitness_mod.fitness_report(level, single_arm, multi_arm)
        report["run_id"] = multi_root.name
        report["months"] = list(months)
        summary["levels"][level] = {
            "single_total": report["baseline"]["total_return"],
            "multi_total": report["candidate"]["total_return"],
            "mean_diff_pp": report["paired"]["mean_diff_pp"],
            "sign_test_p": report["paired"]["sign_test_p"],
            "gates_passed": report["gates"]["passed"],
        }
        sections.append(render_report_markdown(report))

    # Full-window chained replay (train+dev+holdout in chronological order).
    if single_outs and multi_outs:
        all_months = splits.train + splits.dev + splits.holdout
        subset = filter_tasks(tasks, all_months)
        single_full = _evaluate_merged(single_outs, subset, db_path)
        multi_full = _evaluate_merged(multi_outs, subset, db_path)
        paired = fitness_mod.paired_stats(
            single_full["months"], multi_full["months"]
        )
        summary["full_window"] = {
            "months": len(all_months),
            "single_total": round(single_full["total_return"], 6),
            "multi_total": round(multi_full["total_return"], 6),
            "index_total": round(single_full["index_return"], 6),
            "single_max_drawdown": round(single_full["max_drawdown"], 6),
            "multi_max_drawdown": round(multi_full["max_drawdown"], 6),
            "mean_diff_pp": paired["mean_diff_pp"],
            "wins": paired["wins"],
            "losses": paired["losses"],
            "sign_test_p": paired["sign_test_p"],
        }
        sections.append(
            "\n".join(
                [
                    f"# 全窗口 {len(all_months)} 个月（顺序复利）",
                    "",
                    "| 架构 | 总收益 | 指数 | 超额 | 最大回撤 | 最差月 | 月胜率 | 费用 |",
                    "|---|---:|---:|---:|---:|---:|---:|---:|",
                    (
                        f"| 单智能体 | {single_full['total_return']*100:+.2f}% "
                        f"| {single_full['index_return']*100:+.2f}% "
                        f"| {single_full['excess_return']*100:+.2f}% "
                        f"| {single_full['max_drawdown']*100:.2f}% "
                        f"| {single_full['worst_month']*100:+.2f}% "
                        f"| {single_full['win_rate']*100:.0f}% "
                        f"| {single_full['fees']:.0f} |"
                    ),
                    (
                        f"| 多智能体 | {multi_full['total_return']*100:+.2f}% "
                        f"| {multi_full['index_return']*100:+.2f}% "
                        f"| {multi_full['excess_return']*100:+.2f}% "
                        f"| {multi_full['max_drawdown']*100:.2f}% "
                        f"| {multi_full['worst_month']*100:+.2f}% "
                        f"| {multi_full['win_rate']*100:.0f}% "
                        f"| {multi_full['fees']:.0f} |"
                    ),
                    "",
                    f"配对差（pp）: {paired['diffs_pp']}",
                    (
                        f"mean={paired['mean_diff_pp']:+.2f}pp "
                        f"wins={paired['wins']} losses={paired['losses']} "
                        f"sign_p={paired['sign_test_p']}"
                    ),
                    "",
                ]
            )
        )

    text = "\n".join(sections)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        out_path.with_suffix(".json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"report -> {out_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
