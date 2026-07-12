#!/usr/bin/env python3
"""Backtest unified A-share trader allocations with costs and cash."""

from __future__ import annotations

import argparse
import glob
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils.ashare_trader import (  # noqa: E402
    DEFAULT_CLOSE_COST,
    DEFAULT_MIN_COST,
    DEFAULT_OPEN_COST,
    PortfolioParseResult,
    cash_allocation,
    evaluate_portfolio_month,
    parse_portfolio_weights,
)
from src.utils.ashare_trader_features import (  # noqa: E402
    compute_trader_feature_rows,
)

DEFAULT_TASKS = ROOT / "data" / "ashare_trader" / "standardized_data.jsonl"
DATA_DIR = ROOT / "data" / "ashare"


def load_tasks(path: str | Path) -> dict[str, dict[str, Any]]:
    tasks = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {str(task["task_id"]): task for task in tasks}


def load_run(path: str | Path) -> dict[str, str]:
    answers: dict[str, str] = {}
    for filename in glob.glob(str(Path(path) / "task_*_attempt_1.json")):
        data = json.loads(Path(filename).read_text(encoding="utf-8"))
        if data.get("status") != "completed":
            continue
        task_id = str(data.get("task_id") or data.get("task_name") or "")
        if task_id:
            answers[task_id] = str(
                data.get("final_boxed_answer")
                or data.get("model_boxed_answer")
                or ""
            )
    return answers


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def _sample_std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else float("nan")


def _annualized_ratio(values: list[float]) -> float:
    std = _sample_std(values)
    return (
        _mean(values) / std * math.sqrt(12)
        if math.isfinite(std) and std > 0
        else float("nan")
    )


def _max_drawdown(capitals: list[float]) -> float:
    if not capitals:
        return float("nan")
    peak = capitals[0]
    worst = 0.0
    for capital in capitals:
        peak = max(peak, capital)
        if peak > 0:
            worst = min(worst, capital / peak - 1.0)
    return worst


def _finite_mean(values: list[float]) -> float:
    clean = [value for value in values if math.isfinite(value)]
    return _mean(clean)


def parse_run_allocations(
    tasks: Mapping[str, dict[str, Any]],
    answers: Mapping[str, str],
) -> dict[str, PortfolioParseResult]:
    parsed: dict[str, PortfolioParseResult] = {}
    for task_id, task in tasks.items():
        metadata = task["metadata"]
        if task_id not in answers:
            pool = metadata["stock_pool"]
            parsed[task_id] = PortfolioParseResult(
                {code: 0.0 for code in pool},
                1.0,
                False,
                "missing completed answer",
            )
            continue
        parsed[task_id] = parse_portfolio_weights(
            answers[task_id],
            metadata["stock_pool"],
            max_stock_weight=float(metadata.get("max_stock_weight", 0.25)),
        )
    return parsed


def evaluate_allocations(
    name: str,
    tasks: Mapping[str, dict[str, Any]],
    allocations: Mapping[str, PortfolioParseResult],
    *,
    initial_capital: float,
    open_cost: float,
    close_cost: float,
    min_cost: float,
    is_oracle: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    capital = float(initial_capital)
    benchmark_nav = 1.0
    capitals = [capital]
    monthly: list[dict[str, Any]] = []

    ordered_tasks = sorted(
        tasks.items(), key=lambda item: str(item[1]["metadata"]["entry_date"])
    )
    for task_id, task in ordered_tasks:
        metadata = task["metadata"]
        requested = allocations.get(task_id)
        parse_ok = bool(requested and requested.ok)
        allocation = (
            requested
            if requested is not None and requested.ok
            else cash_allocation(metadata["stock_pool"])
        )
        result = evaluate_portfolio_month(
            allocation.weights,
            allocation.cash,
            metadata["stock_returns"],
            float(metadata["index_return"]),
            starting_capital=capital,
            excess_returns=metadata.get("excess_returns"),
            open_cost=open_cost,
            close_cost=close_cost,
            min_cost=min_cost,
        )
        capital = result.ending_capital
        capitals.append(capital)
        benchmark_nav *= 1.0 + float(metadata["index_return"])
        ranked_contributions = sorted(
            result.contributions.items(),
            key=lambda item: (-item[1], item[0]),
        )
        monthly.append(
            {
                "task_id": task_id,
                "entry_date": metadata["entry_date"],
                "exit_date": metadata["exit_date"],
                "parse_ok": parse_ok,
                "parse_error": (
                    ""
                    if parse_ok
                    else (requested.error if requested is not None else "missing")
                ),
                "weights": {
                    code: weight
                    for code, weight in allocation.weights.items()
                    if weight > 0
                },
                "cash": allocation.cash,
                "starting_capital": result.starting_capital,
                "ending_capital": result.ending_capital,
                "gross_return": result.gross_return,
                "net_return": result.net_return,
                "index_return": result.index_return,
                "active_return": result.active_return,
                "buy_cost": result.buy_cost,
                "sell_cost": result.sell_cost,
                "total_cost": result.total_cost,
                "gross_traded_notional": result.gross_traded_notional,
                "invested_weight": result.invested_weight,
                "holding_count": result.holding_count,
                "concentration_hhi": result.concentration_hhi,
                "weight_rank_ic": result.weight_rank_ic,
                "top_contributor": (
                    ranked_contributions[0][0] if ranked_contributions else ""
                ),
                "top_contribution": (
                    ranked_contributions[0][1] if ranked_contributions else 0.0
                ),
                "worst_contributor": (
                    ranked_contributions[-1][0] if ranked_contributions else ""
                ),
                "worst_contribution": (
                    ranked_contributions[-1][1] if ranked_contributions else 0.0
                ),
            }
        )

    net_returns = [float(row["net_return"]) for row in monthly]
    active_returns = [float(row["active_return"]) for row in monthly]
    benchmark_final = initial_capital * benchmark_nav
    summary = {
        "run": name,
        "oracle": bool(is_oracle),
        "months": len(monthly),
        "parsed": sum(bool(row["parse_ok"]) for row in monthly),
        "parse_rate": (
            sum(bool(row["parse_ok"]) for row in monthly) / len(monthly)
            if monthly
            else 0.0
        ),
        "initial_capital": initial_capital,
        "final_capital": capital,
        "net_return": capital / initial_capital - 1.0,
        "benchmark_final_capital": benchmark_final,
        "benchmark_return": benchmark_nav - 1.0,
        "relative_nav_return": (
            capital / benchmark_final - 1.0 if benchmark_final > 0 else float("nan")
        ),
        "annualized_sharpe": _annualized_ratio(net_returns),
        "information_ratio": _annualized_ratio(active_returns),
        "max_drawdown": _max_drawdown(capitals),
        "beat_month_rate": (
            sum(value > 0 for value in active_returns) / len(active_returns)
            if active_returns
            else float("nan")
        ),
        "total_cost": sum(float(row["total_cost"]) for row in monthly),
        "total_traded_notional": sum(
            float(row["gross_traded_notional"]) for row in monthly
        ),
        "average_cash": _mean([float(row["cash"]) for row in monthly]),
        "average_holding_count": _mean(
            [float(row["holding_count"]) for row in monthly]
        ),
        "average_concentration_hhi": _mean(
            [float(row["concentration_hhi"]) for row in monthly]
        ),
        "mean_weight_rank_ic": _finite_mean(
            [float(row["weight_rank_ic"]) for row in monthly]
        ),
    }
    return summary, monthly


def _fixed_allocation(
    pool: list[str],
    selected: list[str],
) -> PortfolioParseResult:
    if not selected:
        return cash_allocation(pool)
    weight = min(0.25, 1.0 / len(selected))
    weights = {code: (weight if code in selected else 0.0) for code in pool}
    cash = 1.0 - sum(weights.values())
    return PortfolioParseResult(weights, cash, True)


def build_baseline_allocations(
    tasks: Mapping[str, dict[str, Any]],
) -> list[tuple[str, dict[str, PortfolioParseResult], bool]]:
    qlib = pd.read_csv(DATA_DIR / "qlib_signal.csv", dtype={"entry_date": str})
    baselines: dict[str, dict[str, PortfolioParseResult]] = {
        "cash(全现金)": {},
        "equal_weight(16股等权)": {},
        "momentum_top4(20日相对动量)": {},
        "qlib_top4(逐月walk-forward)": {},
        "oracle_top4(事后上界,不可交易)": {},
    }

    for task_id, task in tasks.items():
        metadata = task["metadata"]
        pool = list(metadata["stock_pool"])
        baselines["cash(全现金)"][task_id] = cash_allocation(pool)
        baselines["equal_weight(16股等权)"][task_id] = _fixed_allocation(
            pool, pool
        )

        features = compute_trader_feature_rows(
            metadata["entry_date"],
            metadata["stock_info"],
            data_dir=DATA_DIR,
            lookback_days=250,
        )
        momentum_order = [
            row["ts_code"]
            for row in sorted(
                features,
                key=lambda row: (
                    -float(row["rel20"])
                    if row.get("rel20") not in ("", None)
                    else float("inf"),
                    row["ts_code"],
                ),
            )
        ]
        baselines["momentum_top4(20日相对动量)"][task_id] = _fixed_allocation(
            pool, momentum_order[:4]
        )

        # Some trader entries are shifted to the previous portfolio's
        # liquidation close to prevent capital overlap.  Use the latest
        # walk-forward signal already available by that adjusted entry date.
        available_signal = qlib[
            (qlib["entry_date"] <= str(metadata["entry_date"]))
            & qlib["ts_code"].isin(pool)
        ].sort_values(["entry_date", "ts_code"])
        latest_signal = available_signal.groupby("ts_code", as_index=False).tail(1)
        qlib_order = latest_signal.sort_values(
            ["rank", "ts_code"]
        )["ts_code"].tolist()
        baselines["qlib_top4(逐月walk-forward)"][task_id] = _fixed_allocation(
            pool, qlib_order[:4]
        )
        baselines["oracle_top4(事后上界,不可交易)"][task_id] = _fixed_allocation(
            pool, list(metadata["ground_truth_rank"])[:4]
        )

    return [
        (name, allocation, name.startswith("oracle_"))
        for name, allocation in baselines.items()
    ]


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _pct(value: float) -> str:
    return "-" if not math.isfinite(value) else f"{value * 100:.2f}%"


def _number(value: float, digits: int = 2) -> str:
    return "-" if not math.isfinite(value) else f"{value:.{digits}f}"


def render_report(
    evaluated: list[tuple[dict[str, Any], list[dict[str, Any]]]],
    *,
    initial_capital: float,
    open_cost: float,
    close_cost: float,
    min_cost: float,
) -> str:
    lines = [
        "# A股统一交易员组合回测",
        "",
        (
            "口径：每月计划调仓；若与前一20日窗口重叠则顺延到前次平仓收盘，"
            "持有20个交易日后全部卖出；"
            f"初始资金 ¥{initial_capital:,.2f}，买入费 {open_cost:.3%}，"
            f"卖出费 {close_cost:.3%}，每笔最低 ¥{min_cost:.2f}，现金收益 0。"
        ),
        "无效或缺失仓位按全现金执行；oracle 仅是事后上界，不是可交易基线。",
        "",
        "| 运行 | 合法月 | 最终资产 | 净收益 | 沪深300 | 相对净值 | Sharpe | 信息比 | 最大回撤 | 跑赢月 | 总费用 | 平均现金 | 平均持仓 | 权重RankIC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary, _ in evaluated:
        label = summary["run"] + (" [oracle]" if summary["oracle"] else "")
        lines.append(
            f"| {label} | {summary['parsed']}/{summary['months']} | "
            f"¥{summary['final_capital']:,.2f} | {_pct(summary['net_return'])} | "
            f"{_pct(summary['benchmark_return'])} | "
            f"{_pct(summary['relative_nav_return'])} | "
            f"{_number(summary['annualized_sharpe'])} | "
            f"{_number(summary['information_ratio'])} | "
            f"{_pct(summary['max_drawdown'])} | "
            f"{_pct(summary['beat_month_rate'])} | "
            f"¥{summary['total_cost']:,.2f} | {_pct(summary['average_cash'])} | "
            f"{summary['average_holding_count']:.1f} | "
            f"{_number(summary['mean_weight_rank_ic'], 3)} |"
        )

    for summary, monthly in evaluated:
        lines.extend(
            [
                "",
                f"## {summary['run']} 月度明细",
                "",
                "| 买入日 | 卖出日 | 格式 | 股票仓位 | 现金 | 净收益 | 沪深300 | 主动收益 | 费用 | 期末资产 | 最大贡献 | 最大拖累 |",
                "|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|",
            ]
        )
        for row in monthly:
            positions = ",".join(
                f"{code}:{weight:.3f}"
                for code, weight in row["weights"].items()
            ) or "-"
            status = "OK" if row["parse_ok"] else "现金回退"
            lines.append(
                f"| {row['entry_date']} | {row['exit_date']} | {status} | "
                f"{positions} | {_pct(float(row['cash']))} | "
                f"{_pct(float(row['net_return']))} | "
                f"{_pct(float(row['index_return']))} | "
                f"{_pct(float(row['active_return']))} | "
                f"¥{float(row['total_cost']):,.2f} | "
                f"¥{float(row['ending_capital']):,.2f} | "
                f"{row['top_contributor']} | {row['worst_contributor']} |"
            )
            if not row["parse_ok"]:
                lines.append(f"\n格式错误：`{row['parse_error']}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", default=[], help="name=logs/path")
    parser.add_argument("--tasks", default=str(DEFAULT_TASKS))
    parser.add_argument("--out", default="logs/ashare_trader_report.md")
    parser.add_argument("--json-out", default="")
    parser.add_argument("--initial-capital", type=float, default=1_000_000.0)
    parser.add_argument("--open-cost", type=float, default=DEFAULT_OPEN_COST)
    parser.add_argument("--close-cost", type=float, default=DEFAULT_CLOSE_COST)
    parser.add_argument("--min-cost", type=float, default=DEFAULT_MIN_COST)
    args = parser.parse_args()

    tasks = load_tasks(args.tasks)
    evaluated: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for specification in args.run:
        name, separator, run_path = specification.partition("=")
        if not separator or not name or not run_path:
            raise ValueError(f"invalid --run value: {specification!r}")
        answers = load_run(run_path)
        allocations = parse_run_allocations(tasks, answers)
        evaluated.append(
            evaluate_allocations(
                name,
                tasks,
                allocations,
                initial_capital=args.initial_capital,
                open_cost=args.open_cost,
                close_cost=args.close_cost,
                min_cost=args.min_cost,
            )
        )

    for name, allocations, is_oracle in build_baseline_allocations(tasks):
        evaluated.append(
            evaluate_allocations(
                name,
                tasks,
                allocations,
                initial_capital=args.initial_capital,
                open_cost=args.open_cost,
                close_cost=args.close_cost,
                min_cost=args.min_cost,
                is_oracle=is_oracle,
            )
        )

    report = render_report(
        evaluated,
        initial_capital=args.initial_capital,
        open_cost=args.open_cost,
        close_cost=args.close_cost,
        min_cost=args.min_cost,
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    json_output = (
        Path(args.json_out)
        if args.json_out
        else output.with_suffix(".json")
    )
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(
        json.dumps(
            _json_safe(
                [
                    {"summary": summary, "monthly": monthly}
                    for summary, monthly in evaluated
                ]
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote trader report -> {output}")
    print(f"wrote trader JSON -> {json_output}")


if __name__ == "__main__":
    main()
