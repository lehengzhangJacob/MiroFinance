#!/usr/bin/env python3
"""Evaluate monthly A-share ranking runs with deterministic financial metrics."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.memory.monthly_reflection import compute_month_feature_rows  # noqa: E402
from src.utils.ashare_rank import evaluate_ranking, parse_ranked_codes  # noqa: E402

DEFAULT_TASKS = ROOT / "data" / "ashare_rank" / "standardized_data.jsonl"
DATA_DIR = ROOT / "data" / "ashare"


def load_tasks(path: str | Path) -> dict[str, dict[str, Any]]:
    return {
        row["task_id"]: row
        for row in (
            json.loads(line)
            for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def load_run(path: str | Path) -> dict[str, str]:
    answers: dict[str, str] = {}
    for filename in glob.glob(str(Path(path) / "task_*_attempt_1.json")):
        data = json.loads(Path(filename).read_text(encoding="utf-8"))
        if data.get("status") == "completed":
            answers[str(data.get("task_id") or data.get("task_name"))] = str(
                data.get("final_boxed_answer") or ""
            )
    return answers


def count_rank_factor_activations(path: str | Path) -> tuple[int, int]:
    """Count tasks receiving validated rules versus an abstention status."""
    validated_marker = "### 历史因子可靠性（严格 walk-forward）"
    status_marker = "### 历史因子记忆状态（严格 walk-forward）"
    validated = 0
    status = 0
    for filename in glob.glob(str(Path(path) / "task_*_attempt_1.json")):
        data = json.loads(Path(filename).read_text(encoding="utf-8"))
        history = data.get("main_agent_message_history", {})
        serialized = json.dumps(history, ensure_ascii=False)
        validated += validated_marker in serialized
        status += status_marker in serialized
    return validated, status


def load_prompt_fingerprints(path: str | Path) -> dict[str, str]:
    """Hash effective initial inputs, excluding random message identifiers."""
    fingerprints: dict[str, str] = {}
    for filename in glob.glob(str(Path(path) / "task_*_attempt_1.json")):
        data = json.loads(Path(filename).read_text(encoding="utf-8"))
        history = data.get("main_agent_message_history", {})
        system_prompt = str(history.get("system_prompt", ""))
        initial_user_text = ""
        for message in history.get("message_history", []):
            if message.get("role") != "user":
                continue
            content = message.get("content", "")
            if isinstance(content, list) and content:
                initial_user_text = str(content[0].get("text", ""))
            else:
                initial_user_text = str(content)
            break
        initial_user_text = re.sub(
            r"^\[msg_[^\]]+\]\s*", "", initial_user_text
        )
        task_id = str(data.get("task_id") or data.get("task_name") or "")
        if task_id:
            payload = f"{system_prompt}\n---\n{initial_user_text}".encode("utf-8")
            fingerprints[task_id] = hashlib.sha256(payload).hexdigest()
    return fingerprints


def _qlib_order(task: dict[str, Any], signal: pd.DataFrame) -> list[str]:
    date = str(task["metadata"]["entry_date"])
    pool = set(task["metadata"]["stock_pool"])
    month = signal[
        (signal["entry_date"].astype(str) == date) & signal["ts_code"].isin(pool)
    ].sort_values(["rank", "ts_code"])
    return month["ts_code"].tolist()


def _momentum_order(task: dict[str, Any]) -> list[str]:
    metadata = task["metadata"]
    rows = compute_month_feature_rows(
        metadata["entry_date"],
        [
            {
                "ts_code": code,
                "stock_name": "",
                "label": "",
                "predicted": "",
                "judge_result": "",
            }
            for code in metadata["stock_pool"]
        ],
        data_dir=DATA_DIR,
    )
    return [
        row["ts_code"]
        for row in sorted(
            rows,
            key=lambda row: (
                -float(row["rel20"]) if row.get("rel20") != "" else float("inf"),
                row["ts_code"],
            ),
        )
    ]


def evaluate_orders(
    name: str,
    tasks: dict[str, dict[str, Any]],
    orders: dict[str, list[str] | None],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    monthly: list[dict[str, Any]] = []
    for task_id, task in sorted(tasks.items()):
        metadata = task["metadata"]
        order = orders.get(task_id)
        row: dict[str, Any] = {
            "task_id": task_id,
            "entry_date": metadata["entry_date"],
            "parse_ok": bool(order),
        }
        if order:
            row.update(
                evaluate_ranking(
                    order,
                    metadata["ground_truth_rank"],
                    metadata["excess_returns"],
                    top_k=4,
                )
            )
        monthly.append(row)

    valid = [row for row in monthly if row["parse_ok"]]

    def values(key: str) -> list[float]:
        return [float(row[key]) for row in valid]

    def mean(key: str) -> float:
        vals = values(key)
        return statistics.fmean(vals) if vals else float("nan")

    def std(key: str) -> float:
        vals = values(key)
        return statistics.stdev(vals) if len(vals) > 1 else float("nan")

    def cumulative(key: str) -> float:
        result = 1.0
        for value in values(key):
            result *= 1.0 + value
        return result - 1.0 if valid else float("nan")

    ic_std = std("rank_ic")
    top_std = std("top_excess")
    spread_std = std("spread")
    summary = {
        "run": name,
        "tasks": len(monthly),
        "parsed": len(valid),
        "parse_rate": len(valid) / len(monthly) if monthly else 0.0,
        "mean_rank_ic": mean("rank_ic"),
        "rank_ic_ir": (
            mean("rank_ic") / ic_std if ic_std and not math.isnan(ic_std) else float("nan")
        ),
        "mean_top4_excess": mean("top_excess"),
        "mean_bottom4_excess": mean("bottom_excess"),
        "mean_spread": mean("spread"),
        "cum_top4_excess": cumulative("top_excess"),
        "cum_spread": cumulative("spread"),
        "top4_sharpe": (
            mean("top_excess") / top_std * math.sqrt(12)
            if top_std and not math.isnan(top_std)
            else float("nan")
        ),
        "spread_sharpe": (
            mean("spread") / spread_std * math.sqrt(12)
            if spread_std and not math.isnan(spread_std)
            else float("nan")
        ),
    }
    return summary, monthly


def _fmt(value: float, percent: bool = False) -> str:
    if value is None or math.isnan(float(value)):
        return "-"
    return f"{value * 100:.2f}" if percent else f"{value:.3f}"


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _sign_flip_pvalue(differences: list[float]) -> float:
    """Exact paired two-sided randomization test for a zero mean difference."""
    values = [value for value in differences if math.isfinite(value)]
    if not values:
        return float("nan")
    observed = abs(sum(values))
    if observed == 0:
        return 1.0
    if len(values) > 16:
        scale = math.sqrt(sum(value * value for value in values))
        return math.erfc(observed / scale / math.sqrt(2)) if scale > 0 else 1.0
    extreme = 0
    for mask in range(1 << len(values)):
        randomized_sum = sum(
            value if mask & (1 << index) else -value
            for index, value in enumerate(values)
        )
        if abs(randomized_sum) + 1e-12 >= observed:
            extreme += 1
    return extreme / (1 << len(values))


def paired_comparisons(
    runs: list[tuple[dict[str, Any], list[dict[str, Any]]]],
) -> list[dict[str, Any]]:
    """Compare each later user-supplied arm against each earlier arm by month."""
    comparisons: list[dict[str, Any]] = []
    for control_index in range(len(runs)):
        control_summary, control_rows = runs[control_index]
        control_by_task = {row["task_id"]: row for row in control_rows}
        for treatment_index in range(control_index + 1, len(runs)):
            treatment_summary, treatment_rows = runs[treatment_index]
            treatment_by_task = {row["task_id"]: row for row in treatment_rows}
            paired = [
                (control_by_task[task_id], treatment_by_task[task_id])
                for task_id in sorted(control_by_task.keys() & treatment_by_task.keys())
                if control_by_task[task_id].get("parse_ok")
                and treatment_by_task[task_id].get("parse_ok")
            ]

            def differences(key: str) -> list[float]:
                return [
                    float(treatment[key]) - float(control[key])
                    for control, treatment in paired
                    if key in control and key in treatment
                ]

            rank_deltas = differences("rank_ic")
            top_deltas = differences("top_excess")
            spread_deltas = differences("spread")
            prompt_pairs = [
                (control.get("prompt_fingerprint"), treatment.get("prompt_fingerprint"))
                for control, treatment in paired
                if control.get("prompt_fingerprint")
                and treatment.get("prompt_fingerprint")
            ]
            comparisons.append(
                {
                    "comparison": (
                        f"{treatment_summary['run']} - {control_summary['run']}"
                    ),
                    "months": len(rank_deltas),
                    "delta_mean_rank_ic": (
                        statistics.fmean(rank_deltas)
                        if rank_deltas
                        else float("nan")
                    ),
                    "rank_ic_sign_flip_p": _sign_flip_pvalue(rank_deltas),
                    "rank_ic_wins": sum(delta > 0 for delta in rank_deltas),
                    "rank_ic_ties": sum(delta == 0 for delta in rank_deltas),
                    "prompt_changed_months": sum(
                        control != treatment for control, treatment in prompt_pairs
                    ),
                    "delta_mean_top4_excess": (
                        statistics.fmean(top_deltas)
                        if top_deltas
                        else float("nan")
                    ),
                    "delta_mean_spread": (
                        statistics.fmean(spread_deltas)
                        if spread_deltas
                        else float("nan")
                    ),
                }
            )
    return comparisons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default=str(DEFAULT_TASKS))
    parser.add_argument("--run", action="append", default=[], help="name=run/directory")
    parser.add_argument("--out", default="logs/ashare_rank_report.md")
    args = parser.parse_args()

    tasks = load_tasks(args.tasks)
    all_results: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for spec in args.run:
        name, separator, path = spec.partition("=")
        if not separator:
            raise ValueError(f"--run must be name=path, got {spec!r}")
        raw_answers = load_run(path)
        orders: dict[str, list[str] | None] = {}
        for task_id, task in tasks.items():
            parsed = parse_ranked_codes(
                raw_answers.get(task_id, ""),
                task["metadata"]["stock_pool"],
            )
            orders[task_id] = parsed.codes if parsed.ok else None
        result = evaluate_orders(name, tasks, orders)
        validated, status = count_rank_factor_activations(path)
        result[0]["rank_factor_validated"] = validated
        result[0]["rank_factor_status"] = status
        fingerprints = load_prompt_fingerprints(path)
        for row in result[1]:
            row["prompt_fingerprint"] = fingerprints.get(row["task_id"], "")
        all_results.append(result)
    user_results = list(all_results)
    pairwise = paired_comparisons(user_results)

    signal = pd.read_csv(DATA_DIR / "qlib_signal.csv", dtype={"entry_date": str})
    qlib_orders = {task_id: _qlib_order(task, signal) for task_id, task in tasks.items()}
    momentum_orders = {
        task_id: _momentum_order(task) for task_id, task in tasks.items()
    }
    all_results.append(evaluate_orders("qlib-rank", tasks, qlib_orders))
    all_results.append(evaluate_orders("rel20-momentum", tasks, momentum_orders))

    lines = [
        "# A股月度横截面排序评测",
        "",
        f"任务：{len(tasks)} 个月 × 16 股；预测未来20交易日相对沪深300超额收益排序。",
        "",
        "| 运行 | 可解析 | 因子记忆(规则/状态) | 平均RankIC | RankIC IR | Top-4月均超额% | Bottom-4月均超额% | 月均多空差% | Top-4累计超额% | Top-4夏普 | 多空夏普 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    summaries = []
    detail: dict[str, Any] = {}
    for summary, monthly in all_results:
        summaries.append(summary)
        detail[summary["run"]] = monthly
        validated = summary.get("rank_factor_validated")
        status = summary.get("rank_factor_status")
        activation = (
            f"{validated}/{status}" if validated is not None and status is not None else "-"
        )
        lines.append(
            f"| {summary['run']} | {summary['parsed']}/{summary['tasks']} "
            f"| {activation} "
            f"| {_fmt(summary['mean_rank_ic'])} | {_fmt(summary['rank_ic_ir'])} "
            f"| {_fmt(summary['mean_top4_excess'], True)} "
            f"| {_fmt(summary['mean_bottom4_excess'], True)} "
            f"| {_fmt(summary['mean_spread'], True)} "
            f"| {_fmt(summary['cum_top4_excess'], True)} "
            f"| {_fmt(summary['top4_sharpe'])} | {_fmt(summary['spread_sharpe'])} |"
        )

    lines += [
        "",
        "说明：RankIC 为预测全序与真实超额收益全序的 Spearman 相关；"
        "Top-4/Bottom-4 均为等权组合；多空差=Top-4−Bottom-4。",
        "“因子记忆(规则/状态)”分别统计收到 FDR 验证规则和“无规则通过”校准状态的"
        "任务数；两者均为 0 时，该组与对照组的差异不能归因于因子记忆。",
    ]
    if pairwise:
        lines += [
            "",
            "## 配对比较",
            "",
            "| 处理组−对照组 | 配对月数 | 输入变化月数 | Δ平均RankIC | 符号翻转p值 | RankIC胜/月 | ΔTop-4月均超额% | Δ月均多空差% |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in pairwise:
            lines.append(
                f"| {row['comparison']} | {row['months']} "
                f"| {row['prompt_changed_months']} "
                f"| {_fmt(row['delta_mean_rank_ic'])} "
                f"| {_fmt(row['rank_ic_sign_flip_p'])} "
                f"| {row['rank_ic_wins']}/{row['months']} "
                f"| {_fmt(row['delta_mean_top4_excess'], True)} "
                f"| {_fmt(row['delta_mean_spread'], True)} |"
            )
        lines += [
            "",
            "配对 p 值来自逐月差值的双侧精确符号翻转检验；"
            "12 个月样本下应同时报告效应量，不以单一 p 值下结论。",
            "“输入变化月数”为去除随机消息ID后，两组系统提示与初始任务提示实际不同的月份数；"
            "若为 0，组间输出差异来自模型非确定性，不能解释为处理效应。",
        ]
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    output.with_suffix(".json").write_text(
        json.dumps(
            _json_safe(
                {"summaries": summaries, "monthly": detail, "pairwise": pairwise}
            ),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    print(f"report -> {output}")
    for line in lines[4 : 7 + len(all_results)]:
        print(line)


if __name__ == "__main__":
    main()
