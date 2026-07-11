"""Walk-forward factor-reliability memory for cross-sectional ranking tasks."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Any, Mapping, Sequence

from src.memory.memory import Mem0Memory
from src.memory.rolling_reflection import normalize_date, normalize_month

_FACTORS = {
    "rel5": ("近5日相对动量", 1.0),
    "rel20": ("近20日相对动量", 1.0),
    "rel60": ("近60日相对动量", 1.0),
    "pe_pct": ("PE分位", 1.0),
    "pb_pct": ("PB分位", 1.0),
    "turn_pct": ("换手率分位", 1.0),
    # Lower Qlib rank is better, so negate it before computing IC.
    "ml_rank": ("Qlib预测排序", -1.0),
}


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average = (start + 1 + end) / 2
        for position in range(start, end):
            ranks[order[position]] = average
        start = end
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum(
        (x - left_mean) * (y - right_mean) for x, y in zip(left, right)
    )
    left_var = sum((x - left_mean) ** 2 for x in left)
    right_var = sum((y - right_mean) ** 2 for y in right)
    denominator = math.sqrt(left_var * right_var)
    return numerator / denominator if denominator > 0 else None


def spearman(values: Sequence[float], outcomes: Sequence[float]) -> float | None:
    return _pearson(_average_ranks(values), _average_ranks(outcomes))


def _sign_flip_pvalue(values: Sequence[float]) -> float:
    """Exact two-sided randomization p-value for a non-zero mean RankIC."""
    n = len(values)
    if not n:
        return 1.0
    observed = abs(sum(values))
    if n > 16:
        # Normal approximation to the sign-flip distribution prevents
        # exponential work on multi-year benchmarks.
        scale = math.sqrt(sum(value * value for value in values))
        return math.erfc(observed / scale / math.sqrt(2)) if scale > 0 else 1.0
    extreme = 0
    for mask in range(1 << n):
        randomized_sum = sum(
            value if mask & (1 << index) else -value
            for index, value in enumerate(values)
        )
        if abs(randomized_sum) + 1e-12 >= observed:
            extreme += 1
    return extreme / (1 << n)


def _bh_qvalues(pvalues: Sequence[float]) -> list[float]:
    """Benjamini-Hochberg adjusted q-values in original factor order."""
    if not pvalues:
        return []
    order = sorted(range(len(pvalues)), key=lambda index: pvalues[index])
    adjusted = [1.0] * len(pvalues)
    running = 1.0
    total = len(pvalues)
    for rank in range(total, 0, -1):
        index = order[rank - 1]
        running = min(running, pvalues[index] * total / rank)
        adjusted[index] = min(1.0, running)
    return adjusted


def eligible_rank_samples(
    samples: Sequence[Mapping[str, Any]],
    as_of_date: str,
) -> list[dict[str, Any]]:
    cutoff = normalize_date(as_of_date)
    if not cutoff:
        return []
    deduped: dict[str, dict[str, Any]] = {}
    for raw in samples:
        task_id = str(raw.get("task_id", ""))
        exit_date = normalize_date(raw.get("exit_date"))
        entry_date = normalize_date(raw.get("entry_date"))
        excess = _number(raw.get("excess_return"))
        if (
            not task_id
            or not exit_date
            or exit_date > cutoff
            or not entry_date
            or excess is None
        ):
            continue
        row = dict(raw)
        row["entry_date"] = entry_date
        row["entry_month"] = normalize_month(raw.get("entry_month") or entry_date)
        row["exit_date"] = exit_date
        row["excess_return"] = excess
        deduped[task_id] = row
    return sorted(deduped.values(), key=lambda row: (row["entry_date"], row["task_id"]))


def factor_reliability(
    samples: Sequence[Mapping[str, Any]],
    as_of_date: str,
    *,
    min_stocks_per_month: int = 8,
) -> list[dict[str, Any]]:
    """Compute one cross-sectional RankIC per matured month and factor."""
    eligible = eligible_rank_samples(samples, as_of_date)
    by_month: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        by_month[row["entry_month"]].append(row)

    output: list[dict[str, Any]] = []
    for feature, (label, orientation) in _FACTORS.items():
        monthly_ics: list[float] = []
        used_months: list[str] = []
        for month, rows in sorted(by_month.items()):
            pairs = [
                (_number(row.get(feature)), _number(row.get("excess_return")))
                for row in rows
            ]
            clean = [
                (feature_value * orientation, excess)
                for feature_value, excess in pairs
                if feature_value is not None and excess is not None
            ]
            if len(clean) < min_stocks_per_month:
                continue
            ic = spearman(
                [pair[0] for pair in clean],
                [pair[1] for pair in clean],
            )
            if ic is None:
                continue
            monthly_ics.append(ic)
            used_months.append(month)
        if not monthly_ics:
            continue
        mean_ic = statistics.fmean(monthly_ics)
        std = statistics.stdev(monthly_ics) if len(monthly_ics) > 1 else float("nan")
        t_stat = (
            mean_ic / (std / math.sqrt(len(monthly_ics)))
            if len(monthly_ics) > 1 and std > 0
            else float("nan")
        )
        same_sign = sum(
            (ic >= 0) == (mean_ic >= 0) for ic in monthly_ics
        ) / len(monthly_ics)
        output.append(
            {
                "feature": feature,
                "label": label,
                "months": used_months,
                "n_months": len(monthly_ics),
                "mean_ic": mean_ic,
                "ic_std": std,
                "ic_t": t_stat,
                "sign_consistency": same_sign,
                "monthly_ics": monthly_ics,
                "p_value": _sign_flip_pvalue(monthly_ics),
            }
        )
    qvalues = _bh_qvalues([row["p_value"] for row in output])
    for row, qvalue in zip(output, qvalues):
        row["q_value"] = qvalue
    output.sort(key=lambda row: (-abs(row["mean_ic"]), row["feature"]))
    return output


def build_rank_factor_block(
    memory: Mem0Memory,
    as_of_date: str,
    *,
    min_months: int = 3,
    max_factors: int = 5,
    fdr_q: float = 0.10,
    show_status_when_empty: bool = False,
) -> str:
    """Format FDR-controlled historical factor reliability for the next month."""
    eligible = [
        row
        for row in factor_reliability(memory.load_samples(), as_of_date)
        if row["n_months"] >= min_months
    ]
    stats = [row for row in eligible if row["q_value"] <= fdr_q][:max_factors]
    if not stats:
        if show_status_when_empty and eligible:
            matured_months = max(row["n_months"] for row in eligible)
            return "\n".join(
                [
                    "### 历史因子记忆状态（严格 walk-forward）",
                    (
                        f"已有 {matured_months} 个已到期月份，检验了 {len(eligible)} 个"
                        f"候选因子；没有因子通过双侧符号翻转检验与 FDR q≤{fdr_q:.2f}。"
                    ),
                    (
                        "- 这表示历史样本暂不支持稳定的因子方向。不得根据未验证的均值符号"
                        "反转因子，也不得把单一 Qlib、动量或估值信号当作历史上可靠；"
                        "请仅结合当前截面处理冲突并降低排序置信度。"
                    ),
                ]
            )
        return ""
    lines = [
        (
            "### 历史因子可靠性（严格 walk-forward）\n"
            f"以下 RankIC 仅使用截至 {normalize_date(as_of_date)} 已到期月份，"
            "正值表示按该因子的常规方向排序与未来超额收益同向。"
        )
    ]
    for row in stats:
        lines.append(
            f"- {row['label']}：月均 RankIC={row['mean_ic']:+.3f}，"
            f"同号月份={row['sign_consistency']:.0%}，n={row['n_months']}，"
            f"FDR q={row['q_value']:.3f}。"
        )
    lines.append(
        "- 这些统计只用于调整证据权重，不是个股方向标签；样本月份较少或符号不稳定时，"
        "不得机械反转当前排序。"
    )
    return "\n".join(lines)
