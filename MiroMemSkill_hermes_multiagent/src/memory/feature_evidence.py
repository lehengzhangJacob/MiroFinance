"""Point-in-time, feature-conditioned evidence for A-share prediction tasks.

This module deliberately separates two evidence types:

* rolling rules that already passed the existing temporal/FDR gates; and
* descriptive nearest neighbours from matured samples.

Nearest neighbours are only surfaced as directional evidence when a Wilson
interval excludes 50%.  Otherwise no label counts are shown, which avoids
turning a noisy local majority into another prompt direction anchor.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

from src.memory.memory import Mem0Memory
from src.memory.monthly_reflection import DEFAULT_DATA_DIR, compute_month_feature_rows
from src.memory.rolling_reflection import (
    OUTPERFORM,
    UNDERPERFORM,
    _eligible_samples,
    normalize_date,
)

FEATURE_KEYS = ("rel5", "rel20", "rel60", "pe_pct", "pb_pct", "turn_pct", "ml_rank")


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def compute_task_feature_row(
    entry_date: str,
    ts_code: str,
    stock_name: str = "",
    data_dir: str | Path = DEFAULT_DATA_DIR,
) -> dict[str, Any]:
    """Compute the same point-in-time features used by rolling reflection."""
    rows = compute_month_feature_rows(
        entry_date=normalize_date(entry_date),
        stocks=[
            {
                "ts_code": ts_code,
                "stock_name": stock_name,
                "label": "",
                "predicted": "",
                "judge_result": "",
            }
        ],
        data_dir=data_dir,
    )
    return rows[0] if rows else {}


def _feature_stats(rows: list[Mapping[str, Any]]) -> dict[str, tuple[float, float]]:
    stats: dict[str, tuple[float, float]] = {}
    for key in FEATURE_KEYS:
        values = [_number(row.get(key)) for row in rows]
        clean = [value for value in values if value is not None]
        if len(clean) < 2:
            continue
        mean = sum(clean) / len(clean)
        variance = sum((value - mean) ** 2 for value in clean) / (len(clean) - 1)
        std = math.sqrt(variance)
        if std > 1e-12:
            stats[key] = (mean, std)
    return stats


def nearest_neighbors(
    current: Mapping[str, Any],
    samples: list[Mapping[str, Any]],
    *,
    k: int = 12,
    min_common_features: int = 3,
) -> list[dict[str, Any]]:
    """Return standardized Euclidean neighbours with enough shared features."""
    if k <= 0:
        return []
    stats = _feature_stats(list(samples))
    ranked: list[dict[str, Any]] = []
    for sample in samples:
        squared: list[float] = []
        for key, (_, std) in stats.items():
            current_value = _number(current.get(key))
            sample_value = _number(sample.get(key))
            if current_value is None or sample_value is None:
                continue
            squared.append(((current_value - sample_value) / std) ** 2)
        if len(squared) < min_common_features:
            continue
        ranked.append(
            {
                "task_id": str(sample.get("task_id", "")),
                "ts_code": str(sample.get("ts_code", "")),
                "entry_month": str(sample.get("entry_month", "")),
                "label": str(sample.get("label", "")),
                "distance": math.sqrt(sum(squared) / len(squared)),
                "common_features": len(squared),
            }
        )
    ranked.sort(key=lambda row: (row["distance"], row["task_id"]))
    return ranked[:k]


def _wilson_interval(hits: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 1.0
    p = hits / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z / denom * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return center - half, center + half


def _visible(record: Any, as_of_date: str) -> bool:
    available_after = normalize_date(record.metadata.get("available_after"))
    return bool(available_after) and available_after <= normalize_date(as_of_date)


def matching_validated_rules(
    memory: Mem0Memory,
    current: Mapping[str, Any],
    as_of_date: str,
) -> list[Any]:
    """Return only visible FDR-validated rules whose condition matches."""
    matched = []
    for record in memory.store.all_records():
        metadata = record.metadata
        if metadata.get("source") != "rolling_statistical" or not _visible(
            record, as_of_date
        ):
            continue
        condition = metadata.get("condition") or {}
        value = _number(current.get(condition.get("feature")))
        threshold = _number(condition.get("threshold"))
        if value is None or threshold is None:
            continue
        operator = condition.get("operator")
        if (operator == "le" and value <= threshold) or (
            operator == "ge" and value >= threshold
        ):
            matched.append(record)
    matched.sort(
        key=lambda record: (
            float(record.metadata.get("q_value", 1.0)),
            -int(record.metadata.get("validation_support", 0)),
        )
    )
    return matched


def build_feature_evidence_block(
    memory: Mem0Memory,
    *,
    entry_date: str,
    ts_code: str,
    stock_name: str = "",
    data_dir: str | Path = DEFAULT_DATA_DIR,
    k: int = 12,
    min_neighbors: int = 8,
    min_common_features: int = 3,
) -> tuple[str, dict[str, Any]]:
    """Build a conservative post-tool evidence block and an audit payload."""
    as_of_date = normalize_date(entry_date)
    current = compute_task_feature_row(
        entry_date=as_of_date,
        ts_code=ts_code,
        stock_name=stock_name,
        data_dir=data_dir,
    )
    eligible = _eligible_samples(memory.load_samples(), as_of_date)
    neighbors = nearest_neighbors(
        current,
        eligible,
        k=k,
        min_common_features=min_common_features,
    )
    rules = matching_validated_rules(memory, current, as_of_date)

    audit: dict[str, Any] = {
        "as_of_date": as_of_date,
        "ts_code": ts_code,
        "features": {key: current.get(key, "") for key in FEATURE_KEYS},
        "eligible_samples": len(eligible),
        "neighbors": neighbors,
        "matched_rule_ids": [
            str(record.metadata.get("rule_id", record.id)) for record in rules
        ],
    }
    sections: list[str] = []

    if len(neighbors) >= min_neighbors:
        outperform = sum(row["label"] == OUTPERFORM for row in neighbors)
        low, high = _wilson_interval(outperform, len(neighbors))
        audit["neighbor_outperform"] = outperform
        audit["neighbor_wilson_95"] = [round(low, 6), round(high, 6)]
        if low > 0.5 or high < 0.5:
            direction = OUTPERFORM if low > 0.5 else UNDERPERFORM
            hits = outperform if direction == OUTPERFORM else len(neighbors) - outperform
            direction_low, direction_high = (
                (low, high) if direction == OUTPERFORM else (1 - high, 1 - low)
            )
            sections.append(
                "### 相似历史样本（描述性证据）\n"
                f"- 在 {len(neighbors)} 个最近邻已到期样本中，{hits} 个标签为「{direction}」；"
                f"该方向比例的 Wilson 95% 区间为 "
                f"[{direction_low:.0%}, {direction_high:.0%}]。\n"
                "- 该结果只作为当前点时特征的辅助证据，不得覆盖当前 Qlib、动量、"
                "估值和基本面证据。"
            )
        else:
            audit["neighbor_direction"] = "inconclusive"

    if rules:
        lines = []
        for record in rules:
            metadata = record.metadata
            lines.append(
                f"- {record.content} "
                f"(验证 n={metadata.get('validation_support', '?')}, "
                f"q={metadata.get('q_value', '?')})"
            )
        sections.append("### 当前特征命中的已验证规则\n" + "\n".join(lines))

    if not sections:
        return "", audit
    return (
        "\n\n## 系统补充：特征条件化历史证据\n"
        f"以下内容仅使用截至 {as_of_date} 已到期的历史标签，并在当前工具查询完成后生成。\n\n"
        + "\n\n".join(sections),
        audit,
    )
