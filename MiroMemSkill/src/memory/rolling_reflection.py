# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

"""Expanding-window statistical reflection for A-share direction tasks.

The previous monthly reflector asked an LLM to discover thresholds from one
16-stock cross-section.  That is a multiple-testing problem with almost no
sample behind each lesson.  This module keeps rule discovery deterministic:

* labels are eligible only after their 20-day exit date;
* thresholds are fixed before looking at labels;
* the most recent months are a temporal validation set;
* support, lift, cross-month consistency and Benjamini-Hochberg FDR gates must
  all pass before a rule can enter memory.

The LLM never decides whether a pattern is statistically valid.  It only sees
the small set of already validated conditional rules at decision time.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

OUTPERFORM, UNDERPERFORM = "跑赢", "跑输"


_FEATURE_LABELS = {
    "rel5": "近5日相对沪深300超额收益(%)",
    "rel20": "近20日相对沪深300超额收益(%)",
    "rel60": "近60日相对沪深300超额收益(%)",
    "pe_pct": "PE(TTM)近120日分位(%)",
    "pb_pct": "PB近120日分位(%)",
    "turn_pct": "换手率近120日分位(%)",
    "ml_rank": "Qlib截面排名",
}

# Fixed, domain-interpretable thresholds.  They are deliberately not derived
# from the benchmark labels, which prevents the reflector from inventing a
# convenient cut point after seeing one month's winners and losers.
_THRESHOLDS = {
    "rel5": (-10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 10.0, 15.0),
    "rel20": (-10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 10.0, 15.0),
    "rel60": (-10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 10.0, 15.0),
    "pe_pct": (10.0, 25.0, 50.0, 75.0, 90.0),
    "pb_pct": (10.0, 25.0, 50.0, 75.0, 90.0),
    "turn_pct": (10.0, 25.0, 50.0, 75.0, 90.0),
    "ml_rank": (4.0, 8.0, 12.0),
}


def normalize_date(value: Any) -> str:
    """Normalize a date-like value to YYYYMMDD, or return an empty string."""
    digits = re.sub(r"\D", "", str(value or ""))
    return digits[:8] if len(digits) >= 8 else ""


def normalize_month(value: Any) -> str:
    """Normalize a month/date-like value to YYYY-MM."""
    digits = re.sub(r"\D", "", str(value or ""))
    return f"{digits[:4]}-{digits[4:6]}" if len(digits) >= 6 else ""


@dataclass(frozen=True)
class RollingRuleConfig:
    """Statistical gates for expanding-window rule discovery."""

    min_samples: int = 64
    min_history_months: int = 6
    validation_months: int = 2
    min_train_support: int = 16
    min_validation_support: int = 8
    min_support_months: int = 4
    min_validation_months: int = 2
    min_accuracy: float = 0.60
    min_lift: float = 0.08
    min_month_consistency: float = 0.60
    fdr_q: float = 0.10
    max_rules: int = 3

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "RollingRuleConfig":
        values = values or {}

        def value(name: str, default: Any) -> Any:
            return values.get(f"rolling_{name}", default)

        return cls(
            min_samples=int(value("min_samples", cls.min_samples)),
            min_history_months=int(value("min_history_months", cls.min_history_months)),
            validation_months=int(value("validation_months", cls.validation_months)),
            min_train_support=int(value("min_train_support", cls.min_train_support)),
            min_validation_support=int(
                value("min_validation_support", cls.min_validation_support)
            ),
            min_support_months=int(value("min_support_months", cls.min_support_months)),
            min_validation_months=int(
                value("min_validation_months", cls.min_validation_months)
            ),
            min_accuracy=float(value("min_accuracy", cls.min_accuracy)),
            min_lift=float(value("min_lift", cls.min_lift)),
            min_month_consistency=float(
                value("min_month_consistency", cls.min_month_consistency)
            ),
            fdr_q=float(value("fdr_q", cls.fdr_q)),
            max_rules=int(value("max_rules", cls.max_rules)),
        )


@dataclass(frozen=True)
class Candidate:
    feature: str
    operator: str
    threshold: float

    @property
    def rule_id(self) -> str:
        threshold = f"{self.threshold:g}".replace("-", "m").replace(".", "p")
        return f"{self.feature}_{self.operator}_{threshold}"

    def matches(self, row: Mapping[str, Any]) -> bool:
        value = _number(row.get(self.feature))
        if value is None:
            return False
        if self.operator == "le":
            return value <= self.threshold
        return value >= self.threshold


@dataclass(frozen=True)
class ValidatedRule:
    rule_id: str
    content: str
    metadata: dict[str, Any]


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _ratio(hits: int, support: int) -> float:
    return hits / support if support else 0.0


def _binomial_tail(n: int, hits: int, probability: float) -> float:
    """P[X >= hits] for X~Binomial(n, probability), without scipy."""
    if n <= 0:
        return 1.0
    if probability <= 0:
        return 0.0 if hits > 0 else 1.0
    if probability >= 1:
        return 1.0
    return min(
        1.0,
        sum(
            math.comb(n, k) * probability**k * (1.0 - probability) ** (n - k)
            for k in range(hits, n + 1)
        ),
    )


def _bh_qvalues(pvalues: Sequence[float]) -> list[float]:
    """Benjamini-Hochberg adjusted q-values in original order."""
    if not pvalues:
        return []
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted = [1.0] * len(pvalues)
    running = 1.0
    m = len(pvalues)
    for rank_index in range(m - 1, -1, -1):
        original_index = order[rank_index]
        rank = rank_index + 1
        running = min(running, pvalues[original_index] * m / rank)
        adjusted[original_index] = min(1.0, running)
    return adjusted


def _eligible_samples(
    samples: Sequence[Mapping[str, Any]], as_of_date: str
) -> list[dict[str, Any]]:
    cutoff = normalize_date(as_of_date)
    if not cutoff:
        return []

    # Last write wins so a resumed run can repair an earlier partial row.
    deduped: dict[str, dict[str, Any]] = {}
    for raw in samples:
        task_id = str(raw.get("task_id", "") or "")
        label = str(raw.get("label", "") or "")
        exit_date = normalize_date(raw.get("exit_date") or raw.get("available_after"))
        entry_date = normalize_date(raw.get("entry_date"))
        if not task_id or label not in (OUTPERFORM, UNDERPERFORM):
            continue
        if not exit_date or exit_date > cutoff or not entry_date:
            continue
        row = dict(raw)
        row["entry_date"] = entry_date
        row["exit_date"] = exit_date
        row["entry_month"] = normalize_month(raw.get("entry_month") or entry_date)
        deduped[task_id] = row
    return sorted(deduped.values(), key=lambda r: (r["entry_date"], r["task_id"]))


def _candidate_rows(
    rows: Sequence[Mapping[str, Any]], candidate: Candidate
) -> list[Mapping[str, Any]]:
    return [row for row in rows if candidate.matches(row)]


def _valid_feature_rows(
    rows: Sequence[Mapping[str, Any]], feature: str
) -> list[Mapping[str, Any]]:
    return [row for row in rows if _number(row.get(feature)) is not None]


def _direction_stats(
    rows: Sequence[Mapping[str, Any]], direction: str
) -> tuple[int, int, float]:
    support = len(rows)
    hits = sum(row.get("label") == direction for row in rows)
    return support, hits, _ratio(hits, support)


def _format_threshold(value: float) -> str:
    return str(int(value)) if value.is_integer() else f"{value:g}"


def _build_rule(
    result: dict[str, Any],
    *,
    as_of_date: str,
    available_after: str,
    source_months: list[str],
) -> ValidatedRule:
    candidate: Candidate = result["candidate"]
    direction = result["direction"]
    operator = "≤" if candidate.operator == "le" else "≥"
    threshold = _format_threshold(candidate.threshold)
    validation_months = result["validation_month_count"]
    content = (
        f"当{_FEATURE_LABELS[candidate.feature]} {operator} {threshold}时，历史扩展窗口中"
        f"未来20日相对沪深300『{direction}』的概率较高。"
        f"训练样本 {result['train_hits']}/{result['train_support']}"
        f"（{result['train_accuracy']:.0%}），最近{validation_months}个已到期月份的时间验证样本 "
        f"{result['validation_hits']}/{result['validation_support']}"
        f"（{result['validation_accuracy']:.0%}），较同期基准提升 "
        f"{result['validation_lift']:.0%}，FDR q={result['q_value']:.3f}。"
        "仅在当前股票满足该条件时作为辅助证据，不得覆盖当前Qlib、动量和基本面证据。"
    )
    metadata = {
        "rule_id": candidate.rule_id,
        "condition": {
            "feature": candidate.feature,
            "operator": candidate.operator,
            "threshold": candidate.threshold,
        },
        "direction": direction,
        "entry_month": source_months[-1],
        "available_after": available_after,
        "generated_for_date": normalize_date(as_of_date),
        "source": "rolling_statistical",
        "source_months": source_months,
        "source_tasks": [],
        "functional_stance": "neutral",
        "tags": ["rolling", "validated", candidate.feature],
        "train_support": result["train_support"],
        "train_accuracy": round(result["train_accuracy"], 6),
        "validation_support": result["validation_support"],
        "validation_accuracy": round(result["validation_accuracy"], 6),
        "validation_lift": round(result["validation_lift"], 6),
        "q_value": round(result["q_value"], 6),
        "total_support": result["total_support"],
        "total_accuracy": round(result["total_accuracy"], 6),
        "support_months": result["support_months"],
    }
    return ValidatedRule(candidate.rule_id, content, metadata)


def mine_rolling_rules(
    samples: Sequence[Mapping[str, Any]],
    as_of_date: str,
    config: RollingRuleConfig | None = None,
) -> list[ValidatedRule]:
    """Mine temporally validated rules from all labels available at ``as_of_date``."""
    cfg = config or RollingRuleConfig()
    rows = _eligible_samples(samples, as_of_date)
    months = sorted({str(row["entry_month"]) for row in rows})
    if len(rows) < cfg.min_samples or len(months) < cfg.min_history_months:
        return []
    if cfg.validation_months <= 0 or len(months) <= cfg.validation_months:
        return []

    validation_months = set(months[-cfg.validation_months :])
    train = [row for row in rows if row["entry_month"] not in validation_months]
    validation = [row for row in rows if row["entry_month"] in validation_months]
    candidates = [
        Candidate(feature, operator, threshold)
        for feature, thresholds in _THRESHOLDS.items()
        for threshold in thresholds
        for operator in ("le", "ge")
    ]

    evaluated: list[dict[str, Any]] = []
    for candidate in candidates:
        train_condition = _candidate_rows(train, candidate)
        if len(train_condition) < cfg.min_train_support:
            continue

        outperform = sum(row["label"] == OUTPERFORM for row in train_condition)
        underperform = len(train_condition) - outperform
        if outperform == underperform:
            continue
        direction = OUTPERFORM if outperform > underperform else UNDERPERFORM
        train_support, train_hits, train_accuracy = _direction_stats(
            train_condition, direction
        )
        train_valid = _valid_feature_rows(train, candidate.feature)
        train_base = _ratio(
            sum(row["label"] == direction for row in train_valid), len(train_valid)
        )
        if (
            train_accuracy < cfg.min_accuracy
            or train_accuracy - train_base < cfg.min_lift
        ):
            continue

        validation_condition = _candidate_rows(validation, candidate)
        if len(validation_condition) < cfg.min_validation_support:
            continue
        validation_support, validation_hits, validation_accuracy = _direction_stats(
            validation_condition, direction
        )
        validation_valid = _valid_feature_rows(validation, candidate.feature)
        validation_base = _ratio(
            sum(row["label"] == direction for row in validation_valid),
            len(validation_valid),
        )
        represented_validation_months = sorted(
            {str(row["entry_month"]) for row in validation_condition}
        )
        if len(represented_validation_months) < cfg.min_validation_months:
            continue

        # Cross-month preselection must use training data only.  Looking at
        # validation labels here would shrink the tested family before FDR and
        # make the correction anti-conservative.
        train_by_month: dict[str, list[Mapping[str, Any]]] = {}
        for row in train_condition:
            train_by_month.setdefault(str(row["entry_month"]), []).append(row)
        counted_train_months = [
            group for group in train_by_month.values() if len(group) >= 2
        ]
        if len(counted_train_months) < cfg.min_support_months:
            continue
        supportive_months = sum(
            _direction_stats(group, direction)[2] >= 0.5
            for group in counted_train_months
        )
        month_consistency = supportive_months / len(counted_train_months)
        if month_consistency < cfg.min_month_consistency:
            continue

        all_condition = _candidate_rows(rows, candidate)
        all_support_months = len(
            {
                str(row["entry_month"])
                for row in all_condition
                if str(row.get("entry_month", ""))
            }
        )
        total_support, total_hits, total_accuracy = _direction_stats(
            all_condition, direction
        )
        evaluated.append(
            {
                "candidate": candidate,
                "direction": direction,
                "train_support": train_support,
                "train_hits": train_hits,
                "train_accuracy": train_accuracy,
                "validation_support": validation_support,
                "validation_hits": validation_hits,
                "validation_accuracy": validation_accuracy,
                "validation_lift": validation_accuracy - validation_base,
                "validation_month_count": len(represented_validation_months),
                "p_value": _binomial_tail(
                    validation_support, validation_hits, validation_base
                ),
                "total_support": total_support,
                "total_hits": total_hits,
                "total_accuracy": total_accuracy,
                "support_months": all_support_months,
                "month_consistency": month_consistency,
            }
        )

    qvalues = _bh_qvalues([result["p_value"] for result in evaluated])
    for result, qvalue in zip(evaluated, qvalues):
        result["q_value"] = qvalue

    accepted = [
        result
        for result in evaluated
        if result["q_value"] <= cfg.fdr_q
        and result["validation_accuracy"] >= cfg.min_accuracy
        and result["validation_lift"] >= cfg.min_lift
        and result["total_accuracy"] >= cfg.min_accuracy
    ]
    accepted.sort(
        key=lambda result: (
            result["q_value"],
            -result["validation_lift"],
            -result["validation_accuracy"],
            -result["total_support"],
        )
    )

    # Keep at most one rule per feature.  This prevents overlapping thresholds
    # for the same signal from becoming a contradictory prompt-time rule set.
    selected: list[dict[str, Any]] = []
    used_features: set[str] = set()
    for result in accepted:
        feature = result["candidate"].feature
        if feature in used_features:
            continue
        selected.append(result)
        used_features.add(feature)
        if len(selected) >= cfg.max_rules:
            break

    available_after = max(str(row["exit_date"]) for row in rows)
    source_months = sorted({str(row["entry_month"]) for row in rows})
    return [
        _build_rule(
            result,
            as_of_date=as_of_date,
            available_after=available_after,
            source_months=source_months,
        )
        for result in selected
    ]
