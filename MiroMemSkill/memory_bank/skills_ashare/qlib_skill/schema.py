# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

"""Input validation and metric models for qlib_skill."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Optional

_DATE_RE = re.compile(r"^(\d{4})-?(\d{2})-?(\d{2})$")
_RUN_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def normalize_date(value: str, param: str = "date") -> str:
    """Accept YYYYMMDD or YYYY-MM-DD; return ISO YYYY-MM-DD (qlib convention)."""
    m = _DATE_RE.match(value.strip())
    if not m:
        raise ValueError(f"{param} must be YYYYMMDD or YYYY-MM-DD, got: {value!r}")
    y, mo, d = m.groups()
    if not ("01" <= mo <= "12" and "01" <= d <= "31"):
        raise ValueError(f"{param} has out-of-range month/day: {value!r}")
    return f"{y}-{mo}-{d}"


def validate_run_name(value: str) -> str:
    name = value.strip()
    if not _RUN_NAME_RE.match(name):
        raise ValueError(
            f"run-name must match [A-Za-z0-9._-]+ (no path separators), got: {value!r}"
        )
    return name


def label_expression(horizon: int) -> str:
    """N-trading-day forward return, decided at close of day T.

    Ref($close,-(h+1))/Ref($close,-1)-1 == close[T+h]/close[T] - 1 shifted so
    the label sits on the feature row of day T (qlib convention).
    """
    if horizon < 1:
        raise ValueError(f"label_horizon must be >= 1, got {horizon}")
    return f"Ref($close,-{horizon + 1})/Ref($close,-1)-1"


@dataclass
class SignalMetrics:
    """Daily cross-sectional IC / RankIC summary over the test segment."""

    n_days: int
    ic_mean: float
    ic_std: float
    icir: float
    rank_ic_mean: float
    rank_ic_std: float
    rank_icir: float
    rank_ic_positive_rate: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BacktestMetrics:
    """TopkDropout portfolio backtest summary (daily freq, with cost)."""

    start: str
    end: str
    annualized_return: float          # strategy, cost-deducted
    max_drawdown: float               # strategy, cost-deducted
    information_ratio: float          # strategy, cost-deducted
    excess_annualized_return: float   # vs benchmark, cost-deducted
    excess_information_ratio: float
    excess_max_drawdown: float
    turnover_mean: float
    benchmark: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def envelope(
    api: str,
    params: dict[str, Any],
    payload: dict[str, Any],
    out: Optional[str] = None,
) -> dict[str, Any]:
    """Standard stdout envelope for every subcommand."""
    body: dict[str, Any] = {
        "api": api,
        "params": {k: v for k, v in params.items() if v not in (None, "")},
    }
    body.update(payload)
    if out:
        body["out"] = out
    return body
