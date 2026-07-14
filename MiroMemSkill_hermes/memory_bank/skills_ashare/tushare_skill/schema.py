# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

"""Input validation and output models for tushare_skill.

Input side: normalize/validate dates, ts_code, adjust and format flags
so run.py fails fast with a readable message instead of a cryptic API error.

Output side: one dataclass per API describing the record schema (used for
CSV column ordering and documented in SKILL.md), plus the standard JSON
envelope every subcommand emits.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, fields
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

_TS_CODE_RE = re.compile(r"^\d{6}\.(SH|SZ|BJ)$")
_DATE_RE = re.compile(r"^(\d{4})-?(\d{2})-?(\d{2})$")

VALID_ADJUST = ("qfq", "raw")
VALID_FORMAT = ("json", "csv")


def normalize_date(value: str, param: str = "date") -> str:
    """Accept YYYYMMDD or YYYY-MM-DD; return YYYYMMDD."""
    m = _DATE_RE.match(value.strip())
    if not m:
        raise ValueError(f"{param} must be YYYYMMDD or YYYY-MM-DD, got: {value!r}")
    y, mo, d = m.groups()
    if not ("01" <= mo <= "12" and "01" <= d <= "31"):
        raise ValueError(f"{param} has out-of-range month/day: {value!r}")
    return f"{y}{mo}{d}"


def validate_ts_code(value: str) -> str:
    code = value.strip().upper()
    if not _TS_CODE_RE.match(code):
        raise ValueError(
            f"ts_code must look like 600519.SH / 000001.SZ / 830000.BJ, got: {value!r}"
        )
    return code


def validate_choice(value: str, choices: tuple[str, ...], param: str) -> str:
    v = value.strip().lower()
    if v not in choices:
        raise ValueError(f"{param} must be one of {choices}, got: {value!r}")
    return v


# ---------------------------------------------------------------------------
# Output record models (one per API)
# ---------------------------------------------------------------------------


@dataclass
class DailyBar:
    """`daily` (+ optional qfq columns from `adj_factor`)."""

    ts_code: str
    trade_date: str
    open: Optional[float]
    high: Optional[float]
    low: Optional[float]
    close: Optional[float]
    pre_close: Optional[float]
    pct_chg: Optional[float]
    vol: Optional[float]
    amount: Optional[float]
    adj_factor: Optional[float] = None
    open_qfq: Optional[float] = None
    high_qfq: Optional[float] = None
    low_qfq: Optional[float] = None
    close_qfq: Optional[float] = None


@dataclass
class IndexBar:
    """`index_daily`."""

    ts_code: str
    trade_date: str
    open: Optional[float]
    high: Optional[float]
    low: Optional[float]
    close: Optional[float]
    vol: Optional[float]
    amount: Optional[float]


@dataclass
class Valuation:
    """`daily_basic`."""

    ts_code: str
    trade_date: str
    pe_ttm: Optional[float]
    pb: Optional[float]
    ps_ttm: Optional[float]
    turnover_rate: Optional[float]
    total_mv: Optional[float]
    circ_mv: Optional[float]


@dataclass
class FinIndicator:
    """`fina_indicator` — ann_date is the point-in-time key."""

    ts_code: str
    ann_date: str
    end_date: str
    eps: Optional[float]
    roe: Optional[float]
    grossprofit_margin: Optional[float]
    netprofit_margin: Optional[float]
    or_yoy: Optional[float]
    netprofit_yoy: Optional[float]


@dataclass
class StockInfo:
    """`stock_basic` — current snapshot, NOT point-in-time."""

    ts_code: str
    symbol: str
    name: str
    area: Optional[str]
    industry: Optional[str]
    market: Optional[str]
    list_date: Optional[str]


@dataclass
class TradeCalDay:
    """`trade_cal`."""

    cal_date: str
    is_open: int


RESPONSE_MODELS: dict[str, type] = {
    "daily": DailyBar,
    "index": IndexBar,
    "valuation": Valuation,
    "financials": FinIndicator,
    "stock-info": StockInfo,
    "trade-cal": TradeCalDay,
}


def model_columns(api: str) -> list[str]:
    """Ordered column names for the given subcommand's record model."""
    model = RESPONSE_MODELS[api]
    return [f.name for f in fields(model)]


# ---------------------------------------------------------------------------
# Response envelope
# ---------------------------------------------------------------------------


def envelope(
    api: str,
    params: dict[str, Any],
    items: list[dict[str, Any]],
    as_of: Optional[str] = None,
    out: Optional[str] = None,
) -> dict[str, Any]:
    """Standard JSON envelope for every subcommand.

    When `out` is set the CSV was written to disk and `items` are omitted
    from stdout to keep the terminal readable.
    """
    body: dict[str, Any] = {
        "api": api,
        "params": {k: v for k, v in params.items() if v not in (None, "")},
        "as_of": as_of,
        "count": len(items),
    }
    if out:
        body["out"] = out
    else:
        body["items"] = items
    return body
