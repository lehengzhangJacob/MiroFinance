# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

"""Decision-time cross-sectional feature table for monthly memory reflection.

Builds, for one entry month, a compact CSV of what the agent COULD see on the
decision date (relative momentum windows, valuation/turnover percentiles, qlib
ML rank) joined with what actually happened (label) and what the agent
predicted. All features are computed point-in-time from the local benchmark
data cache (`data/ashare/*.csv`) using the same conventions as the ashare MCP
server: close_qfq for stock returns, trade_date string comparison for the
as-of cut, percentile = share of trailing-120-session values <= latest.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "ashare"

_FEATURE_COLUMNS = [
    "ts_code", "name", "rel5", "rel20", "rel60",
    "pe_pct", "pb_pct", "turn_pct", "ml_rank",
    "pred", "correct", "label",
]


def _window_return(closes: pd.Series, periods: int) -> float | None:
    if len(closes) <= periods:
        return None
    last = float(closes.iloc[-1])
    base = float(closes.iloc[-1 - periods])
    if base == 0:
        return None
    return last / base - 1


def _trailing_percentile(series: pd.Series) -> float | None:
    # The percentile must describe the value on the decision date.  Dropping
    # nulls before taking the latest value silently reused stale PE values for
    # loss-making companies whose current PE(TTM) is undefined.
    series = pd.to_numeric(series, errors="coerce").tail(120)
    if series.empty or pd.isna(series.iloc[-1]):
        return None
    latest = float(series.iloc[-1])
    history = series.dropna()
    return float((history <= latest).mean() * 100)


def compute_month_feature_rows(
    entry_date: str,
    stocks: list[dict[str, Any]],
    data_dir: str | Path = DEFAULT_DATA_DIR,
) -> list[dict[str, Any]]:
    """Build structured point-in-time rows for one monthly cross-section.

    Args:
        entry_date: Decision date as YYYYMMDD (the month's first trading day).
        stocks: One dict per pool stock:
            {ts_code, stock_name, label, predicted, judge_result}.
        data_dir: Benchmark data cache directory.

    Rows with entirely missing features still appear so the rolling sample
    ledger preserves the full cross-section.
    """
    data_dir = Path(data_dir)
    entry_date = str(entry_date)

    idx = pd.read_csv(data_dir / "index_000300.SH.csv", dtype={"trade_date": str})
    idx = idx.sort_values("trade_date")
    idx_close = idx[idx.trade_date <= entry_date].close.astype(float)
    idx_ret = {p: _window_return(idx_close, p) for p in (5, 20, 60)}

    try:
        signal = pd.read_csv(data_dir / "qlib_signal.csv", dtype={"entry_date": str})
        signal = signal[signal.entry_date == entry_date].set_index("ts_code")
    except FileNotFoundError:
        signal = pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for stock in stocks:
        ts_code = stock["ts_code"]
        row: dict[str, Any] = {
            "ts_code": ts_code,
            "name": stock.get("stock_name", ""),
            "pred": stock.get("predicted", ""),
            "correct": {"CORRECT": "Y", "INCORRECT": "N"}.get(stock.get("judge_result", ""), ""),
            "label": stock.get("label", ""),
        }

        try:
            daily = pd.read_csv(data_dir / f"daily_{ts_code}.csv", dtype={"trade_date": str})
            closes = daily.sort_values("trade_date")
            closes = closes[closes.trade_date <= entry_date].close_qfq.astype(float)
            for p in (5, 20, 60):
                r_stock, r_idx = _window_return(closes, p), idx_ret[p]
                row[f"rel{p}"] = (
                    round((r_stock - r_idx) * 100, 1)
                    if r_stock is not None and r_idx is not None
                    else ""
                )
        except FileNotFoundError:
            row.update({"rel5": "", "rel20": "", "rel60": ""})

        try:
            basic = pd.read_csv(
                data_dir / f"daily_basic_{ts_code}.csv", dtype={"trade_date": str}
            )
            basic = basic.sort_values("trade_date")
            basic = basic[basic.trade_date <= entry_date]
            for col, key in (("pe_ttm", "pe_pct"), ("pb", "pb_pct"), ("turnover_rate", "turn_pct")):
                pct = _trailing_percentile(basic[col]) if col in basic.columns else None
                row[key] = round(pct, 0) if pct is not None else ""
        except FileNotFoundError:
            row.update({"pe_pct": "", "pb_pct": "", "turn_pct": ""})

        row["ml_rank"] = (
            int(signal.loc[ts_code, "rank"]) if not signal.empty and ts_code in signal.index else ""
        )
        rows.append(row)

    return rows


def compute_month_features(
    entry_date: str,
    stocks: list[dict[str, Any]],
    data_dir: str | Path = DEFAULT_DATA_DIR,
) -> tuple[str, int]:
    """Build the legacy CSV table used by the v2 monthly LLM reflector."""
    rows = compute_month_feature_rows(entry_date, stocks, data_dir=data_dir)
    table = pd.DataFrame(rows, columns=_FEATURE_COLUMNS)
    table = table.sort_values(["label", "ts_code"]).reset_index(drop=True)
    return table.to_csv(index=False), len(table)
