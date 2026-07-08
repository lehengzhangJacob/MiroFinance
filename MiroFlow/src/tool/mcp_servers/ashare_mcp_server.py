# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

"""Point-in-time A-share market data MCP server.

Serves the local CSV cache under data/ashare (built by
scripts/ashare/fetch_data.py). Every tool requires an `as_of` date and
hard-truncates all rows to trade_date <= as_of, so the agent can never
peek at future data (no lookahead bias in the backtest).
"""

import json
import os
from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastmcp import FastMCP

from src.logging.logger import setup_mcp_logging

setup_mcp_logging(tool_name=os.path.basename(__file__))
mcp = FastMCP("ashare-market-mcp-server")

_DATA_DIR = Path(os.environ.get("ASHARE_DATA_DIR", "data/ashare"))


def _norm_date(as_of: str) -> str:
    """Accept 2024-07-01 or 20240701, return YYYYMMDD."""
    d = as_of.strip().replace("-", "").replace("/", "")
    if len(d) != 8 or not d.isdigit():
        raise ValueError(f"Invalid as_of date: {as_of!r}, expected YYYYMMDD or YYYY-MM-DD")
    return d


@lru_cache(maxsize=32)
def _load_csv(name: str) -> pd.DataFrame:
    path = _DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"data file not found: {path}")
    return pd.read_csv(path, dtype={"trade_date": str, "ann_date": str, "end_date": str})


@lru_cache(maxsize=1)
def _meta() -> dict:
    return json.loads((_DATA_DIR / "meta.json").read_text(encoding="utf-8"))


def _cut(df: pd.DataFrame, as_of: str, date_col: str = "trade_date") -> pd.DataFrame:
    return df[df[date_col] <= _norm_date(as_of)]


def _pct(a: float, b: float) -> float:
    return round((a / b - 1.0) * 100, 2)


def _window_summary(df: pd.DataFrame, close_col: str) -> str:
    lines = []
    closes = df[close_col].astype(float)
    last = closes.iloc[-1]
    for w in (5, 20, 60):
        if len(closes) > w:
            lines.append(f"- 近{w}个交易日收益率: {_pct(last, closes.iloc[-1 - w])}%")
    if len(closes) > 20:
        rets = closes.pct_change().dropna().iloc[-20:]
        lines.append(f"- 近20日日收益率波动(标准差): {round(rets.std() * 100, 2)}%")
    return "\n".join(lines)


@mcp.tool()
def ashare_stock_info() -> str:
    """List the stocks available in this dataset (code, name, industry) and the benchmark index."""
    meta = _meta()
    lines = [f"基准指数: {meta['index_code']} (沪深300)", "股票池:"]
    for code, info in meta["stock_pool"].items():
        lines.append(f"- {code} {info['name']} ({info['industry']})")
    lines.append(f"数据范围: {meta['start_date']} ~ {meta['end_date']} (点时截断以 as_of 为准)")
    return "\n".join(lines)


@mcp.tool()
def ashare_price_history(ts_code: str, as_of: str, lookback_days: int = 60) -> str:
    """Get daily OHLCV history (forward-adjusted, qfq) for one stock up to as_of.

    Args:
        ts_code: Stock code like 600519.SH.
        as_of: Point-in-time cutoff date (YYYY-MM-DD). Data after this date is never returned.
        lookback_days: Number of most recent trading days to return (default 60, max 250).
    """
    lookback_days = max(5, min(int(lookback_days), 250))
    df = _cut(_load_csv(f"daily_{ts_code}.csv"), as_of)
    if df.empty:
        return f"No data for {ts_code} on or before {as_of}."
    tail = df.tail(lookback_days)
    cols = ["trade_date", "open_qfq", "high_qfq", "low_qfq", "close_qfq", "pct_chg", "vol", "amount"]
    out = [
        f"# {ts_code} 日线(前复权), 截至 {as_of}, 最近 {len(tail)} 个交易日",
        "## 动量摘要",
        _window_summary(df, "close_qfq"),
        "## 明细 (CSV)",
        tail[cols].round(3).to_csv(index=False),
    ]
    return "\n".join(out)


@mcp.tool()
def ashare_index_history(as_of: str, lookback_days: int = 60) -> str:
    """Get CSI 300 (000300.SH) daily history up to as_of, for relative-strength comparison.

    Args:
        as_of: Point-in-time cutoff date (YYYY-MM-DD).
        lookback_days: Number of most recent trading days to return (default 60, max 250).
    """
    lookback_days = max(5, min(int(lookback_days), 250))
    meta = _meta()
    df = _cut(_load_csv(f"index_{meta['index_code']}.csv"), as_of)
    if df.empty:
        return f"No index data on or before {as_of}."
    tail = df.tail(lookback_days)
    cols = ["trade_date", "open", "high", "low", "close", "vol", "amount"]
    out = [
        f"# {meta['index_code']} 沪深300 日线, 截至 {as_of}, 最近 {len(tail)} 个交易日",
        "## 动量摘要",
        _window_summary(df, "close"),
        "## 明细 (CSV)",
        tail[cols].round(3).to_csv(index=False),
    ]
    return "\n".join(out)


@mcp.tool()
def ashare_valuation(ts_code: str, as_of: str, lookback_days: int = 120) -> str:
    """Get valuation & liquidity metrics (PE-TTM, PB, turnover, market cap) up to as_of.

    Args:
        ts_code: Stock code like 600519.SH.
        as_of: Point-in-time cutoff date (YYYY-MM-DD).
        lookback_days: Trading days of history for percentile context (default 120, max 250).
    """
    lookback_days = max(5, min(int(lookback_days), 250))
    try:
        df = _cut(_load_csv(f"daily_basic_{ts_code}.csv"), as_of)
    except FileNotFoundError:
        return f"Valuation data unavailable for {ts_code}."
    if df.empty:
        return f"No valuation data for {ts_code} on or before {as_of}."
    tail = df.tail(lookback_days)
    latest = tail.iloc[-1]
    lines = [f"# {ts_code} 估值与流动性, 截至 {as_of}", "## 最新值"]
    for col, label in [
        ("pe_ttm", "PE(TTM)"), ("pb", "PB"), ("ps_ttm", "PS(TTM)"),
        ("turnover_rate", "换手率%"), ("total_mv", "总市值(万元)"),
    ]:
        if col in tail.columns and pd.notna(latest[col]):
            series = tail[col].dropna().astype(float)
            pct_rank = round((series <= float(latest[col])).mean() * 100, 1)
            lines.append(
                f"- {label}: {round(float(latest[col]), 3)} (近{len(series)}日分位 {pct_rank}%)"
            )
    lines.append("## 最近10个交易日明细 (CSV)")
    lines.append(tail.tail(10).round(3).to_csv(index=False))
    return "\n".join(lines)


@mcp.tool()
def ashare_financials(ts_code: str, as_of: str) -> str:
    """Get financial indicators from reports ANNOUNCED on or before as_of (point-in-time safe).

    Args:
        ts_code: Stock code like 600519.SH.
        as_of: Point-in-time cutoff date (YYYY-MM-DD). Filters on announcement date.
    """
    try:
        df = _load_csv(f"financials_{ts_code}.csv")
    except FileNotFoundError:
        return f"Financial data unavailable for {ts_code}."
    df = df[df["ann_date"].notna()]
    df = df[df["ann_date"] <= _norm_date(as_of)]
    if df.empty:
        return f"No financial reports announced on or before {as_of} for {ts_code}."
    tail = df.sort_values("ann_date").tail(6)
    out = [
        f"# {ts_code} 财务指标, 仅含公告日 <= {as_of} 的报告 (最近{len(tail)}期)",
        "字段: ann_date=公告日, end_date=报告期, eps=每股收益, roe=净资产收益率%, "
        "or_yoy=营收同比%, netprofit_yoy=净利润同比%",
        tail.round(3).to_csv(index=False),
    ]
    return "\n".join(out)


if __name__ == "__main__":
    mcp.run()
