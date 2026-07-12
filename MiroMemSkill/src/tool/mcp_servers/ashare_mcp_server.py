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
from src.memory.monthly_reflection import compute_month_feature_rows
from src.utils.ashare_momentum import render_relative_momentum_baseline
from src.utils.ashare_trader_features import (
    render_trader_universe_context,
    resolve_history_sessions,
)

setup_mcp_logging(tool_name=os.path.basename(__file__))
mcp = FastMCP("ashare-market-mcp-server")

_DATA_DIR = Path(os.environ.get("ASHARE_DATA_DIR", "data/ashare"))

# 0 = return every trading row available on or before as_of (no artificial cap).
DEFAULT_PRICE_LOOKBACK = 0
DEFAULT_VALUATION_LOOKBACK = 0


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


def _history_tail(df: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    sessions = resolve_history_sessions(len(df), lookback_days)
    return df if sessions >= len(df) else df.tail(sessions)


def _pct(a: float, b: float) -> float:
    return round((a / b - 1.0) * 100, 2)


def _window_summary(df: pd.DataFrame, close_col: str) -> str:
    lines = []
    closes = df[close_col].astype(float)
    last = closes.iloc[-1]
    for w in (5, 20, 60, 120):
        if len(closes) > w:
            lines.append(f"- 近{w}个交易日收益率: {_pct(last, closes.iloc[-1 - w])}%")
    full_w = len(closes) - 1
    if full_w > 120:
        lines.append(f"- 近{full_w}个交易日收益率: {_pct(last, closes.iloc[-1 - full_w])}%")
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
def ashare_price_history(ts_code: str, as_of: str, lookback_days: int = DEFAULT_PRICE_LOOKBACK) -> str:
    """Get daily OHLCV history (forward-adjusted, qfq) for one stock up to as_of.

    Args:
        ts_code: Stock code like 600519.SH.
        as_of: Point-in-time cutoff date (YYYY-MM-DD). Data after this date is never returned.
        lookback_days: Most recent trading days to return; 0 = all rows on or before as_of.
    """
    df = _cut(_load_csv(f"daily_{ts_code}.csv"), as_of)
    if df.empty:
        return f"No data for {ts_code} on or before {as_of}."
    tail = _history_tail(df, lookback_days)
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
def ashare_index_history(as_of: str, lookback_days: int = DEFAULT_PRICE_LOOKBACK) -> str:
    """Get CSI 300 (000300.SH) daily history up to as_of, for relative-strength comparison.

    Args:
        as_of: Point-in-time cutoff date (YYYY-MM-DD).
        lookback_days: Most recent trading days to return; 0 = all rows on or before as_of.
    """
    meta = _meta()
    df = _cut(_load_csv(f"index_{meta['index_code']}.csv"), as_of)
    if df.empty:
        return f"No index data on or before {as_of}."
    tail = _history_tail(df, lookback_days)
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
def ashare_valuation(ts_code: str, as_of: str, lookback_days: int = DEFAULT_VALUATION_LOOKBACK) -> str:
    """Get valuation & liquidity metrics (PE-TTM, PB, turnover, market cap) up to as_of.

    Args:
        ts_code: Stock code like 600519.SH.
        as_of: Point-in-time cutoff date (YYYY-MM-DD).
        lookback_days: Trading days of history for percentile context; 0 = all available.
    """
    try:
        df = _cut(_load_csv(f"daily_basic_{ts_code}.csv"), as_of)
    except FileNotFoundError:
        return f"Valuation data unavailable for {ts_code}."
    if df.empty:
        return f"No valuation data for {ts_code} on or before {as_of}."
    tail = _history_tail(df, lookback_days)
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
    lines.append("## 明细 (CSV)")
    lines.append(tail.round(3).to_csv(index=False))
    return "\n".join(lines)


@mcp.tool()
def ashare_ml_signal(ts_code: str, as_of: str) -> str:
    """Get the point-in-time qlib ML score (LightGBM+Alpha158, walk-forward) for one stock.

    Scores are cross-sectionally comparable within the same month: rank 1 means
    the highest predicted 20-day forward return in the pool. Each monthly score
    comes from a model trained ONLY on data whose labels settled before that
    month's decision date, so no future information is embedded. Rows after
    as_of are never returned.

    Args:
        ts_code: Stock code like 600519.SH.
        as_of: Point-in-time cutoff date (YYYY-MM-DD).
    """
    try:
        df = _load_csv("qlib_signal.csv")
    except FileNotFoundError:
        return "ML signal unavailable (qlib_signal.csv not built; see qlib_skill walkforward)."
    df = df[df["entry_date"].astype(str) <= _norm_date(as_of)]
    df = df[df["ts_code"] == ts_code]
    if df.empty:
        return f"No ML signal for {ts_code} on or before {as_of}."
    tail = df.sort_values("entry_date")
    latest = tail.iloc[-1]
    out = [
        f"# {ts_code} qlib机器学习信号 (LightGBM+Alpha158, 逐月walk-forward训练), 截至 {as_of}",
        f"最新: 决策日 {latest['entry_date']}, score={latest['score']:.4f}, "
        f"池内排名 {int(latest['rank'])}/{int(latest['n_stocks'])} (rank 1 = 预测未来20日收益最高)",
        "## 全部可用月度信号 (CSV)",
        tail[["entry_date", "score", "rank", "n_stocks"]].to_csv(index=False),
    ]
    return "\n".join(out)


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
    tail = df.sort_values("ann_date")
    out = [
        f"# {ts_code} 财务指标, 仅含公告日 <= {as_of} 的报告 (共{len(tail)}期)",
        "字段: ann_date=公告日, end_date=报告期, eps=每股收益, roe=净资产收益率%, "
        "or_yoy=营收同比%, netprofit_yoy=净利润同比%",
        tail.round(3).to_csv(index=False),
    ]
    return "\n".join(out)


@mcp.tool()
def ashare_trader_universe_context(
    as_of: str,
    lookback_days: int = DEFAULT_PRICE_LOOKBACK,
) -> str:
    """Get one compact 16-stock history panel for a unified portfolio decision.

    The panel gives every stock the same point-in-time information budget:
    absolute and CSI-300-relative returns over 5/20/60/120/250 sessions,
    volatility, drawdown, trend, liquidity, 250-session valuation percentiles,
    latest announced fundamentals, walk-forward Qlib signals, and compressed
    monthly paths.  It never includes future returns or benchmark labels.

    Args:
        as_of: Point-in-time cutoff date (YYYY-MM-DD or YYYYMMDD).
        lookback_days: History window in trading sessions; 0 = all available on or before as_of.
    """
    return render_trader_universe_context(
        as_of,
        _meta()["stock_pool"],
        data_dir=_DATA_DIR,
        lookback_days=lookback_days,
    )


@mcp.tool()
def ashare_momentum_baseline(
    as_of: str,
    window: int = 20,
    top_k: int = 4,
) -> str:
    """Build a point-in-time relative-momentum soft anchor for the whole pool.

    The tool ranks every stock by its trailing adjusted return minus the CSI
    300 return over the same window, then forms a max-25%-per-stock top-k
    reference portfolio.  It never reads rows after ``as_of`` and never uses
    realized holding-period returns or benchmark labels.

    Args:
        as_of: Decision-date cutoff (YYYY-MM-DD or YYYYMMDD).
        window: Relative-return window; one of 5, 20, 60, 120, or 250.
        top_k: Number of leading stocks in the anchor, between 1 and 4.
    """
    return render_relative_momentum_baseline(
        as_of,
        _meta()["stock_pool"],
        data_dir=_DATA_DIR,
        window=window,
        top_k=top_k,
    )


@mcp.tool()
def ashare_cross_section_snapshot(as_of: str) -> str:
    """Get a point-in-time feature snapshot for every stock in the A-share pool.

    This batch tool is intended for cross-sectional ranking.  It returns only
    information available on or before ``as_of``: relative 5/20/60-day
    momentum, valuation/liquidity percentiles, and the walk-forward Qlib score
    and rank.  It never returns future returns or benchmark labels.

    Args:
        as_of: Point-in-time cutoff date (YYYY-MM-DD or YYYYMMDD).
    """
    entry_date = _norm_date(as_of)
    meta = _meta()
    stocks = [
        {
            "ts_code": ts_code,
            "stock_name": info["name"],
            "label": "",
            "predicted": "",
            "judge_result": "",
        }
        for ts_code, info in meta["stock_pool"].items()
    ]
    rows = compute_month_feature_rows(entry_date, stocks, data_dir=_DATA_DIR)
    industries = {
        ts_code: info.get("industry", "")
        for ts_code, info in meta["stock_pool"].items()
    }

    try:
        signal = _load_csv("qlib_signal.csv")
        signal = signal[signal["entry_date"].astype(str) == entry_date].set_index(
            "ts_code"
        )
    except FileNotFoundError:
        signal = pd.DataFrame()

    for row in rows:
        ts_code = row["ts_code"]
        row["industry"] = industries.get(ts_code, "")
        row["ml_score"] = (
            round(float(signal.loc[ts_code, "score"]), 6)
            if not signal.empty and ts_code in signal.index
            else ""
        )
        try:
            financials = _load_csv(f"financials_{ts_code}.csv")
            financials = financials[financials["ann_date"].notna()]
            financials = financials[
                financials["ann_date"].astype(str) <= entry_date
            ].sort_values("ann_date")
            latest = financials.iloc[-1] if not financials.empty else None
        except FileNotFoundError:
            latest = None
        row["financial_ann_date"] = (
            str(latest["ann_date"]) if latest is not None else ""
        )
        for column in ("roe", "or_yoy", "netprofit_yoy"):
            row[column] = (
                round(float(latest[column]), 3)
                if latest is not None
                and column in latest.index
                and pd.notna(latest[column])
                else ""
            )

    columns = [
        "ts_code",
        "name",
        "industry",
        "rel5",
        "rel20",
        "rel60",
        "pe_pct",
        "pb_pct",
        "turn_pct",
        "ml_score",
        "ml_rank",
        "financial_ann_date",
        "roe",
        "or_yoy",
        "netprofit_yoy",
    ]
    frame = pd.DataFrame(rows, columns=columns).sort_values("ts_code")
    return "\n".join(
        [
            f"# A股池截面快照（严格点时），决策日 {entry_date}",
            "rel5/20/60 为相对沪深300超额收益率(%)；分位字段范围 0-100；"
            "ml_rank=1 表示 Qlib 预测未来20日收益最高；财务字段来自决策日前最近公告。",
            frame.to_csv(index=False),
        ]
    )


if __name__ == "__main__":
    mcp.run()
