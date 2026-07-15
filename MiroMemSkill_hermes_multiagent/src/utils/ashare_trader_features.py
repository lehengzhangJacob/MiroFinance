"""Compact point-in-time history panel for the unified A-share trader.

The trader must compare the whole stock pool in one decision.  Returning every
daily bar for every stock would consume most model context, so this module
compresses the same trailing history into symmetric, auditable features.  All
inputs are cut at ``as_of`` before any feature is calculated.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


DEFAULT_LOOKBACK_DAYS = 0  # 0 = all trading sessions available on or before as_of
RETURN_WINDOWS = (5, 20, 60, 120, 250)


def resolve_history_sessions(n_available: int, lookback_days: int = DEFAULT_LOOKBACK_DAYS) -> int:
    """How many sessions to use; 0 or negative means all available rows."""
    if lookback_days is None or int(lookback_days) <= 0:
        return max(int(n_available), 0)
    return min(int(lookback_days), max(int(n_available), 0))

TRADER_FEATURE_COLUMNS = [
    "ts_code",
    "name",
    "industry",
    "history_sessions",
    "ret5",
    "ret20",
    "ret60",
    "ret120",
    "ret250",
    "rel5",
    "rel20",
    "rel60",
    "rel120",
    "rel250",
    "vol20_ann",
    "vol60_ann",
    "max_dd120",
    "max_dd250",
    "from_high250",
    "ma20_gap",
    "ma60_gap",
    "ma120_gap",
    "amount20_vs120",
    "pe_ttm",
    "pe_pct250",
    "pb",
    "pb_pct250",
    "turnover_rate",
    "turn_pct250",
    "total_mv_yi",
    "ml_score",
    "ml_rank",
    "financial_ann_date",
    "roe",
    "or_yoy",
    "netprofit_yoy",
    "monthly_path",
    "relative_monthly_path",
]


def normalize_date(value: str) -> str:
    """Return a compact YYYYMMDD date or raise a useful error."""
    date = str(value or "").strip().replace("-", "").replace("/", "")
    if len(date) != 8 or not date.isdigit():
        raise ValueError(
            f"Invalid as_of date: {value!r}; expected YYYYMMDD or YYYY-MM-DD"
        )
    return date


def _read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _return(closes: pd.Series, periods: int) -> float | None:
    clean = pd.to_numeric(closes, errors="coerce").dropna()
    if len(clean) <= periods:
        return None
    base = float(clean.iloc[-1 - periods])
    return float(clean.iloc[-1]) / base - 1.0 if base else None


def _annualized_volatility(closes: pd.Series, periods: int) -> float | None:
    returns = pd.to_numeric(closes, errors="coerce").pct_change().dropna().tail(periods)
    if len(returns) < 2:
        return None
    return float(returns.std(ddof=1) * math.sqrt(252))


def _max_drawdown(closes: pd.Series, periods: int) -> float | None:
    clean = pd.to_numeric(closes, errors="coerce").dropna().tail(periods)
    if clean.empty:
        return None
    drawdown = clean / clean.cummax() - 1.0
    return float(drawdown.min())


def _distance_from_high(closes: pd.Series, periods: int) -> float | None:
    clean = pd.to_numeric(closes, errors="coerce").dropna().tail(periods)
    if clean.empty:
        return None
    high = float(clean.max())
    return float(clean.iloc[-1]) / high - 1.0 if high else None


def _moving_average_gap(closes: pd.Series, periods: int) -> float | None:
    clean = pd.to_numeric(closes, errors="coerce").dropna()
    if len(clean) < periods:
        return None
    average = float(clean.tail(periods).mean())
    return float(clean.iloc[-1]) / average - 1.0 if average else None


def _trailing_percentile(series: pd.Series, periods: int) -> float | None:
    trailing = pd.to_numeric(series, errors="coerce").tail(periods)
    if trailing.empty or pd.isna(trailing.iloc[-1]):
        return None
    latest = float(trailing.iloc[-1])
    history = trailing.dropna()
    return float((history <= latest).mean()) if not history.empty else None


def _latest(series: pd.Series) -> float | None:
    if series.empty or pd.isna(series.iloc[-1]):
        return None
    return _number(series.iloc[-1])


def _ratio_of_means(series: pd.Series, short: int, long: int) -> float | None:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if len(clean) < long:
        return None
    long_mean = float(clean.tail(long).mean())
    return float(clean.tail(short).mean()) / long_mean if long_mean else None


def _block_returns(closes: pd.Series, *, block: int = 21, count: int = 12) -> list[float]:
    """Return chronological, non-overlapping block returns ending at as_of."""
    clean = pd.to_numeric(closes, errors="coerce").dropna()
    blocks = min(count, (len(clean) - 1) // block)
    if blocks <= 0:
        return []
    window = clean.iloc[-(blocks * block + 1) :].reset_index(drop=True)
    return [
        float(window.iloc[(index + 1) * block]) / float(window.iloc[index * block]) - 1.0
        for index in range(blocks)
        if float(window.iloc[index * block]) != 0
    ]


def _percent(value: float | None, digits: int = 1) -> float | str:
    return round(value * 100.0, digits) if value is not None else ""


def _rounded(value: float | None, digits: int = 3) -> float | str:
    return round(value, digits) if value is not None else ""


def _path(values: list[float]) -> str:
    return "|".join(f"{value * 100:+.1f}" for value in values)


def compute_trader_feature_rows(
    as_of: str,
    stock_pool: Mapping[str, Mapping[str, Any]],
    *,
    data_dir: str | Path,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> list[dict[str, Any]]:
    """Compute one symmetric point-in-time feature row per pool stock."""
    entry_date = normalize_date(as_of)
    data_path = Path(data_dir)

    index = _read_csv(
        data_path / "index_000300.SH.csv", dtype={"trade_date": str}
    ).sort_values("trade_date")
    index = index[index["trade_date"] <= entry_date]
    index_close = pd.to_numeric(index["close"], errors="coerce").dropna()
    index_sessions = resolve_history_sessions(len(index_close), lookback_days)
    index_returns = {
        window: _return(index_close, window)
        for window in RETURN_WINDOWS
        if window <= index_sessions
    }
    index_blocks = _block_returns(index_close, count=max(1, index_sessions // 21))

    try:
        signal = _read_csv(
            data_path / "qlib_signal.csv", dtype={"entry_date": str}
        )
        signal = signal[signal["entry_date"] <= entry_date].sort_values("entry_date")
    except FileNotFoundError:
        signal = pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for ts_code, info in stock_pool.items():
        row: dict[str, Any] = {
            "ts_code": ts_code,
            "name": str(info.get("name", "")),
            "industry": str(info.get("industry", "")),
        }
        try:
            daily = _read_csv(
                data_path / f"daily_{ts_code}.csv", dtype={"trade_date": str}
            ).sort_values("trade_date")
            daily = daily[daily["trade_date"] <= entry_date]
            closes = pd.to_numeric(daily["close_qfq"], errors="coerce").dropna()
            lookback = resolve_history_sessions(len(closes), lookback_days)
            row["history_sessions"] = lookback
            for window in RETURN_WINDOWS:
                stock_return = _return(closes, window) if window <= lookback else None
                index_return = index_returns.get(window)
                row[f"ret{window}"] = _percent(stock_return)
                row[f"rel{window}"] = _percent(
                    stock_return - index_return
                    if stock_return is not None and index_return is not None
                    else None
                )
            row["vol20_ann"] = _percent(_annualized_volatility(closes, 20))
            row["vol60_ann"] = _percent(_annualized_volatility(closes, 60))
            row["max_dd120"] = _percent(_max_drawdown(closes, min(120, lookback)))
            row["max_dd250"] = _percent(_max_drawdown(closes, lookback))
            row["from_high250"] = _percent(_distance_from_high(closes, lookback))
            for window in (20, 60, 120):
                row[f"ma{window}_gap"] = _percent(
                    _moving_average_gap(closes, window)
                )
            amount_column = "amount" if "amount" in daily.columns else "vol"
            row["amount20_vs120"] = _rounded(
                _ratio_of_means(daily[amount_column], 20, min(120, lookback)), 2
            )
            stock_blocks = _block_returns(
                closes, count=max(1, lookback // 21)
            )
            comparable = min(len(stock_blocks), len(index_blocks))
            relative_blocks = [
                stock_blocks[-comparable + index] - index_blocks[-comparable + index]
                for index in range(comparable)
            ] if comparable else []
            row["monthly_path"] = _path(stock_blocks)
            row["relative_monthly_path"] = _path(relative_blocks)
        except FileNotFoundError:
            row["history_sessions"] = 0
            for window in RETURN_WINDOWS:
                row[f"ret{window}"] = ""
                row[f"rel{window}"] = ""
            for key in (
                "vol20_ann",
                "vol60_ann",
                "max_dd120",
                "max_dd250",
                "from_high250",
                "ma20_gap",
                "ma60_gap",
                "ma120_gap",
                "amount20_vs120",
                "monthly_path",
                "relative_monthly_path",
            ):
                row[key] = ""

        try:
            basic = _read_csv(
                data_path / f"daily_basic_{ts_code}.csv",
                dtype={"trade_date": str},
            ).sort_values("trade_date")
            basic = basic[basic["trade_date"] <= entry_date].tail(lookback)
            latest = basic.iloc[-1] if not basic.empty else None
            for source, value_key, percentile_key in (
                ("pe_ttm", "pe_ttm", "pe_pct250"),
                ("pb", "pb", "pb_pct250"),
                ("turnover_rate", "turnover_rate", "turn_pct250"),
            ):
                series = basic[source] if source in basic.columns else pd.Series(dtype=float)
                row[value_key] = _rounded(_latest(series))
                row[percentile_key] = _percent(
                    _trailing_percentile(series, lookback), 0
                )
            total_mv = (
                _number(latest["total_mv"])
                if latest is not None and "total_mv" in latest.index
                else None
            )
            # Tushare total_mv is in CNY 10,000; divide by 10,000 for CNY 100m.
            row["total_mv_yi"] = _rounded(
                total_mv / 10_000.0 if total_mv is not None else None, 1
            )
        except FileNotFoundError:
            for key in (
                "pe_ttm",
                "pe_pct250",
                "pb",
                "pb_pct250",
                "turnover_rate",
                "turn_pct250",
                "total_mv_yi",
            ):
                row[key] = ""

        stock_signal = (
            signal[signal["ts_code"] == ts_code].tail(1)
            if not signal.empty
            else pd.DataFrame()
        )
        row["ml_score"] = (
            _rounded(_number(stock_signal.iloc[-1]["score"]), 6)
            if not stock_signal.empty
            else ""
        )
        row["ml_rank"] = (
            int(stock_signal.iloc[-1]["rank"]) if not stock_signal.empty else ""
        )

        try:
            financials = _read_csv(
                data_path / f"financials_{ts_code}.csv",
                dtype={"ann_date": str, "end_date": str},
            )
            financials = financials[
                financials["ann_date"].notna()
                & (financials["ann_date"] <= entry_date)
            ].sort_values(["ann_date", "end_date"])
            latest_financial = financials.iloc[-1] if not financials.empty else None
        except FileNotFoundError:
            latest_financial = None
        row["financial_ann_date"] = (
            str(latest_financial["ann_date"])
            if latest_financial is not None
            else ""
        )
        for column in ("roe", "or_yoy", "netprofit_yoy"):
            row[column] = (
                _rounded(_number(latest_financial[column]))
                if latest_financial is not None and column in latest_financial.index
                else ""
            )

        rows.append({column: row.get(column, "") for column in TRADER_FEATURE_COLUMNS})

    return sorted(rows, key=lambda item: str(item["ts_code"]))


def render_trader_universe_context(
    as_of: str,
    stock_pool: Mapping[str, Mapping[str, Any]],
    *,
    data_dir: str | Path,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> str:
    """Render the compact panel returned by the trader's batch MCP tool."""
    entry_date = normalize_date(as_of)
    rows = compute_trader_feature_rows(
        entry_date,
        stock_pool,
        data_dir=data_dir,
        lookback_days=lookback_days,
    )
    sample_sessions = max(
        (int(row.get("history_sessions") or 0) for row in rows),
        default=0,
    )
    window_note = (
        "全部可用交易日"
        if lookback_days is None or int(lookback_days) <= 0
        else f"最近 {sample_sessions} 个交易日"
    )
    frame = pd.DataFrame(rows, columns=TRADER_FEATURE_COLUMNS)
    return "\n".join(
        [
            f"# A股统一交易员全池历史面板（严格点时），决策日 {entry_date}",
            (
                f"历史窗口：{window_note}；ret/rel/vol/dd/high/ma 单位为%；"
                "amount20_vs120 为近20日成交额均值/近120日均值；"
                "估值与换手分位范围 0-100；total_mv_yi 单位亿元。"
            ),
            (
                "monthly_path 与 relative_monthly_path 是从早到晚的约21交易日"
                "收益路径（%）；所有字段只使用决策日及以前数据，不含未来收益或标签。"
            ),
            frame.to_csv(index=False),
        ]
    )
