# SPDX-FileCopyrightText: 2026 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

"""Open-universe (whole A-share market) point-in-time MCP server.

Backed by the SQLite mirror data/ashare_pools.db built by
scripts/ashare/fetch_full_market.py (market_daily, market_daily_basic,
stock_basic_all, index_daily, trade_cal).  Every tool takes `as_of` and
hard-truncates to trade_date <= as_of, so the agent can screen and inspect
any of ~5,300 A-share stocks without look-ahead.

Return math note: market_daily.close is UNADJUSTED; all window returns are
compounded from Tushare pct_chg (which is computed against pre_close and is
therefore split/dividend-safe).  Financials are lazily fetched from Tushare
per stock and cached in the same database.
"""

from __future__ import annotations

import os
import sqlite3
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from fastmcp import FastMCP

from src.logging.logger import setup_mcp_logging

setup_mcp_logging(tool_name=os.path.basename(__file__))
mcp = FastMCP("ashare-open-market-mcp-server")

_DB_PATH = Path(
    os.environ.get("ASHARE_OPEN_DB", "data/ashare_pools.db")
)
_TOKEN_FILE = Path(
    os.environ.get("TUSHARE_TOKEN_FILE", str(Path.cwd().parent / "tushare_token"))
)
TUSHARE_API = "http://api.tushare.pro"
INDEX_CODE = "000300.SH"


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{_DB_PATH}?mode=ro", uri=True)
    conn.execute("PRAGMA busy_timeout=15000")
    return conn


def _rw_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(_DB_PATH)
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def _norm_date(as_of: str) -> str:
    d = str(as_of).strip().replace("-", "").replace("/", "")
    if len(d) != 8 or not d.isdigit():
        raise ValueError(f"Invalid as_of: {as_of!r}, expected YYYYMMDD or YYYY-MM-DD")
    return d


def _norm_code(ts_code: str) -> str:
    code = str(ts_code).strip().upper()
    if len(code) == 6 and code.isdigit():
        suffix = ".SH" if code[0] in "569" else ".SZ"
        code = code + suffix
    return code


def _sessions_before(conn: sqlite3.Connection, as_of: str, n: int) -> list[str]:
    rows = conn.execute(
        "SELECT cal_date FROM trade_cal WHERE is_open=1 AND cal_date<=? "
        "ORDER BY cal_date DESC LIMIT ?",
        (as_of, n),
    ).fetchall()
    return [r[0] for r in rows][::-1]


def _index_window_return(conn: sqlite3.Connection, dates: list[str]) -> float | None:
    if len(dates) < 2:
        return None
    rows = dict(
        conn.execute(
            "SELECT trade_date, close FROM index_daily WHERE ts_code=? "
            "AND trade_date IN (%s)" % ",".join("?" * len(dates)),
            (INDEX_CODE, *dates),
        ).fetchall()
    )
    first, last = rows.get(dates[0]), rows.get(dates[-1])
    if not first or not last:
        return None
    return (last / first - 1.0) * 100.0


@mcp.tool()
def ashare_universe_stats(as_of: str) -> str:
    """Get the size and composition of the tradable A-share universe at as_of.

    Args:
        as_of: Point-in-time cutoff (YYYY-MM-DD or YYYYMMDD).
    """
    day = _norm_date(as_of)
    conn = _conn()
    try:
        latest = conn.execute(
            "SELECT MAX(trade_date) FROM market_daily WHERE trade_date<=?", (day,)
        ).fetchone()[0]
        if not latest:
            return f"No market data on or before {as_of}."
        n = conn.execute(
            "SELECT COUNT(*) FROM market_daily WHERE trade_date=?", (latest,)
        ).fetchone()[0]
        st = conn.execute(
            "SELECT COUNT(*) FROM market_daily d JOIN stock_basic_all b USING(ts_code) "
            "WHERE d.trade_date=? AND b.name LIKE '%ST%'",
            (latest,),
        ).fetchone()[0]
        ind = conn.execute(
            "SELECT COUNT(DISTINCT industry) FROM stock_basic_all WHERE status='L'"
        ).fetchone()[0]
        return (
            f"截至 {as_of}（最近交易日 {latest}）：全市场可见 {n} 只股票，"
            f"其中 ST/*ST {st} 只；申万行业约 {ind} 个。\n"
            f"可用工具：ashare_screen_market 全市场排序筛选；"
            f"ashare_price_history / ashare_valuation / ashare_financials 看个股；"
            f"ashare_stock_info 按代码或名称查基本信息。"
        )
    finally:
        conn.close()


@mcp.tool()
def ashare_screen_market(
    as_of: str,
    sort_by: str = "rel_momentum",
    window: int = 20,
    top_n: int = 20,
    ascending: bool = False,
    exclude_st: bool = True,
    min_list_sessions: int = 250,
    min_avg_amount: float = 50000.0,
    industry: str = "",
) -> str:
    """Screen and rank the ENTIRE A-share market point-in-time.

    Args:
        as_of: Cutoff date (YYYY-MM-DD or YYYYMMDD); only data <= as_of is used.
        sort_by: One of rel_momentum (window return minus CSI300), momentum
            (raw window return), turnover_rate, pe_ttm, pb, total_mv, amount.
        window: Trading-session window for momentum/volatility (5-250).
        top_n: Rows to return (1-50).
        ascending: Sort ascending instead of descending (e.g. low pe_ttm).
        exclude_st: Drop ST/*ST/delisting-risk names.
        min_list_sessions: Require at least this many sessions of history.
        min_avg_amount: Min average daily amount (千元) over the window.
        industry: Optional exact industry filter, e.g. 半导体.
    """
    day = _norm_date(as_of)
    window = max(5, min(250, int(window)))
    top_n = max(1, min(50, int(top_n)))
    conn = _conn()
    try:
        dates = _sessions_before(conn, day, window + 1)
        if len(dates) < window + 1:
            return f"Not enough history on or before {as_of}."
        placeholder = ",".join("?" * len(dates))
        df = pd.read_sql_query(
            f"SELECT ts_code, trade_date, pct_chg, amount FROM market_daily "
            f"WHERE trade_date IN ({placeholder})",
            conn,
            params=dates,
        )
        info = pd.read_sql_query(
            "SELECT ts_code, name, industry, list_date FROM stock_basic_all",
            conn,
        )
        hist_count = pd.read_sql_query(
            "SELECT ts_code, COUNT(*) AS n FROM market_daily "
            "WHERE trade_date<=? GROUP BY ts_code",
            conn,
            params=(day,),
        )
        latest = dates[-1]
        basic = pd.read_sql_query(
            "SELECT ts_code, pe_ttm, pb, turnover_rate, total_mv FROM "
            "market_daily_basic WHERE trade_date=?",
            conn,
            params=(latest,),
        )
        index_ret = _index_window_return(conn, dates)
    finally:
        conn.close()

    # window return via pct_chg compounding (dividend/split safe); skip the
    # first row of the window (it anchors the entry close).
    df = df.sort_values(["ts_code", "trade_date"])
    grp = df[df["trade_date"] > dates[0]].groupby("ts_code")
    ret = (grp["pct_chg"].apply(lambda s: ((1 + s / 100.0).prod() - 1) * 100.0)
           .rename("window_ret"))
    vol = grp["pct_chg"].std().rename("vol")
    amt = grp["amount"].mean().rename("avg_amount")
    traded = grp.size().rename("traded_sessions")
    panel = pd.concat([ret, vol, amt, traded], axis=1).reset_index()
    panel = panel.merge(info, on="ts_code", how="left")
    panel = panel.merge(hist_count, on="ts_code", how="left")
    panel = panel.merge(basic, on="ts_code", how="left")

    panel = panel[panel["n"].fillna(0) >= int(min_list_sessions)]
    panel = panel[panel["traded_sessions"] >= max(3, window // 2)]
    panel = panel[panel["avg_amount"].fillna(0) >= float(min_avg_amount)]
    if exclude_st:
        panel = panel[~panel["name"].fillna("").str.contains("ST|退")]
    if industry.strip():
        panel = panel[panel["industry"] == industry.strip()]
    if index_ret is not None:
        panel["rel_ret"] = panel["window_ret"] - index_ret
    else:
        panel["rel_ret"] = panel["window_ret"]

    sort_map = {
        "rel_momentum": "rel_ret",
        "momentum": "window_ret",
        "turnover_rate": "turnover_rate",
        "pe_ttm": "pe_ttm",
        "pb": "pb",
        "total_mv": "total_mv",
        "amount": "avg_amount",
    }
    key = sort_map.get(sort_by, "rel_ret")
    panel = panel.dropna(subset=[key])
    panel = panel.sort_values(key, ascending=bool(ascending))
    head = panel.head(top_n)

    idx_line = (
        f"沪深300近{window}日收益 {index_ret:+.2f}%"
        if index_ret is not None
        else "沪深300窗口收益不可用"
    )
    lines = [
        f"# 全A股筛选 截至 {as_of}（窗口 {window} 日，按 {sort_by} "
        f"{'升序' if ascending else '降序'}，剔除ST={exclude_st}，"
        f"最少上市 {min_list_sessions} 个交易日，窗口日均成交额≥{min_avg_amount:.0f}千元）",
        idx_line,
        "代码 | 名称 | 行业 | 窗口收益% | 相对沪深300% | 日收益率std% | "
        "日均成交额(千元) | PE_TTM | PB | 换手率% | 总市值(万元)",
    ]
    for _, r in head.iterrows():
        mv = "" if pd.isna(r.total_mv) else f"{r.total_mv:,.0f}"
        lines.append(
            f"{r.ts_code} | {r['name']} | {r.industry} | {r.window_ret:+.2f} | "
            f"{r.rel_ret:+.2f} | {0.0 if pd.isna(r.vol) else r.vol:.2f} | "
            f"{0.0 if pd.isna(r.avg_amount) else r.avg_amount:,.0f} | "
            f"{'' if pd.isna(r.pe_ttm) else round(r.pe_ttm, 1)} | "
            f"{'' if pd.isna(r.pb) else round(r.pb, 2)} | "
            f"{'' if pd.isna(r.turnover_rate) else round(r.turnover_rate, 2)} | "
            f"{mv}"
        )
    lines.append(f"(符合过滤条件的股票共 {len(panel)} 只，仅显示前 {len(head)} 只)")
    return "\n".join(lines)


@mcp.tool()
def ashare_price_history(ts_code: str, as_of: str, lookback_days: int = 60) -> str:
    """Get one stock's daily bars (close, pct_chg, vol, amount) up to as_of.

    Args:
        ts_code: Stock code like 600519.SH (or bare 6 digits).
        as_of: Point-in-time cutoff; rows after it are never returned.
        lookback_days: Most recent sessions to return (1-250, default 60).
    """
    day = _norm_date(as_of)
    code = _norm_code(ts_code)
    lookback = max(1, min(250, int(lookback_days)))
    conn = _conn()
    try:
        df = pd.read_sql_query(
            "SELECT trade_date, close, pct_chg, vol, amount FROM market_daily "
            "WHERE ts_code=? AND trade_date<=? ORDER BY trade_date DESC LIMIT ?",
            conn,
            params=(code, day, lookback),
        )
        name_row = conn.execute(
            "SELECT name, industry FROM stock_basic_all WHERE ts_code=?", (code,)
        ).fetchone()
    finally:
        conn.close()
    if df.empty:
        return f"No data for {code} on or before {as_of}."
    df = df.sort_values("trade_date")
    rets = (1 + df["pct_chg"] / 100.0).cumprod()
    summary = []
    for w in (5, 20, 60):
        if len(df) > w:
            r = (rets.iloc[-1] / rets.iloc[-1 - w] - 1) * 100
            summary.append(f"近{w}日收益 {r:+.2f}%")
    label = f"{name_row[0]}（{name_row[1]}）" if name_row else ""
    lines = [
        f"# {code} {label} 日线截至 {as_of}，最近 {len(df)} 个交易日",
        "（close 为未复权价，收益请以 pct_chg 复合计算；下方摘要已按 pct_chg 复合）",
        "; ".join(summary),
        "trade_date | close | pct_chg% | vol(手) | amount(千元)",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"{r.trade_date} | {r.close:.2f} | {r.pct_chg:+.2f} | "
            f"{r.vol:,.0f} | {r.amount:,.0f}"
        )
    return "\n".join(lines)


@mcp.tool()
def ashare_valuation(ts_code: str, as_of: str, lookback_days: int = 60) -> str:
    """Get one stock's valuation/liquidity series (PE-TTM, PB, turnover, mv).

    Args:
        ts_code: Stock code like 600519.SH.
        as_of: Point-in-time cutoff (YYYY-MM-DD or YYYYMMDD).
        lookback_days: Most recent sessions to return (1-250, default 60).
    """
    day = _norm_date(as_of)
    code = _norm_code(ts_code)
    lookback = max(1, min(250, int(lookback_days)))
    conn = _conn()
    try:
        df = pd.read_sql_query(
            "SELECT trade_date, pe_ttm, pb, turnover_rate, total_mv, circ_mv "
            "FROM market_daily_basic WHERE ts_code=? AND trade_date<=? "
            "ORDER BY trade_date DESC LIMIT ?",
            conn,
            params=(code, day, lookback),
        )
    finally:
        conn.close()
    if df.empty:
        return f"No valuation data for {code} on or before {as_of}."
    df = df.sort_values("trade_date")
    last = df.iloc[-1]
    lines = [
        f"# {code} 估值与流动性 截至 {as_of}（最近 {len(df)} 个交易日）",
        f"最新：PE_TTM={last.pe_ttm}, PB={last.pb}, 换手率={last.turnover_rate}%, "
        f"总市值={last.total_mv:,.0f}万元, 流通市值={last.circ_mv:,.0f}万元",
        "trade_date | pe_ttm | pb | turnover% | total_mv(万元)",
    ]
    step = max(1, len(df) // 30)
    for _, r in df.iloc[::step].iterrows():
        lines.append(
            f"{r.trade_date} | {'' if pd.isna(r.pe_ttm) else round(r.pe_ttm, 1)} | "
            f"{'' if pd.isna(r.pb) else round(r.pb, 2)} | "
            f"{'' if pd.isna(r.turnover_rate) else round(r.turnover_rate, 2)} | "
            f"{'' if pd.isna(r.total_mv) else format(round(r.total_mv), ',')}"
        )
    return "\n".join(lines)


@mcp.tool()
def ashare_stock_info(query: str, as_of: str = "") -> str:
    """Look up stocks by code or name substring (whole market).

    Args:
        query: 6-digit code, full ts_code, or a name fragment like 宁德.
        as_of: Optional cutoff used only to flag not-yet-listed stocks.
    """
    q = str(query).strip().upper()
    conn = _conn()
    try:
        rows = conn.execute(
            "SELECT ts_code, name, industry, list_date, delist_date, status "
            "FROM stock_basic_all WHERE ts_code LIKE ? OR name LIKE ? LIMIT 20",
            (f"%{q}%", f"%{query.strip()}%"),
        ).fetchall()
    finally:
        conn.close()
    if not rows:
        return f"No stock matches {query!r}."
    day = _norm_date(as_of) if as_of else ""
    lines = ["ts_code | 名称 | 行业 | 上市日 | 退市日 | 状态"]
    for code, name, ind, ld, dd, st in rows:
        flag = ""
        if day and ld and ld > day:
            flag = "（as_of 时尚未上市！不可交易)"
        if day and dd and dd <= day:
            flag = "（as_of 时已退市！不可交易)"
        lines.append(f"{code} | {name} | {ind} | {ld} | {dd or '-'} | {st}{flag}")
    return "\n".join(lines)


@mcp.tool()
def ashare_index_history(as_of: str, lookback_days: int = 60) -> str:
    """Get CSI 300 (000300.SH) daily closes up to as_of.

    Args:
        as_of: Point-in-time cutoff (YYYY-MM-DD or YYYYMMDD).
        lookback_days: Most recent sessions to return (1-500, default 60).
    """
    day = _norm_date(as_of)
    lookback = max(1, min(500, int(lookback_days)))
    conn = _conn()
    try:
        df = pd.read_sql_query(
            "SELECT trade_date, close FROM index_daily WHERE ts_code=? "
            "AND trade_date<=? ORDER BY trade_date DESC LIMIT ?",
            conn,
            params=(INDEX_CODE, day, lookback),
        )
    finally:
        conn.close()
    if df.empty:
        return f"No index data on or before {as_of}."
    df = df.sort_values("trade_date")
    closes = df["close"]
    summary = []
    for w in (5, 20, 60):
        if len(df) > w:
            summary.append(
                f"近{w}日 {((closes.iloc[-1] / closes.iloc[-1 - w]) - 1) * 100:+.2f}%"
            )
    lines = [
        f"# 沪深300 截至 {as_of}: " + "; ".join(summary),
        "trade_date | close",
    ]
    step = max(1, len(df) // 40)
    for _, r in df.iloc[::step].iterrows():
        lines.append(f"{r.trade_date} | {r.close:.2f}")
    return "\n".join(lines)


@mcp.tool()
def ashare_market_breadth(as_of: str, window: int = 20) -> str:
    """Whole-market breadth: advancers, share above zero window return.

    Args:
        as_of: Point-in-time cutoff (YYYY-MM-DD or YYYYMMDD).
        window: Session window for the positive-return share (default 20).
    """
    day = _norm_date(as_of)
    window = max(5, min(120, int(window)))
    conn = _conn()
    try:
        dates = _sessions_before(conn, day, window + 1)
        if len(dates) < window + 1:
            return f"Not enough history on or before {as_of}."
        latest = dates[-1]
        adv, dec = conn.execute(
            "SELECT SUM(pct_chg>0), SUM(pct_chg<0) FROM market_daily "
            "WHERE trade_date=?",
            (latest,),
        ).fetchone()
        placeholder = ",".join("?" * len(dates[1:]))
        df = pd.read_sql_query(
            f"SELECT ts_code, pct_chg FROM market_daily "
            f"WHERE trade_date IN ({placeholder})",
            conn,
            params=dates[1:],
        )
        index_ret = _index_window_return(conn, dates)
    finally:
        conn.close()
    cum = df.groupby("ts_code")["pct_chg"].apply(
        lambda s: (1 + s / 100.0).prod() - 1
    )
    pos_share = float((cum > 0).mean()) * 100
    total = len(cum)
    idx_label = f"{index_ret:+.2f}%" if index_ret is not None else "n/a"
    regime = (
        "risk_on" if pos_share > 60 else
        "defensive" if pos_share < 40 else "neutral"
    )
    return (
        f"# 全A市场广度 截至 {as_of}（最近交易日 {latest}）\n"
        f"- 当日上涨/下跌家数: {int(adv)}/{int(dec)}\n"
        f"- 近{window}日累计收益为正的股票占比: {pos_share:.1f}% (样本 {total})\n"
        f"- 沪深300近{window}日: {idx_label}\n"
        f"- 简易状态判定: {regime}（>60% risk_on, <40% defensive, 其余 neutral）"
    )


def _load_token() -> str:
    raw = _TOKEN_FILE.read_text(encoding="utf-8").strip()
    if "=" in raw:
        raw = raw.split("=", 1)[1].strip().strip('"').strip("'")
    return raw


@mcp.tool()
def ashare_financials(ts_code: str, as_of: str) -> str:
    """Get financial indicators ANNOUNCED on or before as_of (PIT-safe).

    Lazily fetched from Tushare per stock and cached locally; repeated calls
    are served from cache.

    Args:
        ts_code: Stock code like 600519.SH.
        as_of: Cutoff date; filters on announcement date (ann_date).
    """
    day = _norm_date(as_of)
    code = _norm_code(ts_code)
    conn = _rw_conn()
    try:
        cached = conn.execute(
            "SELECT 1 FROM fina_cache_meta WHERE ts_code=?", (code,)
        ).fetchone()
        if not cached:
            try:
                token = _load_token()
                resp = requests.post(
                    TUSHARE_API,
                    json={
                        "api_name": "fina_indicator",
                        "token": token,
                        "params": {
                            "ts_code": code,
                            "start_date": "20220101",
                            "end_date": "20250731",
                        },
                        "fields": (
                            "ts_code,ann_date,end_date,eps,roe,"
                            "grossprofit_margin,netprofit_margin,"
                            "or_yoy,netprofit_yoy"
                        ),
                    },
                    timeout=60,
                )
                resp.raise_for_status()
                payload = resp.json()
                if payload.get("code") != 0:
                    return f"财务数据暂不可用: {payload.get('msg')}"
                data = payload["data"]
                df = pd.DataFrame(data["items"], columns=data["fields"])
                df = df.dropna(subset=["ann_date"]).drop_duplicates(
                    subset=["ts_code", "ann_date", "end_date"]
                )
                df.to_sql("tmp_fina", conn, if_exists="replace", index=False)
                conn.execute(
                    "INSERT OR REPLACE INTO fina_cache SELECT * FROM tmp_fina"
                )
                conn.execute("DROP TABLE tmp_fina")
                conn.execute(
                    "INSERT OR REPLACE INTO fina_cache_meta VALUES (?,?)",
                    (code, datetime.now().isoformat(timespec="seconds")),
                )
                conn.commit()
                time.sleep(0.1)
            except requests.RequestException as exc:
                return f"财务数据获取失败（网络）: {exc}"
        rows = conn.execute(
            "SELECT ann_date, end_date, eps, roe, grossprofit_margin, "
            "netprofit_margin, or_yoy, netprofit_yoy FROM fina_cache "
            "WHERE ts_code=? AND ann_date<=? ORDER BY ann_date",
            (code, day),
        ).fetchall()
    finally:
        conn.close()
    if not rows:
        return f"No financials announced for {code} on or before {as_of}."
    lines = [
        f"# {code} 已公告财务指标 截至 {as_of}（按公告日过滤，点时安全）",
        "公告日 | 报告期 | EPS | ROE% | 毛利率% | 净利率% | 营收YoY% | 净利YoY%",
    ]
    for r in rows[-12:]:
        lines.append(" | ".join("" if v is None else str(v) for v in r))
    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run()
