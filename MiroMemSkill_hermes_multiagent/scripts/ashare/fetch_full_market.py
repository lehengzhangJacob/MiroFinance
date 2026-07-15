# SPDX-FileCopyrightText: 2026 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

"""Mirror full-market A-share daily data from Tushare into SQLite.

Fetches by trade_date (one call returns the whole market), so ~510 sessions
x 2 endpoints ~= 1000 calls covers every SH/SZ stock from 2023-06 through
2025-07.  The mirror backs the open-universe trader benchmark: agents screen
the entire market point-in-time instead of a hand-picked 16-stock pool.

Tables added to data/ashare_pools.db:
    market_daily(ts_code, trade_date, close, pct_chg, vol, amount)
    market_daily_basic(ts_code, trade_date, pe_ttm, pb, turnover_rate,
                       total_mv, circ_mv)
    etf_daily(ts_code, trade_date, close, pct_chg, vol, amount)
    stock_basic_all(ts_code, name, industry, list_date, delist_date, status)
    fina_cache(ts_code, ann_date, end_date, ...)   -- lazy, per-stock
    fetch_progress(kind, trade_date)               -- resume bookkeeping

Usage:
    conda run -n Miro python scripts/ashare/fetch_full_market.py \
        [--db PATH] [--start YYYYMMDD] [--end YYYYMMDD]
Safe to re-run: already-fetched dates are skipped.  Extending --end on an
existing mirror fetches only the missing sessions (incl. trade_cal,
index_daily and the ETF core, which are range-checked rather than marker
checked).
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_ROOT = REPO_ROOT.parent
DB_PATH = REPO_ROOT / "data" / "ashare_pools.db"
TUSHARE_API = "http://api.tushare.pro"

START_DATE = "20230601"
END_DATE = "20250731"
INDEX_CODE = "000300.SH"
ETF_CORE_CODES = (
    "510300.SH",  # 沪深300ETF
    "510500.SH",  # 中证500ETF
    "512100.SH",  # 中证1000ETF
)


def load_token() -> str:
    token_file = AGENT_ROOT / "tushare_token"
    if not token_file.exists():
        sys.exit(f"tushare token file not found: {token_file}")
    raw = token_file.read_text(encoding="utf-8").strip()
    if "=" in raw:
        raw = raw.split("=", 1)[1].strip().strip('"').strip("'")
    return raw


def tushare_query(
    token: str, api_name: str, params: dict, fields: str = ""
) -> pd.DataFrame:
    for attempt in range(6):
        resp = requests.post(
            TUSHARE_API,
            json={
                "api_name": api_name,
                "token": token,
                "params": params,
                "fields": fields,
            },
            timeout=90,
        )
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("code") == 0:
            data = payload["data"]
            return pd.DataFrame(data["items"], columns=data["fields"])
        msg = str(payload.get("msg"))
        if "每分钟" in msg or "频率" in msg or "limit" in msg.lower():
            wait = 15 * (attempt + 1)
            print(f"  rate limited on {api_name} {params}, sleep {wait}s")
            time.sleep(wait)
            continue
        raise RuntimeError(f"tushare {api_name} error: {msg}")
    raise RuntimeError(f"tushare {api_name}: retries exhausted")


def init_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS market_daily (
            ts_code TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            close REAL, pct_chg REAL, vol REAL, amount REAL,
            PRIMARY KEY (ts_code, trade_date)
        );
        CREATE TABLE IF NOT EXISTS market_daily_basic (
            ts_code TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            pe_ttm REAL, pb REAL, turnover_rate REAL,
            total_mv REAL, circ_mv REAL,
            PRIMARY KEY (ts_code, trade_date)
        );
        CREATE TABLE IF NOT EXISTS etf_daily (
            ts_code TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            close REAL, pct_chg REAL, vol REAL, amount REAL,
            PRIMARY KEY (ts_code, trade_date)
        );
        CREATE TABLE IF NOT EXISTS stock_basic_all (
            ts_code TEXT PRIMARY KEY,
            name TEXT, industry TEXT, list_date TEXT,
            delist_date TEXT, status TEXT
        );
        CREATE TABLE IF NOT EXISTS fina_cache (
            ts_code TEXT NOT NULL,
            ann_date TEXT NOT NULL,
            end_date TEXT NOT NULL,
            eps REAL, roe REAL, grossprofit_margin REAL,
            netprofit_margin REAL, or_yoy REAL, netprofit_yoy REAL,
            PRIMARY KEY (ts_code, ann_date, end_date)
        );
        CREATE TABLE IF NOT EXISTS fina_cache_meta (
            ts_code TEXT PRIMARY KEY,
            fetched_at TEXT
        );
        CREATE TABLE IF NOT EXISTS fetch_progress (
            kind TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            PRIMARY KEY (kind, trade_date)
        );
        CREATE INDEX IF NOT EXISTS idx_market_daily_date
            ON market_daily(trade_date);
        CREATE INDEX IF NOT EXISTS idx_market_basic_date
            ON market_daily_basic(trade_date);
        CREATE INDEX IF NOT EXISTS idx_etf_daily_date
            ON etf_daily(trade_date);
        """
    )
    conn.commit()


def fetch_stock_basic(conn: sqlite3.Connection, token: str) -> None:
    frames = []
    for status in ("L", "D", "P"):
        df = tushare_query(
            token,
            "stock_basic",
            {"list_status": status},
            fields="ts_code,name,industry,list_date,delist_date",
        )
        df["status"] = status
        frames.append(df)
        time.sleep(0.2)
    allb = pd.concat(frames, ignore_index=True)
    allb = allb[allb["ts_code"].str.endswith((".SH", ".SZ"))]
    allb.to_sql("tmp_basic", conn, if_exists="replace", index=False)
    conn.execute("INSERT OR REPLACE INTO stock_basic_all SELECT * FROM tmp_basic")
    conn.execute("DROP TABLE tmp_basic")
    conn.commit()
    print(f"stock_basic_all: {len(allb)} rows (L/D/P)")


def fetch_trade_cal(
    conn: sqlite3.Connection, token: str, start: str, end: str
) -> None:
    """Ensure trade_cal covers [start, end]; fetch missing tail/head."""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS trade_cal "
        "(cal_date TEXT PRIMARY KEY, is_open INTEGER)"
    )
    row = conn.execute("SELECT MIN(cal_date), MAX(cal_date) FROM trade_cal").fetchone()
    have_lo, have_hi = row if row else (None, None)
    if have_lo is not None and have_lo <= start and have_hi >= end:
        print(f"trade_cal: already covers {start}..{end}")
        return
    cal = tushare_query(
        token,
        "trade_cal",
        {"exchange": "SSE", "start_date": start, "end_date": end},
        fields="cal_date,is_open",
    )
    if cal.empty:
        raise RuntimeError(f"trade_cal returned no rows for {start}..{end}")
    cal.to_sql("tmp_cal", conn, if_exists="replace", index=False)
    conn.execute(
        "INSERT OR REPLACE INTO trade_cal (cal_date, is_open) "
        "SELECT cal_date, is_open FROM tmp_cal"
    )
    conn.execute("DROP TABLE tmp_cal")
    conn.commit()
    n_open = int(cal["is_open"].astype(int).sum())
    print(f"trade_cal: merged {len(cal)} rows ({n_open} trading days) {start}..{end}")


def fetch_index_daily(
    conn: sqlite3.Connection, token: str, start: str, end: str
) -> None:
    """Ensure index_daily (CSI300 close) covers the requested range."""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS index_daily ("
        "ts_code TEXT NOT NULL, trade_date TEXT NOT NULL, close REAL, "
        "PRIMARY KEY (ts_code, trade_date))"
    )
    row = conn.execute(
        "SELECT MAX(trade_date) FROM index_daily WHERE ts_code=?", (INDEX_CODE,)
    ).fetchone()
    have_hi = row[0] if row else None
    expected_last = conn.execute(
        "SELECT MAX(cal_date) FROM trade_cal WHERE is_open=1 AND cal_date<=?",
        (end,),
    ).fetchone()[0]
    if have_hi is not None and expected_last is not None and have_hi >= expected_last:
        print(f"index_daily: already covers up to {have_hi}")
        return
    idx = tushare_query(
        token,
        "index_daily",
        {"ts_code": INDEX_CODE, "start_date": start, "end_date": end},
        fields="ts_code,trade_date,close",
    )
    if idx.empty:
        raise RuntimeError(f"index_daily returned no rows for {start}..{end}")
    idx = idx[["ts_code", "trade_date", "close"]].copy()
    idx.to_sql("tmp_idx", conn, if_exists="replace", index=False)
    conn.execute("INSERT OR REPLACE INTO index_daily SELECT * FROM tmp_idx")
    conn.execute("DROP TABLE tmp_idx")
    conn.commit()
    print(
        f"index_daily {INDEX_CODE}: merged {len(idx)} rows "
        f"({idx.trade_date.min()}..{idx.trade_date.max()})"
    )


def trading_days(conn: sqlite3.Connection, start: str, end: str) -> list[str]:
    return [
        d
        for (d,) in conn.execute(
            "SELECT cal_date FROM trade_cal WHERE is_open=1 "
            "AND cal_date BETWEEN ? AND ? ORDER BY cal_date",
            (start, end),
        )
    ]


def done_dates(conn: sqlite3.Connection, kind: str) -> set[str]:
    return {
        d
        for (d,) in conn.execute(
            "SELECT trade_date FROM fetch_progress WHERE kind=?", (kind,)
        )
    }


def fetch_by_date(
    conn: sqlite3.Connection,
    token: str,
    kind: str,
    api_name: str,
    fields: str,
    table: str,
    columns: list[str],
    start: str,
    end: str,
) -> None:
    days = trading_days(conn, start, end)
    done = done_dates(conn, kind)
    todo = [d for d in days if d not in done]
    print(f"{kind}: {len(todo)} dates to fetch ({len(done)} done)")
    for i, day in enumerate(todo, 1):
        df = tushare_query(token, api_name, {"trade_date": day}, fields=fields)
        df = df[df["ts_code"].str.endswith((".SH", ".SZ"))]
        df = df[columns].copy()
        df.to_sql("tmp_day", conn, if_exists="replace", index=False)
        conn.execute(
            f"INSERT OR REPLACE INTO {table} SELECT * FROM tmp_day"
        )
        conn.execute(
            "INSERT OR REPLACE INTO fetch_progress VALUES (?,?)", (kind, day)
        )
        conn.commit()
        if i % 25 == 0 or i == len(todo):
            print(f"  {kind} {i}/{len(todo)} (latest {day}, {len(df)} rows)")
        time.sleep(0.12)
    conn.execute("DROP TABLE IF EXISTS tmp_day")
    conn.commit()


def fetch_core_etfs(
    conn: sqlite3.Connection, token: str, start: str, end: str
) -> None:
    """Fetch the fixed v4 ETF core; range-checked so --end extensions work."""
    expected_last = conn.execute(
        "SELECT MAX(cal_date) FROM trade_cal WHERE is_open=1 AND cal_date<=?",
        (end,),
    ).fetchone()[0]
    todo = []
    for code in ETF_CORE_CODES:
        row = conn.execute(
            "SELECT MAX(trade_date) FROM etf_daily WHERE ts_code=?", (code,)
        ).fetchone()
        have_hi = row[0] if row else None
        if have_hi is None or expected_last is None or have_hi < expected_last:
            todo.append(code)
    print(f"etf_daily: {len(todo)} codes to fetch ({len(ETF_CORE_CODES) - len(todo)} current)")
    fields = "ts_code,trade_date,close,pct_chg,vol,amount"
    columns = ["ts_code", "trade_date", "close", "pct_chg", "vol", "amount"]
    for code in todo:
        df = tushare_query(
            token,
            "fund_daily",
            {
                "ts_code": code,
                "start_date": start,
                "end_date": end,
            },
            fields=fields,
        )
        if df.empty:
            raise RuntimeError(f"fund_daily returned no rows for {code}")
        df = df[columns].copy()
        df.to_sql("tmp_etf", conn, if_exists="replace", index=False)
        conn.execute("INSERT OR REPLACE INTO etf_daily SELECT * FROM tmp_etf")
        conn.execute(
            "INSERT OR REPLACE INTO fetch_progress VALUES (?,?)",
            ("etf_daily", code),
        )
        conn.commit()
        print(
            f"  etf_daily {code}: {len(df)} rows "
            f"({df.trade_date.min()}..{df.trade_date.max()})"
        )
        time.sleep(0.2)
    conn.execute("DROP TABLE IF EXISTS tmp_etf")
    conn.commit()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(DB_PATH), help="sqlite mirror path")
    parser.add_argument("--start", default=START_DATE, help="YYYYMMDD")
    parser.add_argument("--end", default=END_DATE, help="YYYYMMDD")
    args = parser.parse_args()

    token = load_token()
    conn = sqlite3.connect(args.db)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    init_tables(conn)

    fetch_trade_cal(conn, token, args.start, args.end)
    fetch_index_daily(conn, token, args.start, args.end)
    fetch_stock_basic(conn, token)
    fetch_core_etfs(conn, token, args.start, args.end)
    fetch_by_date(
        conn,
        token,
        kind="daily",
        api_name="daily",
        fields="ts_code,trade_date,close,pct_chg,vol,amount",
        table="market_daily",
        columns=["ts_code", "trade_date", "close", "pct_chg", "vol", "amount"],
        start=args.start,
        end=args.end,
    )
    fetch_by_date(
        conn,
        token,
        kind="daily_basic",
        api_name="daily_basic",
        fields="ts_code,trade_date,pe_ttm,pb,turnover_rate,total_mv,circ_mv",
        table="market_daily_basic",
        columns=[
            "ts_code", "trade_date", "pe_ttm", "pb",
            "turnover_rate", "total_mv", "circ_mv",
        ],
        start=args.start,
        end=args.end,
    )

    n1 = conn.execute("SELECT COUNT(*) FROM market_daily").fetchone()[0]
    n2 = conn.execute("SELECT COUNT(*) FROM market_daily_basic").fetchone()[0]
    nc = conn.execute(
        "SELECT COUNT(DISTINCT ts_code) FROM market_daily"
    ).fetchone()[0]
    ne = conn.execute("SELECT COUNT(*) FROM etf_daily").fetchone()[0]
    hi = conn.execute("SELECT MAX(trade_date) FROM market_daily").fetchone()[0]
    print(
        f"\nDONE_FULL_MARKET daily={n1} basic={n2} "
        f"etf_daily={ne} stocks={nc} max_date={hi}"
    )
    conn.close()


if __name__ == "__main__":
    main()
