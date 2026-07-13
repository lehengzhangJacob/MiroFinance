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
    stock_basic_all(ts_code, name, industry, list_date, delist_date, status)
    fina_cache(ts_code, ann_date, end_date, ...)   -- lazy, per-stock
    fetch_progress(kind, trade_date)               -- resume bookkeeping

Usage:
    conda run -n Miro python scripts/ashare/fetch_full_market.py
Safe to re-run: already-fetched dates are skipped.
"""

from __future__ import annotations

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


def trading_days(conn: sqlite3.Connection) -> list[str]:
    return [
        d
        for (d,) in conn.execute(
            "SELECT cal_date FROM trade_cal WHERE is_open=1 "
            "AND cal_date BETWEEN ? AND ? ORDER BY cal_date",
            (START_DATE, END_DATE),
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
) -> None:
    days = trading_days(conn)
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


def main() -> None:
    token = load_token()
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    init_tables(conn)

    fetch_stock_basic(conn, token)
    fetch_by_date(
        conn,
        token,
        kind="daily",
        api_name="daily",
        fields="ts_code,trade_date,close,pct_chg,vol,amount",
        table="market_daily",
        columns=["ts_code", "trade_date", "close", "pct_chg", "vol", "amount"],
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
    )

    n1 = conn.execute("SELECT COUNT(*) FROM market_daily").fetchone()[0]
    n2 = conn.execute("SELECT COUNT(*) FROM market_daily_basic").fetchone()[0]
    nc = conn.execute(
        "SELECT COUNT(DISTINCT ts_code) FROM market_daily"
    ).fetchone()[0]
    print(f"\nDONE_FULL_MARKET daily={n1} basic={n2} stocks={nc}")
    conn.close()


if __name__ == "__main__":
    main()
