# SPDX-FileCopyrightText: 2026 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

"""Build point-in-time balanced/random A-share pools and store them in SQLite.

Motivation: the legacy 16-stock trader pool was hand-picked (pool v4 even
added four known 2024H2-2025H1 outperformers), so momentum-top4 results on it
carry look-ahead/selection bias.  This script constructs pools using ONLY
information visible on the PIT snapshot date (the last trading day before the
backtest starts) with mechanical rules, fetches their daily bars from Tushare,
and stores everything in one SQLite database.

Pools created:
  legacy16      the existing hand-picked pool (for comparison)
  balanced16    top-1 free-float-cap stock in each of the 16 largest
                industries (by aggregate free-float cap) -- sector balanced
  random16_s1..s5  uniform random 16-stock samples from the PIT universe
  random48_s1   one 48-stock random sample (pool-size effect)

Universe filter (PIT, mechanical): listed .SH/.SZ, list_date <= 2023-01-01,
name without ST/退, tradable on the snapshot date (has close & circ_mv).

Usage:
    conda run -n Miro python scripts/ashare/build_balanced_pools.py
"""

from __future__ import annotations

import json
import random
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_ROOT = REPO_ROOT.parent
DATA_DIR = REPO_ROOT / "data" / "ashare"
DB_PATH = REPO_ROOT / "data" / "ashare_pools.db"
TUSHARE_API = "http://api.tushare.pro"

SNAPSHOT_DATE = "20240628"  # last trading day before the 2024-07-01 backtest start
START_DATE = "20230101"
END_DATE = "20250731"

LEGACY_POOL = {
    "603259.SH": "药明康德", "002415.SZ": "海康威视", "601899.SH": "紫金矿业",
    "601021.SH": "春秋航空", "300012.SZ": "华测检测", "688005.SH": "容百科技",
    "603345.SH": "安井食品", "601155.SH": "新城控股", "002507.SZ": "涪陵榨菜",
    "002508.SZ": "老板电器", "601233.SH": "桐昆股份", "688169.SH": "石头科技",
    "688981.SH": "中芯国际", "300476.SZ": "胜宏科技", "002371.SZ": "北方华创",
    "600418.SH": "江淮汽车",
}


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
    for attempt in range(5):
        resp = requests.post(
            TUSHARE_API,
            json={
                "api_name": api_name,
                "token": token,
                "params": params,
                "fields": fields,
            },
            timeout=60,
        )
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("code") == 0:
            data = payload["data"]
            return pd.DataFrame(data["items"], columns=data["fields"])
        msg = str(payload.get("msg"))
        if "每分钟" in msg or "频率" in msg or "limit" in msg.lower():
            wait = 20 * (attempt + 1)
            print(f"  rate limited on {api_name}, sleeping {wait}s ...")
            time.sleep(wait)
            continue
        raise RuntimeError(f"tushare {api_name} error: {msg}")
    raise RuntimeError(f"tushare {api_name}: rate limit retries exhausted")


def build_universe(token: str) -> pd.DataFrame:
    basic = tushare_query(
        token,
        "stock_basic",
        {"list_status": "L"},
        fields="ts_code,name,industry,market,list_date",
    )
    snap = tushare_query(
        token,
        "daily_basic",
        {"trade_date": SNAPSHOT_DATE},
        fields="ts_code,close,pe_ttm,pb,turnover_rate,total_mv,circ_mv",
    )
    df = basic.merge(snap, on="ts_code", how="inner")
    df = df[df["ts_code"].str.endswith((".SH", ".SZ"))]
    df = df[df["list_date"] <= "20230101"]
    df = df[~df["name"].str.contains("ST|退", na=False)]
    df = df[(df["close"] > 0) & (df["circ_mv"] > 0)]
    df = df.dropna(subset=["industry"])
    return df.reset_index(drop=True)


def build_pools(universe: pd.DataFrame) -> dict[str, dict]:
    pools: dict[str, dict] = {}

    pools["legacy16"] = {
        "rule": "hand-picked legacy pool v4 (known selection bias, kept for comparison)",
        "codes": list(LEGACY_POOL),
    }

    by_ind = (
        universe.groupby("industry")["circ_mv"].sum().sort_values(ascending=False)
    )
    top_industries = list(by_ind.index[:16])
    balanced: list[str] = []
    for ind in top_industries:
        rows = universe[universe["industry"] == ind].sort_values(
            "circ_mv", ascending=False
        )
        balanced.append(rows.iloc[0]["ts_code"])
    pools["balanced16"] = {
        "rule": (
            f"PIT {SNAPSHOT_DATE}: 16 largest industries by aggregate circ_mv, "
            "top-1 circ_mv stock per industry"
        ),
        "codes": balanced,
    }

    all_codes = sorted(universe["ts_code"].tolist())
    for seed in (1, 2, 3, 4, 5):
        rng = random.Random(seed)
        pools[f"random16_s{seed}"] = {
            "rule": f"PIT {SNAPSHOT_DATE}: uniform random 16 of {len(all_codes)}, seed={seed}",
            "codes": sorted(rng.sample(all_codes, 16)),
        }
    rng = random.Random(1)
    pools["random48_s1"] = {
        "rule": f"PIT {SNAPSHOT_DATE}: uniform random 48 of {len(all_codes)}, seed=1",
        "codes": sorted(rng.sample(all_codes, 48)),
    }
    return pools


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS stock_info (
            ts_code TEXT PRIMARY KEY,
            name TEXT, industry TEXT, market TEXT, list_date TEXT
        );
        CREATE TABLE IF NOT EXISTS daily (
            ts_code TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            open REAL, high REAL, low REAL, close REAL,
            pre_close REAL, pct_chg REAL, vol REAL, amount REAL,
            adj_factor REAL, close_qfq REAL,
            PRIMARY KEY (ts_code, trade_date)
        );
        CREATE TABLE IF NOT EXISTS index_daily (
            ts_code TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            close REAL,
            PRIMARY KEY (ts_code, trade_date)
        );
        CREATE TABLE IF NOT EXISTS trade_cal (
            cal_date TEXT PRIMARY KEY,
            is_open INTEGER
        );
        CREATE TABLE IF NOT EXISTS pools (
            pool_id TEXT NOT NULL,
            ts_code TEXT NOT NULL,
            PRIMARY KEY (pool_id, ts_code)
        );
        CREATE TABLE IF NOT EXISTS pool_meta (
            pool_id TEXT PRIMARY KEY,
            rule TEXT,
            snapshot_date TEXT,
            created_at TEXT
        );
        CREATE TABLE IF NOT EXISTS pit_snapshot (
            ts_code TEXT PRIMARY KEY,
            trade_date TEXT,
            close REAL, pe_ttm REAL, pb REAL,
            turnover_rate REAL, total_mv REAL, circ_mv REAL
        );
        """
    )
    conn.commit()


def existing_daily_codes(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT DISTINCT ts_code FROM daily").fetchall()
    return {r[0] for r in rows}


def import_legacy_csvs(conn: sqlite3.Connection) -> None:
    for csv in sorted(DATA_DIR.glob("daily_*.csv")):
        if csv.name.startswith("daily_basic_"):
            continue
        code = csv.stem.replace("daily_", "")
        df = pd.read_csv(csv, dtype={"trade_date": str})
        if "close_qfq" not in df.columns:
            df["close_qfq"] = df["close"]
        if "adj_factor" not in df.columns:
            df["adj_factor"] = 1.0
        cols = [
            "ts_code", "trade_date", "open", "high", "low", "close",
            "pre_close", "pct_chg", "vol", "amount", "adj_factor", "close_qfq",
        ]
        for c in cols:
            if c not in df.columns:
                df[c] = None
        df["ts_code"] = code
        df[cols].to_sql("tmp_daily", conn, if_exists="replace", index=False)
        conn.execute(
            "INSERT OR REPLACE INTO daily SELECT * FROM tmp_daily"
        )
    idx = pd.read_csv(DATA_DIR / "index_000300.SH.csv", dtype={"trade_date": str})
    idx["ts_code"] = "000300.SH"
    idx[["ts_code", "trade_date", "close"]].to_sql(
        "tmp_idx", conn, if_exists="replace", index=False
    )
    conn.execute("INSERT OR REPLACE INTO index_daily SELECT * FROM tmp_idx")
    cal = pd.read_csv(DATA_DIR / "trade_cal.csv", dtype={"cal_date": str})
    cal[["cal_date", "is_open"]].to_sql(
        "tmp_cal", conn, if_exists="replace", index=False
    )
    conn.execute("INSERT OR REPLACE INTO trade_cal SELECT * FROM tmp_cal")
    conn.execute("DROP TABLE IF EXISTS tmp_daily")
    conn.execute("DROP TABLE IF EXISTS tmp_idx")
    conn.execute("DROP TABLE IF EXISTS tmp_cal")
    conn.commit()


def fetch_daily_into_db(
    conn: sqlite3.Connection, token: str, codes: list[str]
) -> None:
    have = existing_daily_codes(conn)
    todo = [c for c in codes if c not in have]
    print(f"fetching daily bars for {len(todo)} new codes "
          f"({len(codes) - len(todo)} already cached)")
    for i, ts_code in enumerate(todo, 1):
        daily = tushare_query(
            token,
            "daily",
            {"ts_code": ts_code, "start_date": START_DATE, "end_date": END_DATE},
            fields="ts_code,trade_date,open,high,low,close,pre_close,pct_chg,vol,amount",
        )
        daily = daily.sort_values("trade_date").reset_index(drop=True)
        try:
            adj = tushare_query(
                token,
                "adj_factor",
                {"ts_code": ts_code, "start_date": START_DATE, "end_date": END_DATE},
                fields="trade_date,adj_factor",
            )
            adj = adj.sort_values("trade_date").reset_index(drop=True)
            daily = daily.merge(adj, on="trade_date", how="left")
            daily["adj_factor"] = daily["adj_factor"].ffill()
            latest = daily["adj_factor"].iloc[-1]
            daily["close_qfq"] = daily["close"] * daily["adj_factor"] / latest
        except Exception as exc:
            print(f"  WARN adj_factor {ts_code}: {exc} -> raw close as qfq")
            daily["adj_factor"] = 1.0
            daily["close_qfq"] = daily["close"]
        cols = [
            "ts_code", "trade_date", "open", "high", "low", "close",
            "pre_close", "pct_chg", "vol", "amount", "adj_factor", "close_qfq",
        ]
        daily[cols].to_sql("tmp_fetch", conn, if_exists="replace", index=False)
        conn.execute("INSERT OR REPLACE INTO daily SELECT * FROM tmp_fetch")
        conn.commit()
        if i % 10 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)} done (latest {ts_code}, {len(daily)} rows)")
        time.sleep(0.15)
    conn.execute("DROP TABLE IF EXISTS tmp_fetch")
    conn.commit()


def main() -> None:
    token = load_token()
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    init_db(conn)

    print("building PIT universe ...")
    universe = build_universe(token)
    print(f"universe: {len(universe)} stocks as of {SNAPSHOT_DATE}")

    universe.rename(columns={"close": "snap_close"}, inplace=True)
    universe[["ts_code", "name", "industry", "market", "list_date"]].to_sql(
        "tmp_info", conn, if_exists="replace", index=False
    )
    conn.execute("INSERT OR REPLACE INTO stock_info SELECT * FROM tmp_info")
    snap = universe[
        ["ts_code", "snap_close", "pe_ttm", "pb", "turnover_rate", "total_mv", "circ_mv"]
    ].copy()
    snap.insert(1, "trade_date", SNAPSHOT_DATE)
    snap.columns = [
        "ts_code", "trade_date", "close", "pe_ttm", "pb",
        "turnover_rate", "total_mv", "circ_mv",
    ]
    snap.to_sql("tmp_snap", conn, if_exists="replace", index=False)
    conn.execute("INSERT OR REPLACE INTO pit_snapshot SELECT * FROM tmp_snap")
    conn.execute("DROP TABLE IF EXISTS tmp_info")
    conn.execute("DROP TABLE IF EXISTS tmp_snap")
    conn.commit()

    pools = build_pools(universe)
    now = datetime.now().isoformat(timespec="seconds")
    for pool_id, spec in pools.items():
        conn.execute(
            "INSERT OR REPLACE INTO pool_meta VALUES (?,?,?,?)",
            (pool_id, spec["rule"], SNAPSHOT_DATE, now),
        )
        conn.execute("DELETE FROM pools WHERE pool_id = ?", (pool_id,))
        conn.executemany(
            "INSERT INTO pools VALUES (?,?)",
            [(pool_id, c) for c in spec["codes"]],
        )
        print(f"pool {pool_id}: {len(spec['codes'])} codes | {spec['rule']}")
    conn.commit()

    print("importing legacy CSV cache into sqlite ...")
    import_legacy_csvs(conn)

    all_codes = sorted({c for spec in pools.values() for c in spec["codes"]})
    fetch_daily_into_db(conn, token, all_codes)

    n_daily = conn.execute("SELECT COUNT(*) FROM daily").fetchone()[0]
    n_codes = conn.execute("SELECT COUNT(DISTINCT ts_code) FROM daily").fetchone()[0]
    print(f"\nDone. {DB_PATH}: {n_codes} stocks, {n_daily} daily rows")
    names = pd.read_sql_query(
        "SELECT p.pool_id, p.ts_code, i.name, i.industry FROM pools p "
        "LEFT JOIN stock_info i USING(ts_code) "
        "WHERE p.pool_id IN ('balanced16','random16_s1') ORDER BY p.pool_id, p.ts_code",
        conn,
    )
    print("\nsample pools:")
    print(names.to_string(index=False))
    conn.close()


if __name__ == "__main__":
    main()
