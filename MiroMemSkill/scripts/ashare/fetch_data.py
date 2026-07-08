# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

"""Fetch A-share daily data from Tushare into local CSV cache.

Zero extra dependencies: calls the Tushare Pro HTTP API directly with
`requests` and stores plain CSV under MiroFlow/data/ashare/.

Usage:
    uv run python scripts/ashare/fetch_data.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[2]  # MiroFlow/
AGENT_ROOT = REPO_ROOT.parent  # agent/
DATA_DIR = REPO_ROOT / "data" / "ashare"
TUSHARE_API = "http://api.tushare.pro"

START_DATE = "20230101"
END_DATE = "20250731"

# Pool v2 (2026-07-08). Ex-ante rule, formation date 2024-06-28 (before the
# backtest window) to avoid survivorship/selection bias toward index winners:
#   - 3 prior losers + 3 prior winners by 2024H1 excess return vs CSI300;
#   - 2 names NOT in CSI300 (300012.SZ, 688169.SH); 6 distinct industries;
#   - main board / ChiNext / STAR mix; no overlap with pool v1 mega-caps.
# Pool v1 (茅台/宁德/招行/比亚迪/长电/海控) archived in meta_pool1.json.
STOCK_POOL: dict[str, dict[str, str]] = {
    "603259.SH": {"name": "药明康德", "industry": "医药外包"},
    "000002.SZ": {"name": "万科A", "industry": "房地产"},
    "300012.SZ": {"name": "华测检测", "industry": "检测服务"},
    "601899.SH": {"name": "紫金矿业", "industry": "有色金属"},
    "601138.SH": {"name": "工业富联", "industry": "AI服务器"},
    "688169.SH": {"name": "石头科技", "industry": "智能清洁电器"},
}
INDEX_CODE = "000300.SH"  # CSI 300


def load_token() -> str:
    token_file = AGENT_ROOT / "tushare_token"
    if not token_file.exists():
        sys.exit(f"tushare token file not found: {token_file}")
    raw = token_file.read_text(encoding="utf-8").strip()
    # File may be `TUSHARE_TOKEN=xxx` or the bare token.
    if "=" in raw:
        raw = raw.split("=", 1)[1].strip().strip('"').strip("'")
    return raw


def tushare_query(token: str, api_name: str, params: dict, fields: str = "") -> pd.DataFrame:
    resp = requests.post(
        TUSHARE_API,
        json={"api_name": api_name, "token": token, "params": params, "fields": fields},
        timeout=60,
    )
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("code") != 0:
        raise RuntimeError(f"tushare {api_name} error: {payload.get('msg')}")
    data = payload["data"]
    return pd.DataFrame(data["items"], columns=data["fields"])


def fetch_required(token: str) -> None:
    """Trade calendar, index daily, per-stock daily bars (must succeed)."""
    cal = tushare_query(
        token,
        "trade_cal",
        {"exchange": "SSE", "start_date": START_DATE, "end_date": END_DATE},
        fields="cal_date,is_open",
    )
    cal = cal.sort_values("cal_date").reset_index(drop=True)
    cal.to_csv(DATA_DIR / "trade_cal.csv", index=False)
    print(f"trade_cal: {len(cal)} rows, {int(cal['is_open'].astype(int).sum())} trading days")

    idx = tushare_query(
        token,
        "index_daily",
        {"ts_code": INDEX_CODE, "start_date": START_DATE, "end_date": END_DATE},
        fields="ts_code,trade_date,open,high,low,close,vol,amount",
    )
    idx = idx.sort_values("trade_date").reset_index(drop=True)
    idx.to_csv(DATA_DIR / f"index_{INDEX_CODE}.csv", index=False)
    print(f"index {INDEX_CODE}: {len(idx)} rows")

    for ts_code in STOCK_POOL:
        daily = tushare_query(
            token,
            "daily",
            {"ts_code": ts_code, "start_date": START_DATE, "end_date": END_DATE},
            fields="ts_code,trade_date,open,high,low,close,pre_close,pct_chg,vol,amount",
        )
        daily = daily.sort_values("trade_date").reset_index(drop=True)

        # Forward-adjusted (qfq) prices so that return windows spanning
        # dividends/splits stay correct: price * factor / latest_factor.
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
            for col in ("open", "high", "low", "close"):
                daily[f"{col}_qfq"] = daily[col] * daily["adj_factor"] / latest
        except Exception as exc:  # degraded but usable (no dividends adjustment)
            print(f"  WARN adj_factor {ts_code}: {exc} -> using raw prices as qfq")
            for col in ("open", "high", "low", "close"):
                daily[f"{col}_qfq"] = daily[col]

        daily.to_csv(DATA_DIR / f"daily_{ts_code}.csv", index=False)
        print(f"daily {ts_code}: {len(daily)} rows")
        time.sleep(0.3)  # stay well under rate limits


def fetch_optional(token: str) -> dict[str, bool]:
    """Valuation and financials (skip gracefully if token lacks points)."""
    ok = {"daily_basic": True, "fina_indicator": True}

    for ts_code in STOCK_POOL:
        try:
            basic = tushare_query(
                token,
                "daily_basic",
                {"ts_code": ts_code, "start_date": START_DATE, "end_date": END_DATE},
                fields="trade_date,pe_ttm,pb,ps_ttm,turnover_rate,total_mv,circ_mv",
            )
            basic = basic.sort_values("trade_date").reset_index(drop=True)
            basic.to_csv(DATA_DIR / f"daily_basic_{ts_code}.csv", index=False)
            print(f"daily_basic {ts_code}: {len(basic)} rows")
        except Exception as exc:
            print(f"  WARN daily_basic {ts_code}: {exc}")
            ok["daily_basic"] = False
        time.sleep(0.3)

    for ts_code in STOCK_POOL:
        try:
            fin = tushare_query(
                token,
                "fina_indicator",
                {"ts_code": ts_code, "start_date": "20220101", "end_date": END_DATE},
                fields="ts_code,ann_date,end_date,eps,roe,grossprofit_margin,netprofit_margin,or_yoy,netprofit_yoy",
            )
            fin = fin.sort_values("ann_date").reset_index(drop=True)
            fin.to_csv(DATA_DIR / f"financials_{ts_code}.csv", index=False)
            print(f"financials {ts_code}: {len(fin)} rows")
        except Exception as exc:
            print(f"  WARN fina_indicator {ts_code}: {exc}")
            ok["fina_indicator"] = False
        time.sleep(0.3)

    return ok


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    token = load_token()

    fetch_required(token)
    optional_ok = fetch_optional(token)

    meta = {
        "fetched_at": datetime.now().isoformat(timespec="seconds"),
        "start_date": START_DATE,
        "end_date": END_DATE,
        "index_code": INDEX_CODE,
        "stock_pool": STOCK_POOL,
        "optional_datasets": optional_ok,
    }
    (DATA_DIR / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nDone. Cache dir: {DATA_DIR}")


if __name__ == "__main__":
    main()
