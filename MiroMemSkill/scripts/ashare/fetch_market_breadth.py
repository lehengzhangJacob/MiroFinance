#!/usr/bin/env python3
"""Build a resumable, point-in-time full-A-share breadth cache from Tushare."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests


REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_ROOT = REPO_ROOT.parent
DATA_DIR = REPO_ROOT / "data" / "ashare"
TUSHARE_API = "http://api.tushare.pro"
START_DATE = os.getenv("ASHARE_BREADTH_START_DATE", "20230101")
END_DATE = os.getenv("ASHARE_BREADTH_END_DATE", "20250731")
REQUEST_INTERVAL = float(os.getenv("ASHARE_BREADTH_REQUEST_INTERVAL", "0.35"))
OUTPUT_FILE = DATA_DIR / "market_breadth_daily.csv"
MANIFEST_FILE = DATA_DIR / "market_breadth_manifest.json"
CHECKPOINT_FILE = DATA_DIR / ".market_breadth_checkpoint.json.gz"


def load_token() -> str:
    token_file = AGENT_ROOT / "tushare_token"
    if not token_file.exists():
        sys.exit(f"tushare token file not found: {token_file}")
    raw = token_file.read_text(encoding="utf-8").strip()
    return raw.split("=", 1)[-1].strip().strip('"').strip("'")


def tushare_query(
    token: str,
    api_name: str,
    params: dict[str, Any],
    fields: str,
) -> pd.DataFrame:
    response = requests.post(
        TUSHARE_API,
        json={
            "api_name": api_name,
            "token": token,
            "params": params,
            "fields": fields,
        },
        timeout=90,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("code") != 0:
        raise RuntimeError(f"tushare {api_name} error: {payload.get('msg')}")
    data = payload["data"]
    return pd.DataFrame(data["items"], columns=data["fields"])


def _is_a_share(ts_code: str) -> bool:
    code = str(ts_code).upper()
    if code.endswith(".BJ"):
        return True
    if code.endswith(".SZ"):
        return not code.startswith("200")
    if code.endswith(".SH"):
        return not code.startswith("900")
    return False


def _initial_state() -> dict[str, Any]:
    return {
        "version": 1,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "completed_through": "",
        "synthetic_prices": {},
        "histories": {},
        "adv_ratio_history": [],
    }


def _load_resume_state() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not CHECKPOINT_FILE.exists() or not OUTPUT_FILE.exists():
        return _initial_state(), []
    try:
        with gzip.open(CHECKPOINT_FILE, "rt", encoding="utf-8") as handle:
            state = json.load(handle)
        rows = pd.read_csv(OUTPUT_FILE, dtype={"trade_date": str}).to_dict("records")
        last_output = str(rows[-1]["trade_date"]) if rows else ""
        valid = (
            state.get("version") == 1
            and state.get("start_date") == START_DATE
            and state.get("end_date") == END_DATE
            and state.get("completed_through") == last_output
        )
        if valid:
            return state, rows
    except Exception as exc:
        print(f"WARN invalid breadth checkpoint ({exc}); rebuilding from scratch")
    return _initial_state(), []


def _atomic_write_json_gzip(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
    temporary.replace(path)


def _persist(state: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    temporary = OUTPUT_FILE.with_suffix(".csv.tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    temporary.replace(OUTPUT_FILE)
    _atomic_write_json_gzip(CHECKPOINT_FILE, state)


def _trade_dates(token: str) -> list[str]:
    calendar_file = DATA_DIR / "trade_cal.csv"
    if calendar_file.exists():
        calendar = pd.read_csv(calendar_file, dtype={"cal_date": str})
    else:
        calendar = tushare_query(
            token,
            "trade_cal",
            {"exchange": "SSE", "start_date": START_DATE, "end_date": END_DATE},
            "cal_date,is_open",
        )
    calendar["cal_date"] = calendar["cal_date"].astype(str)
    return (
        calendar[
            (calendar["is_open"].astype(int) == 1)
            & (calendar["cal_date"] >= START_DATE)
            & (calendar["cal_date"] <= END_DATE)
        ]["cal_date"]
        .sort_values()
        .tolist()
    )


def _derive_day(
    daily: pd.DataFrame,
    *,
    state: dict[str, Any],
    trade_date: str,
) -> dict[str, Any]:
    daily = daily[daily["ts_code"].map(_is_a_share)].copy()
    daily["pct_chg"] = pd.to_numeric(daily["pct_chg"], errors="coerce")
    daily = daily[
        daily["pct_chg"].notna()
        & (daily["pct_chg"] > -100.0)
        & (daily["pct_chg"] < 1000.0)
    ]
    if daily.empty:
        raise RuntimeError(f"no valid A-share rows for {trade_date}")
    if len(daily) >= 6000:
        raise RuntimeError(
            f"{trade_date} returned {len(daily)} rows; possible API truncation"
        )

    prices: dict[str, float] = state["synthetic_prices"]
    histories: dict[str, list[float]] = state["histories"]
    above20 = above60 = positive20 = 0
    eligible20 = eligible60 = eligible_ret20 = 0

    for record in daily[["ts_code", "pct_chg"]].to_dict("records"):
        code = str(record["ts_code"]).upper()
        pct_change = float(record["pct_chg"]) / 100.0
        previous = float(prices.get(code, 1.0))
        current = previous * (1.0 + pct_change) if code in prices else 1.0
        prices[code] = current
        history = deque(
            (float(value) for value in histories.get(code, [])),
            maxlen=61,
        )
        history.append(current)
        values = list(history)
        histories[code] = values

        if len(values) >= 20:
            eligible20 += 1
            above20 += current >= sum(values[-20:]) / 20.0
        if len(values) >= 60:
            eligible60 += 1
            above60 += current >= sum(values[-60:]) / 60.0
        if len(values) >= 21 and values[-21] != 0.0:
            eligible_ret20 += 1
            positive20 += current / values[-21] - 1.0 > 0.0

    advance = int((daily["pct_chg"] > 0.0).sum())
    decline = int((daily["pct_chg"] < 0.0).sum())
    flat = int((daily["pct_chg"] == 0.0).sum())
    universe = advance + decline + flat
    adv_ratio_1d = advance / universe
    adv_history = deque(
        (float(value) for value in state.get("adv_ratio_history", [])),
        maxlen=5,
    )
    adv_history.append(adv_ratio_1d)
    state["adv_ratio_history"] = list(adv_history)

    return {
        "trade_date": trade_date,
        "universe_count": universe,
        "advance_count": advance,
        "decline_count": decline,
        "flat_count": flat,
        "adv_ratio_1d": round(adv_ratio_1d, 8),
        "adv_ratio_5d": round(sum(adv_history) / len(adv_history), 8),
        "ma20_eligible": eligible20,
        "above_ma20_count": int(above20),
        "above_ma20_ratio": round(above20 / eligible20, 8) if eligible20 else 0.0,
        "ma60_eligible": eligible60,
        "above_ma60_count": int(above60),
        "above_ma60_ratio": round(above60 / eligible60, 8) if eligible60 else 0.0,
        "ret20_eligible": eligible_ret20,
        "positive_ret20_count": int(positive20),
        "positive_ret20_ratio": (
            round(positive20 / eligible_ret20, 8) if eligible_ret20 else 0.0
        ),
    }


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    token = load_token()
    state, rows = _load_resume_state()
    completed = str(state.get("completed_through", ""))
    dates = [date for date in _trade_dates(token) if not completed or date > completed]
    print(
        f"breadth range {START_DATE}..{END_DATE}; "
        f"resuming after {completed or 'start'}; remaining={len(dates)}"
    )

    try:
        for number, trade_date in enumerate(dates, 1):
            daily = tushare_query(
                token,
                "daily",
                {"trade_date": trade_date},
                "ts_code,trade_date,close,pre_close,pct_chg",
            )
            rows.append(_derive_day(daily, state=state, trade_date=trade_date))
            state["completed_through"] = trade_date
            if number % 5 == 0 or number == len(dates):
                _persist(state, rows)
                print(
                    f"[{number}/{len(dates)}] {trade_date}: "
                    f"universe={rows[-1]['universe_count']}"
                )
            time.sleep(REQUEST_INTERVAL)
    except BaseException:
        _persist(state, rows)
        raise

    digest = hashlib.sha256(OUTPUT_FILE.read_bytes()).hexdigest()
    manifest = {
        "version": 1,
        "source": "Tushare Pro daily queried by historical trade_date",
        "api": "daily",
        "fields": ["ts_code", "trade_date", "close", "pre_close", "pct_chg"],
        "start_date": START_DATE,
        "end_date": END_DATE,
        "row_count": len(rows),
        "last_trade_date": rows[-1]["trade_date"] if rows else "",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "sha256": digest,
        "method": (
            "Per-date traded A shares; synthetic prices compounded from pct_chg; "
            "MA/return breadth uses only observations available through each date."
        ),
    }
    MANIFEST_FILE.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    # The checkpoint is only an interruption-recovery artifact.  A completed
    # cache keeps the derived daily series and its source manifest only.
    CHECKPOINT_FILE.unlink(missing_ok=True)
    print(f"wrote {len(rows)} rows -> {OUTPUT_FILE}")
    print(f"sha256={digest}")


if __name__ == "__main__":
    main()
