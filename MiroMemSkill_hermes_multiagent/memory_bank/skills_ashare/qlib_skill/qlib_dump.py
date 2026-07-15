# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

"""Minimal Tushare-CSV -> qlib bin converter.

The pip build of pyqlib does not ship the repo's scripts/dump_bin.py, so this
module implements the small subset of the qlib local-storage layout we need:

    <provider_uri>/
        calendars/day.txt              one trading date (YYYY-MM-DD) per line
        instruments/all.txt            CODE<TAB>start<TAB>end per stock
        features/<code_lower>/<field>.day.bin
            float32 little-endian: [start_calendar_index, v0, v1, ...]
            values are aligned to consecutive calendar slots; gaps are NaN.

Input is this repo's A-share cache (scripts/ashare/fetch_data.py output):
    trade_cal.csv                      cal_date,is_open
    daily_<ts_code>.csv                raw OHLC + qfq columns + adj_factor
    index_000300.SH.csv                raw index OHLC

Conventions: prices are forward-adjusted (qfq) with factor=1.0 written
alongside; `600519.SH` maps to `SH600519`; the CSI300 index becomes
`SH000300` with feature bins only (kept OUT of instruments/all.txt so the
trading universe stays stocks-only).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

FIELDS = ("open", "high", "low", "close", "vwap", "volume", "amount", "factor")


def to_qlib_code(ts_code: str) -> str:
    """600519.SH -> SH600519 (qlib cn convention)."""
    num, exch = ts_code.strip().upper().split(".")
    return f"{exch}{num}"


def _iso(date: str | int) -> str:
    s = str(date)
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}"


def load_calendar(csv_cache_dir: Path) -> list[str]:
    cal = pd.read_csv(csv_cache_dir / "trade_cal.csv", dtype={"cal_date": str})
    days = cal[cal["is_open"].astype(int) == 1]["cal_date"].sort_values()
    return [_iso(d) for d in days]


def _stock_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Field frame (indexed by ISO date) for a stock daily_*.csv."""
    out = pd.DataFrame(index=[_iso(d) for d in df["trade_date"]])
    for field in ("open", "high", "low", "close"):
        out[field] = df[f"{field}_qfq"].to_numpy()
    # Tushare units: amount in k CNY, vol in lots(100 shares):
    # raw vwap = amount*1000 / (vol*100); adjust with the qfq ratio.
    vol = df["vol"].to_numpy(dtype=float)
    amount = df["amount"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        vwap_raw = np.where(vol > 0, amount * 10.0 / vol, np.nan)
        adj_ratio = df["close_qfq"].to_numpy() / df["close"].to_numpy()
    out["vwap"] = vwap_raw * adj_ratio
    out["volume"] = vol
    out["amount"] = amount
    out["factor"] = 1.0  # prices already adjusted
    return out


def _index_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Field frame for the benchmark index CSV (raw prices, no adjustment)."""
    out = pd.DataFrame(index=[_iso(d) for d in df["trade_date"]])
    for field in ("open", "high", "low", "close"):
        out[field] = df[field].to_numpy()
    vol = df["vol"].to_numpy(dtype=float)
    amount = df["amount"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["vwap"] = np.where(vol > 0, amount * 10.0 / vol, np.nan)
    out["volume"] = vol
    out["amount"] = amount
    out["factor"] = 1.0
    return out


def _write_bins(
    features_dir: Path, code: str, frame: pd.DataFrame, calendar: list[str]
) -> tuple[str, str]:
    """Align frame to the calendar and write one .bin per field.

    Returns (start_date, end_date) of the instrument's data span.
    """
    cal_pos = {d: i for i, d in enumerate(calendar)}
    dates = [d for d in frame.index if d in cal_pos]
    if not dates:
        raise ValueError(f"{code}: no dates overlap the calendar")
    frame = frame.loc[dates]
    start_i, end_i = cal_pos[dates[0]], cal_pos[dates[-1]]
    # Reindex over the full consecutive calendar slice; suspended days -> NaN.
    span = calendar[start_i : end_i + 1]
    frame = frame.reindex(span)

    inst_dir = features_dir / code.lower()
    inst_dir.mkdir(parents=True, exist_ok=True)
    for field in FIELDS:
        values = frame[field].to_numpy(dtype="<f4")
        payload = np.hstack([np.array([start_i], dtype="<f4"), values])
        payload.tofile(inst_dir / f"{field}.day.bin")
    return dates[0], dates[-1]


def convert(
    csv_cache_dir: str | Path,
    provider_uri: str | Path,
    index_csv: str = "index_000300.SH.csv",
    index_code: str = "SH000300",
) -> dict:
    """Convert the CSV cache into a qlib provider directory."""
    src = Path(csv_cache_dir)
    dst = Path(provider_uri)
    if not (src / "trade_cal.csv").exists():
        raise FileNotFoundError(f"trade_cal.csv not found in {src}")

    calendar = load_calendar(src)
    (dst / "calendars").mkdir(parents=True, exist_ok=True)
    (dst / "calendars" / "day.txt").write_text("\n".join(calendar) + "\n")

    features_dir = dst / "features"
    instruments: list[str] = []
    # Strictly daily_<code>.csv — the cache also holds daily_basic_*.csv
    # (valuation) files which are not bar data. The cache may also keep CSVs
    # of retired pool members; meta.json's stock_pool is the source of truth
    # for the current universe, so ranks/cross-sections match the benchmark.
    stock_re = re.compile(r"^daily_(\d{6}\.(?:SH|SZ|BJ))$")
    pool: set[str] | None = None
    meta_path = src / "meta.json"
    if meta_path.exists():
        import json as _json

        pool = set(_json.loads(meta_path.read_text(encoding="utf-8"))["stock_pool"])
    stocks = sorted(p for p in src.glob("daily_*.csv") if stock_re.match(p.stem))
    for path in stocks:
        ts_code = stock_re.match(path.stem).group(1)
        if pool is not None and ts_code not in pool:
            continue
        code = to_qlib_code(ts_code)
        df = pd.read_csv(path, dtype={"trade_date": str})
        start, end = _write_bins(features_dir, code, _stock_frame(df), calendar)
        instruments.append(f"{code}\t{start}\t{end}")

    (dst / "instruments").mkdir(parents=True, exist_ok=True)
    (dst / "instruments" / "all.txt").write_text("\n".join(instruments) + "\n")

    # Benchmark index: feature bins only, NOT part of the trading universe.
    index_path = src / index_csv
    benchmark = None
    if index_path.exists():
        df = pd.read_csv(index_path, dtype={"trade_date": str})
        _write_bins(features_dir, index_code, _index_frame(df), calendar)
        benchmark = index_code

    return {
        "provider_uri": str(dst),
        "calendar_days": len(calendar),
        "instruments": len(instruments),
        "benchmark": benchmark,
    }


def read_bin(provider_uri: str | Path, code: str, field: str) -> tuple[int, np.ndarray]:
    """Read one .bin back (for tests/verification): (start_index, values)."""
    path = Path(provider_uri) / "features" / code.lower() / f"{field}.day.bin"
    raw = np.fromfile(path, dtype="<f4")
    return int(raw[0]), raw[1:]


def iter_instruments(provider_uri: str | Path) -> Iterable[tuple[str, str, str]]:
    for line in (Path(provider_uri) / "instruments" / "all.txt").read_text().splitlines():
        if line.strip():
            code, start, end = line.split("\t")
            yield code, start, end
