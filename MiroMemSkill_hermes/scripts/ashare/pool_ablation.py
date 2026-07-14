# SPDX-FileCopyrightText: 2026 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

"""Momentum-top4 ablation across PIT-constructed pools stored in SQLite.

Replays the exact 12 monthly windows of the trader benchmark on every pool in
data/ashare_pools.db and compares: momentum top4 (rel20 vs CSI300), equal
weight, oracle top4 (hindsight upper bound), and the index itself.  Same fee
model as eval_trader (buy 0.05%, sell 0.15%, min 5 CNY per leg), sequential
compounding from 1,000,000.

Purpose: test whether the legacy pool's engineered winner/loser balance is
what makes momentum look good, or momentum survives on balanced/random pools.

Usage:
    conda run -n Miro python scripts/ashare/pool_ablation.py
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = REPO_ROOT / "data" / "ashare_pools.db"

# Exact benchmark windows (entry close -> exit close, 20 trading days).
WINDOWS = [
    ("20240701", "20240729"), ("20240801", "20240829"),
    ("20240902", "20241009"), ("20241009", "20241106"),
    ("20241106", "20241204"), ("20241204", "20250102"),
    ("20250102", "20250207"), ("20250207", "20250307"),
    ("20250307", "20250407"), ("20250407", "20250508"),
    ("20250508", "20250606"), ("20250606", "20250704"),
]
MOMENTUM_WINDOW = 20
TOP_K = 4
INITIAL_CAPITAL = 1_000_000.0
BUY_FEE, SELL_FEE, MIN_FEE = 0.0005, 0.0015, 5.0


def load_prices(conn: sqlite3.Connection) -> dict[str, dict[str, float]]:
    prices: dict[str, dict[str, float]] = {}
    for code, date, qfq in conn.execute(
        "SELECT ts_code, trade_date, close_qfq FROM daily WHERE close_qfq > 0"
    ):
        prices.setdefault(code, {})[date] = qfq
    return prices


def load_index(conn: sqlite3.Connection) -> dict[str, float]:
    return {
        d: c
        for d, c in conn.execute(
            "SELECT trade_date, close FROM index_daily WHERE ts_code='000300.SH'"
        )
    }


def load_calendar(conn: sqlite3.Connection) -> list[str]:
    return [
        d
        for (d,) in conn.execute(
            "SELECT cal_date FROM trade_cal WHERE is_open=1 ORDER BY cal_date"
        )
    ]


def price_on_or_before(series: dict[str, float], date: str, cal: list[str]) -> float | None:
    if date in series:
        return series[date]
    # walk back over recent sessions (suspension tolerance: 15 sessions)
    idx = None
    lo, hi = 0, len(cal) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if cal[mid] <= date:
            idx = mid
            lo = mid + 1
        else:
            hi = mid - 1
    if idx is None:
        return None
    for back in range(1, 16):
        j = idx - back
        if j < 0:
            return None
        px = series.get(cal[j])
        if px is not None:
            return px
    return None


def window_return(
    series: dict[str, float], entry: str, exit_: str, cal: list[str]
) -> float | None:
    p0 = price_on_or_before(series, entry, cal)
    p1 = price_on_or_before(series, exit_, cal)
    if not p0 or not p1:
        return None
    return p1 / p0 - 1.0


def rel20(
    series: dict[str, float], index: dict[str, float], as_of: str, cal: list[str]
) -> float | None:
    # start date = MOMENTUM_WINDOW sessions before as_of
    dates = [d for d in cal if d <= as_of]
    if len(dates) < MOMENTUM_WINDOW + 1:
        return None
    start = dates[-(MOMENTUM_WINDOW + 1)]
    r_stock = window_return(series, start, as_of, cal)
    r_index = window_return(index, start, as_of, cal)
    if r_stock is None or r_index is None:
        return None
    return r_stock - r_index


def run_arm(
    pool: list[str],
    prices: dict[str, dict[str, float]],
    index: dict[str, float],
    cal: list[str],
    picker,
) -> tuple[float, list[float]]:
    capital = INITIAL_CAPITAL
    monthly: list[float] = []
    for entry, exit_ in WINDOWS:
        weights = picker(pool, entry, exit_)
        start_cap = capital
        invested_value = 0.0
        fees = 0.0
        for code, w in weights.items():
            if w <= 0:
                continue
            amt = start_cap * w
            buy_fee = max(MIN_FEE, amt * BUY_FEE)
            r = window_return(prices[code], entry, exit_, cal)
            if r is None:
                r = 0.0
            end_val = (amt - buy_fee) * (1.0 + r)
            sell_fee = max(MIN_FEE, end_val * SELL_FEE)
            invested_value += end_val - sell_fee
            fees += buy_fee + sell_fee
        cash = start_cap * max(0.0, 1.0 - sum(weights.values()))
        capital = invested_value + cash
        monthly.append(capital / start_cap - 1.0)
    return capital / INITIAL_CAPITAL - 1.0, monthly


def main() -> None:
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.execute("PRAGMA busy_timeout=10000")
    cal = load_calendar(conn)
    index = load_index(conn)
    prices = load_prices(conn)
    pools = {}
    for pool_id, code in conn.execute("SELECT pool_id, ts_code FROM pools"):
        pools.setdefault(pool_id, []).append(code)
    names = dict(conn.execute("SELECT ts_code, name FROM stock_info"))
    conn.close()

    def momentum_picker(pool, entry, exit_):
        scored = []
        for code in pool:
            s = rel20(prices.get(code, {}), index, entry, cal)
            if s is not None:
                scored.append((code, s))
        scored.sort(key=lambda x: (-x[1], x[0]))
        top = [c for c, _ in scored[:TOP_K]]
        return {c: 1.0 / TOP_K for c in top}

    def equal_picker(pool, entry, exit_):
        live = [c for c in pool if price_on_or_before(prices.get(c, {}), entry, cal)]
        return {c: 1.0 / len(live) for c in live}

    def oracle_picker(pool, entry, exit_):
        scored = []
        for code in pool:
            r = window_return(prices.get(code, {}), entry, exit_, cal)
            if r is not None:
                scored.append((code, r))
        scored.sort(key=lambda x: (-x[1], x[0]))
        top = [c for c, _ in scored[:TOP_K]]
        return {c: 1.0 / TOP_K for c in top}

    r_index = 1.0
    for entry, exit_ in WINDOWS:
        r = window_return(index, entry, exit_, cal)
        r_index *= 1.0 + (r or 0.0)
    print(f"CSI300 over the 12 windows: {(r_index - 1) * 100:+.2f}%\n")

    header = (
        f"{'pool':14s} {'momentum4':>10s} {'equal_w':>10s} {'oracle4':>10s} "
        f"{'mom-eq':>8s} {'mom-index':>10s}"
    )
    print(header)
    print("-" * len(header))
    ordered = ["legacy16", "balanced16"] + sorted(
        p for p in pools if p.startswith("random16")
    ) + ["random48_s1"]
    rand_edges = []
    for pool_id in ordered:
        pool = pools[pool_id]
        mom, mom_m = run_arm(pool, prices, index, cal, momentum_picker)
        eq, _ = run_arm(pool, prices, index, cal, equal_picker)
        orc, _ = run_arm(pool, prices, index, cal, oracle_picker)
        edge = mom - eq
        if pool_id.startswith("random16"):
            rand_edges.append(edge)
        print(
            f"{pool_id:14s} {mom * 100:+9.2f}% {eq * 100:+9.2f}% "
            f"{orc * 100:+9.2f}% {edge * 100:+7.2f}% {(mom - (r_index - 1)) * 100:+9.2f}%"
        )
    if rand_edges:
        mean_edge = sum(rand_edges) / len(rand_edges)
        print(
            f"\nrandom16 momentum-minus-equalweight edge: "
            f"mean {mean_edge * 100:+.2f}%, "
            f"range [{min(rand_edges) * 100:+.2f}%, {max(rand_edges) * 100:+.2f}%]"
        )

    # show what momentum held in the balanced pool (context for the report)
    print("\nbalanced16 momentum holdings by month:")
    for entry, exit_ in WINDOWS:
        picks = momentum_picker(pools["balanced16"], entry, exit_)
        label = " ".join(names.get(c, c) for c in picks)
        print(f"  {entry}: {label}")


if __name__ == "__main__":
    main()
