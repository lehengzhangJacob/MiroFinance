# SPDX-FileCopyrightText: 2026 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

"""Evaluate open-universe trader runs against whole-market baselines.

Reads task_*_attempt_1.json files from one or more run directories, parses
each month's boxed allocation, and replays the 12 windows with the same fee
model as the legacy trader eval (buy 0.05%, sell 0.15%, min 5 CNY/leg,
sequential compounding from 1,000,000).

Window returns come from Tushare pct_chg compounding in the SQLite mirror
(dividend/split-safe).  A stock suspended during the window is implicitly
sold at its last traded close.  Invalid or missing allocations fall back to
100% cash for that month.

Baselines: CSI300 buy-window compounding; whole-market rel20 momentum top4
(non-ST, >=250 sessions history, window avg amount >= 50,000 千元).

Usage:
    conda run -n Miro python scripts/ashare/eval_open_trader.py \
        --run "Agent(GLM-5.2)=logs/ashare_trader_open_glm_YYYYMMDD" \
        --out logs/tmpfiles/ashare_open_eval.md
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sqlite3
import statistics
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = REPO_ROOT / "data" / "ashare_pools.db"
TASKS = REPO_ROOT / "data" / "ashare_trader_open" / "standardized_data.jsonl"

INITIAL_CAPITAL = 1_000_000.0
BUY_FEE, SELL_FEE, MIN_FEE = 0.0005, 0.0015, 5.0
MOMENTUM_WINDOW = 20
TOP_K = 4
MIN_HISTORY_SESSIONS = 250
MIN_AVG_AMOUNT = 50_000.0  # 千元/day over the momentum window
V4_MIN_AVG_AMOUNT = 200_000.0
LOT_SIZE = 100
V4_CORE_WEIGHTS = {
    "510300.SH": 0.20,
    "510500.SH": 0.20,
    "512100.SH": 0.20,
}
V4_CORE_ONLY_WEIGHTS = {
    code: 0.30 for code in V4_CORE_WEIGHTS
}
V4_ALPHA_COUNT = 3
V4_ALPHA_WEIGHT = 0.10
V4_CASH_WEIGHT = 0.10


def load_tasks() -> list[dict]:
    return [
        json.loads(line)
        for line in TASKS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def parse_boxed(text: str) -> dict[str, float] | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    boxed = re.findall(r"\\boxed\{([^{}]*)\}", raw, flags=re.DOTALL)
    # task logs store final_boxed_answer already unwrapped, so fall back to
    # the raw content when no \boxed{} shell is present.
    candidate = boxed[-1] if boxed else raw
    tokens = [t.strip() for t in re.split(r"[,，;；\n]+", candidate) if t.strip()]
    out: dict[str, float] = {}
    for token in tokens:
        m = re.fullmatch(
            r"(CASH|现金|\d{6}(?:\.(?:SH|SZ))?)\s*[:=：]\s*([0-9.eE+-]+)",
            token,
            flags=re.IGNORECASE,
        )
        if not m:
            return None
        key = m.group(1).upper()
        if key in ("CASH", "现金"):
            key = "CASH"
        try:
            out[key] = float(m.group(2))
        except ValueError:
            return None
    if "CASH" not in out:
        return None
    total = sum(out.values())
    if abs(total - 1.0) > 1e-4:
        return None
    if any(v < -1e-9 for v in out.values()):
        return None
    if any(v > 0.25 + 1e-9 for k, v in out.items() if k != "CASH"):
        return None
    return out


class Market:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self.cal = [
            d
            for (d,) in conn.execute(
                "SELECT cal_date FROM trade_cal WHERE is_open=1 ORDER BY cal_date"
            )
        ]
        self.index_close = dict(
            conn.execute(
                "SELECT trade_date, close FROM index_daily WHERE ts_code='000300.SH'"
            )
        )

    def sessions(self, start: str, end: str) -> list[str]:
        return [d for d in self.cal if start < d <= end]

    def _asset_table(self, ts_code: str) -> str:
        return "etf_daily" if ts_code in V4_CORE_WEIGHTS else "market_daily"

    def asset_entry_close(self, ts_code: str, entry: str) -> float | None:
        table = self._asset_table(ts_code)
        row = self.conn.execute(
            f"SELECT close FROM {table} WHERE ts_code=? AND trade_date=?",
            (ts_code, entry),
        ).fetchone()
        close = float(row[0]) if row and row[0] is not None else None
        return close if close is not None and close > 0.0 else None

    def asset_window_return(
        self, ts_code: str, entry: str, exit_: str
    ) -> float | None:
        table = self._asset_table(ts_code)
        rows = self.conn.execute(
            f"SELECT pct_chg FROM {table} WHERE ts_code=? "
            "AND trade_date>? AND trade_date<=? ORDER BY trade_date",
            (ts_code, entry, exit_),
        ).fetchall()
        if not rows:
            return None
        nav = 1.0
        for (chg,) in rows:
            if chg is not None:
                nav *= 1.0 + chg / 100.0
        return nav - 1.0

    def stock_window_return(self, ts_code: str, entry: str, exit_: str) -> float | None:
        """Compatibility alias for stock-only baseline helpers."""
        return self.asset_window_return(ts_code, entry, exit_)

    def index_window_return(self, entry: str, exit_: str) -> float:
        p0, p1 = self.index_close.get(entry), self.index_close.get(exit_)
        if not p0 or not p1:
            raise ValueError(f"index close missing for {entry}..{exit_}")
        return p1 / p0 - 1.0

    def _filtered_candidates(self, as_of: str) -> set[str]:
        """Non-ST liquid names with enough history at as_of (shared by baselines)."""
        dates = [d for d in self.cal if d <= as_of][-(MOMENTUM_WINDOW + 1):]
        if len(dates) < MOMENTUM_WINDOW + 1:
            return set()
        placeholder = ",".join("?" * (len(dates) - 1))
        rows = self.conn.execute(
            f"""
            SELECT d.ts_code,
                   SUM(CASE WHEN d.pct_chg IS NULL THEN 0 ELSE 1 END) AS n,
                   AVG(d.amount) AS avg_amount
            FROM market_daily d
            WHERE d.trade_date IN ({placeholder})
            GROUP BY d.ts_code
            """,
            dates[1:],
        ).fetchall()
        candidates = {
            code
            for code, n, avg_amount in rows
            if n >= MOMENTUM_WINDOW * 0.8 and (avg_amount or 0) >= MIN_AVG_AMOUNT
        }
        hist = {
            code
            for (code,) in self.conn.execute(
                "SELECT ts_code FROM market_daily WHERE trade_date<=? "
                "GROUP BY ts_code HAVING COUNT(*)>=?",
                (as_of, MIN_HISTORY_SESSIONS),
            )
        }
        non_st = {
            code
            for (code,) in self.conn.execute(
                "SELECT ts_code FROM stock_basic_all "
                "WHERE name NOT LIKE '%ST%' AND name NOT LIKE '%退%'"
            )
        }
        return candidates & hist & non_st

    def momentum_top4(self, as_of: str) -> list[str]:
        dates = [d for d in self.cal if d <= as_of][-(MOMENTUM_WINDOW + 1):]
        if len(dates) < MOMENTUM_WINDOW + 1:
            return []
        start, end = dates[0], dates[-1]
        candidates = self._filtered_candidates(as_of)
        if not candidates:
            return []
        idx_ret = None
        p0, p1 = self.index_close.get(start), self.index_close.get(end)
        if p0 and p1:
            idx_ret = p1 / p0 - 1.0
        scored = []
        placeholder2 = ",".join("?" * len(candidates))
        cum: dict[str, float] = {}
        for code, chg, date in self.conn.execute(
            f"SELECT ts_code, pct_chg, trade_date FROM market_daily "
            f"WHERE trade_date>? AND trade_date<=? AND ts_code IN ({placeholder2}) "
            f"ORDER BY ts_code, trade_date",
            (start, end, *candidates),
        ):
            cum[code] = cum.get(code, 1.0) * (1.0 + (chg or 0.0) / 100.0)
        for code, nav in cum.items():
            rel = nav - 1.0 - (idx_ret or 0.0)
            scored.append((code, rel))
        scored.sort(key=lambda x: (-x[1], x[0]))
        return [c for c, _ in scored[:TOP_K]]

    def low_pe_top4(self, as_of: str) -> list[str]:
        """Cheapest positive-PE liquid names at as_of (value baseline)."""
        candidates = self._filtered_candidates(as_of)
        if not candidates:
            return []
        placeholder = ",".join("?" * len(candidates))
        rows = self.conn.execute(
            f"""
            SELECT b.ts_code, b.pe_ttm
            FROM market_daily_basic b
            JOIN (
                SELECT ts_code, MAX(trade_date) AS md
                FROM market_daily_basic
                WHERE trade_date <= ? AND ts_code IN ({placeholder})
                GROUP BY ts_code
            ) latest ON b.ts_code = latest.ts_code AND b.trade_date = latest.md
            WHERE b.pe_ttm IS NOT NULL AND b.pe_ttm > 0
            """,
            (as_of, *candidates),
        ).fetchall()
        rows.sort(key=lambda x: (x[1], x[0]))
        return [code for code, _ in rows[:TOP_K]]

    def v4_alpha_candidates(self, as_of: str) -> list[str]:
        """Liquid, profitable, non-extreme point-in-time v4 stock universe."""
        candidates = self._filtered_candidates(as_of)
        dates = [d for d in self.cal if d <= as_of][-(MOMENTUM_WINDOW + 1):]
        if not candidates or len(dates) < MOMENTUM_WINDOW + 1:
            return []
        placeholder = ",".join("?" * len(candidates))
        stats: dict[str, dict[str, float]] = {}
        for code, change, amount in self.conn.execute(
            f"SELECT ts_code,pct_chg,amount FROM market_daily "
            f"WHERE trade_date>? AND trade_date<=? "
            f"AND ts_code IN ({placeholder}) ORDER BY ts_code,trade_date",
            (dates[0], dates[-1], *candidates),
        ):
            row = stats.setdefault(code, {"nav": 1.0, "amount": 0.0, "n": 0.0})
            row["nav"] *= 1.0 + float(change or 0.0) / 100.0
            if amount is not None:
                row["amount"] += float(amount)
                row["n"] += 1.0
        latest = dates[-1]
        basics = {
            code: pe
            for code, pe in self.conn.execute(
                f"SELECT ts_code,pe_ttm FROM market_daily_basic "
                f"WHERE trade_date=? AND ts_code IN ({placeholder})",
                (latest, *candidates),
            )
        }
        eligible = []
        for code, row in stats.items():
            pe = basics.get(code)
            ret = (row["nav"] - 1.0) * 100.0
            avg_amount = row["amount"] / row["n"] if row["n"] else 0.0
            if (
                pe is not None
                and 5.0 <= float(pe) <= 60.0
                and -20.0 <= ret <= 50.0
                and avg_amount >= V4_MIN_AVG_AMOUNT
            ):
                eligible.append(code)
        return sorted(eligible)

    def deterministic_growth_top3(self, as_of: str) -> tuple[list[str], int]:
        """Rank cached PIT financial candidates with the frozen v4 rule score.

        The cache is populated by point-in-time Agent comparisons.  Coverage is
        returned and printed in the report because it is not a full-market
        financial baseline.
        """
        candidates = self.v4_alpha_candidates(as_of)
        if not candidates:
            return [], 0
        candidate_set = set(candidates)
        financial_rows = self.conn.execute(
            "SELECT ts_code,ann_date,end_date,roe,grossprofit_margin,"
            "or_yoy,netprofit_yoy FROM fina_cache WHERE ann_date<=? "
            "ORDER BY ts_code,end_date DESC,ann_date DESC",
            (as_of,),
        ).fetchall()
        by_code: dict[str, list[tuple]] = {}
        for row in financial_rows:
            if row[0] in candidate_set:
                by_code.setdefault(row[0], []).append(row[1:])

        dates60 = [d for d in self.cal if d <= as_of][-61:]
        old_date = dates60[0] if dates60 else as_of
        latest_basic = dict(
            self.conn.execute(
                "SELECT ts_code,pe_ttm FROM market_daily_basic WHERE trade_date=?",
                (as_of,),
            )
        )
        old_basic = dict(
            self.conn.execute(
                "SELECT ts_code,pe_ttm FROM market_daily_basic WHERE trade_date=?",
                (old_date,),
            )
        )
        dates20 = [d for d in self.cal if d <= as_of][-21:]
        idx_ret = self._index_return_for_dates(dates20)
        window: dict[str, dict[str, float]] = {}
        if len(dates20) == 21:
            placeholder = ",".join("?" * len(candidates))
            for code, change, amount in self.conn.execute(
                f"SELECT ts_code,pct_chg,amount FROM market_daily "
                f"WHERE trade_date>? AND trade_date<=? "
                f"AND ts_code IN ({placeholder}) ORDER BY ts_code,trade_date",
                (dates20[0], dates20[-1], *candidates),
            ):
                row = window.setdefault(
                    code, {"nav": 1.0, "amount": 0.0, "n": 0.0}
                )
                row["nav"] *= 1.0 + float(change or 0.0) / 100.0
                if amount is not None:
                    row["amount"] += float(amount)
                    row["n"] += 1.0

        ranked: list[tuple[float, str]] = []
        for code, rows in by_code.items():
            if not rows:
                continue
            latest = rows[0]
            previous = next((row for row in rows[1:] if row[1] < latest[1]), None)
            roe = latest[2]
            margin = latest[3]
            revenue_yoy = latest[4]
            profit_yoy = latest[5]
            if (
                revenue_yoy is not None
                and profit_yoy is not None
                and float(revenue_yoy) < 0.0
                and float(profit_yoy) < 0.0
            ):
                continue
            score = 0.0
            if profit_yoy is not None and float(profit_yoy) >= 30.0:
                score += 2.0
            elif profit_yoy is not None and float(profit_yoy) >= 0.0:
                score += 0.5
            if revenue_yoy is not None and float(revenue_yoy) >= 20.0:
                score += 1.0
            if roe is not None and float(roe) >= 8.0:
                score += 1.0
            if (
                previous is not None
                and margin is not None
                and previous[3] is not None
                and float(margin) >= float(previous[3])
            ):
                score += 0.5
            if (
                previous is not None
                and profit_yoy is not None
                and previous[5] is not None
                and float(profit_yoy) > float(previous[5])
            ):
                score += 1.0
            pe = latest_basic.get(code)
            if pe is not None and 5.0 <= float(pe) <= 40.0:
                score += 1.0
            old_pe = old_basic.get(code)
            if (
                pe is not None
                and old_pe is not None
                and float(old_pe) > 0.0
                and float(pe) < float(old_pe)
            ):
                score += 0.5
            stats = window.get(code, {})
            rel20 = (
                (float(stats.get("nav", 1.0)) - 1.0) * 100.0 - idx_ret
                if idx_ret is not None
                else None
            )
            if rel20 is not None and 0.0 <= rel20 <= 30.0:
                score += 1.0
            avg_amount = (
                float(stats.get("amount", 0.0)) / float(stats.get("n", 1.0))
                if stats.get("n", 0.0)
                else 0.0
            )
            if avg_amount >= 500_000.0:
                score += 1.0
            ranked.append((score, code))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [code for _, code in ranked[:V4_ALPHA_COUNT]], len(ranked)

    def _index_return_for_dates(self, dates: list[str]) -> float | None:
        if len(dates) < 2:
            return None
        first = self.index_close.get(dates[0])
        last = self.index_close.get(dates[-1])
        if not first or not last:
            return None
        return (last / first - 1.0) * 100.0

    def equal_weight_return(
        self, pool: Sequence[str], entry: str, exit_: str
    ) -> float:
        """Average window return of the tradable pool (proportional fees only).

        The per-trade 5 CNY minimum fee is skipped: with ~4,800 names it would
        dominate returns and the baseline is meant as a style reference, not
        an implementable portfolio.
        """
        pool_set = set(pool)
        cum: dict[str, float] = {}
        for code, chg in self.conn.execute(
            "SELECT ts_code, pct_chg FROM market_daily "
            "WHERE trade_date>? AND trade_date<=? ORDER BY ts_code, trade_date",
            (entry, exit_),
        ):
            if code in pool_set:
                cum[code] = cum.get(code, 1.0) * (1.0 + (chg or 0.0) / 100.0)
        if not cum:
            return 0.0
        # Names with no traded row stay at cash-equivalent 1.0.
        navs = [cum.get(code, 1.0) for code in pool]
        gross = sum(navs) / len(navs)
        return gross * (1.0 - BUY_FEE) * (1.0 - SELL_FEE) - 1.0


def _affordable_lot(
    target: float,
    price: float,
    buy_fee_rate: float = BUY_FEE,
    min_fee: float = MIN_FEE,
) -> tuple[int, float]:
    """Largest 100-unit lot whose notional plus buy fee fits the target."""
    if target <= 0.0 or price <= 0.0:
        return 0, 0.0
    shares = int(target // (price * LOT_SIZE)) * LOT_SIZE
    while shares > 0:
        notional = shares * price
        fee = max(min_fee, notional * buy_fee_rate)
        if notional + fee <= target + 1e-9:
            return shares, fee
        shares -= LOT_SIZE
    return 0, 0.0


def replay(
    market: Market,
    tasks: list[dict],
    allocations: dict[str, dict[str, float] | None],
) -> tuple[float, list[dict]]:
    """Replay fully liquidated monthly portfolios with 100-unit lot rounding."""
    capital = INITIAL_CAPITAL
    months = []
    for task in tasks:
        meta = task["metadata"]
        entry, exit_ = meta["entry_date"], meta["exit_date"]
        as_of = meta["as_of"]
        weights = allocations.get(as_of)
        start_cap = capital
        detail = {"as_of": as_of, "entry": entry, "exit": exit_}
        if not weights:
            months.append(
                {
                    **detail,
                    "net": 0.0,
                    "capital": capital,
                    "index": meta["index_return"],
                    "note": "invalid->cash",
                }
            )
            continue
        ending_value = start_cap * weights.get("CASH", 0.0)
        entry_cash = ending_value
        fees = 0.0
        missing: list[str] = []
        unfilled: list[str] = []
        executed_holdings = 0
        for code, w in weights.items():
            if code == "CASH" or w <= 0:
                continue
            target = start_cap * w
            price = market.asset_entry_close(code, entry)
            period_return = market.asset_window_return(code, entry, exit_)
            if price is None or period_return is None:
                missing.append(code)
                ending_value += target
                entry_cash += target
                continue
            shares, buy_fee = _affordable_lot(target, price)
            if shares <= 0:
                unfilled.append(code)
                ending_value += target
                entry_cash += target
                continue
            notional = shares * price
            residual = target - notional - buy_fee
            value_before_sale = notional * (1.0 + period_return)
            sell_fee = max(MIN_FEE, value_before_sale * SELL_FEE)
            ending_value += residual + value_before_sale - sell_fee
            entry_cash += residual
            fees += buy_fee + sell_fee
            executed_holdings += 1
        capital = ending_value
        months.append(
            {
                **detail,
                "net": capital / start_cap - 1.0,
                "capital": capital,
                "fees": round(fees, 2),
                "holdings": executed_holdings,
                "cash_w": round(weights.get("CASH", 0.0), 4),
                "effective_cash_w": round(entry_cash / start_cap, 4),
                "missing": missing,
                "unfilled": unfilled,
                "index": meta["index_return"],
            }
        )
    return capital / INITIAL_CAPITAL - 1.0, months


def annualized_sharpe(
    monthly_returns: Sequence[float], risk_free_monthly: float = 0.0
) -> float | None:
    """Monthly-frequency annualized Sharpe: sqrt(12) * mean / sample std.

    Uses net monthly returns with a flat monthly risk-free rate (0 by
    default, i.e. a same-window comparison convention, not an absolute
    performance claim). Returns None when there are fewer than two
    observations or the sample standard deviation is zero.
    """
    excess = [float(r) - risk_free_monthly for r in monthly_returns]
    if len(excess) < 2:
        return None
    std = statistics.stdev(excess)
    if not math.isfinite(std) or std <= 0.0:
        return None
    return math.sqrt(12.0) * statistics.mean(excess) / std


def replay_metrics(total: float, months: list[dict]) -> dict[str, float | None]:
    peak = INITIAL_CAPITAL
    max_drawdown = 0.0
    valid = [month for month in months if "note" not in month]
    for month in months:
        capital = float(month.get("capital", INITIAL_CAPITAL))
        peak = max(peak, capital)
        if peak > 0.0:
            max_drawdown = min(max_drawdown, capital / peak - 1.0)
    return {
        "total": total,
        "max_drawdown": max_drawdown,
        "worst_month": min((float(m["net"]) for m in months), default=0.0),
        "win_rate": (
            sum(float(m["net"]) > 0.0 for m in valid) / len(valid)
            if valid
            else 0.0
        ),
        "fees": sum(float(m.get("fees", 0.0)) for m in months),
        "valid_months": float(len(valid)),
        # Invalid months fall back to cash and enter as 0-return months.
        "annualized_sharpe": annualized_sharpe(
            [float(m["net"]) for m in months]
        ),
    }


def hybrid_allocation(alpha_codes: Sequence[str]) -> dict[str, float]:
    selected = list(dict.fromkeys(str(code).upper() for code in alpha_codes))[
        :V4_ALPHA_COUNT
    ]
    weights = {
        **V4_CORE_WEIGHTS,
        **{code: V4_ALPHA_WEIGHT for code in selected},
    }
    weights["CASH"] = round(
        1.0 - sum(V4_CORE_WEIGHTS.values()) - len(selected) * V4_ALPHA_WEIGHT,
        10,
    )
    return weights


def core_only_allocation() -> dict[str, float]:
    return {**V4_CORE_ONLY_WEIGHTS, "CASH": V4_CASH_WEIGHT}


def random_alpha_allocations(
    tasks: list[dict],
    seed: int,
    candidates_by_as_of: dict[str, list[str]],
) -> dict[str, dict[str, float]]:
    rng = random.Random(seed)
    out: dict[str, dict[str, float]] = {}
    for task in tasks:
        meta = task["metadata"]
        candidates = candidates_by_as_of[meta["as_of"]]
        picks = rng.sample(candidates, V4_ALPHA_COUNT)
        out[meta["as_of"]] = hybrid_allocation(picks)
    return out


def extract_run_allocations(run_dir: Path) -> dict[str, dict[str, float] | None]:
    out: dict[str, dict[str, float] | None] = {}
    for p in sorted(run_dir.glob("task_*_attempt_1.json")):
        data = json.loads(p.read_text(encoding="utf-8"))
        as_of = str((data.get("input", {}).get("metadata", {}) or {}).get("as_of", ""))
        if not as_of:
            m = re.search(r"(\d{4}-\d{2}-\d{2})", p.stem)
            as_of = m.group(1) if m else p.stem
        out[as_of] = parse_boxed(data.get("final_boxed_answer", ""))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="LABEL=/path/to/run_dir (repeatable)",
    )
    parser.add_argument("--out", default="")
    parser.add_argument(
        "--random-seeds",
        type=int,
        default=100,
        help="number of seeded random-alpha portfolios (default: 100)",
    )
    args = parser.parse_args()

    tasks = load_tasks()
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.execute("PRAGMA busy_timeout=30000")
    market = Market(conn)

    run_allocations: list[
        tuple[str, Path, dict[str, dict[str, float] | None]]
    ] = []
    for spec in args.run:
        label, separator, path = spec.partition("=")
        if not separator or not label or not path:
            raise ValueError(f"invalid --run spec: {spec!r}")
        run_dir = Path(path)
        run_allocations.append(
            (label, run_dir, extract_run_allocations(run_dir))
        )

    lines = [
        "# 开放池（全A股）交易员评测",
        "",
        "- 可交易组合统一按 100 股/份整手向下取整，未成交余额留现金；"
        "买入 0.05%、卖出 0.15%、每笔最低 5 元。",
        "- 全市场全等权仍是不可交易风格参考：仅计比例费用，不计整手和最低费。",
        "",
    ]

    index_nav = 1.0
    index_months: list[dict] = []
    for task in tasks:
        meta = task["metadata"]
        index_ret = market.index_window_return(
            meta["entry_date"], meta["exit_date"]
        )
        index_nav *= 1.0 + index_ret
        index_months.append(
            {
                "as_of": meta["as_of"],
                "net": index_ret,
                "capital": INITIAL_CAPITAL * index_nav,
                "index": index_ret,
            }
        )
    index_total = index_nav - 1.0

    mom_alloc: dict[str, dict[str, float]] = {}
    for task in tasks:
        meta = task["metadata"]
        top = market.momentum_top4(meta["entry_date"])
        mom_alloc[meta["as_of"]] = (
            {**{c: 1.0 / TOP_K for c in top}, "CASH": 0.0} if top else None
        )
    mom_total, mom_months = replay(market, tasks, mom_alloc)

    pe_alloc: dict[str, dict[str, float]] = {}
    for task in tasks:
        meta = task["metadata"]
        top = market.low_pe_top4(meta["entry_date"])
        pe_alloc[meta["as_of"]] = (
            {**{c: 1.0 / TOP_K for c in top}, "CASH": 0.0} if top else None
        )
    pe_total, pe_months = replay(market, tasks, pe_alloc)

    ew_nav = 1.0
    ew_months: list[tuple[str, float]] = []
    ew_metric_months: list[dict] = []
    for task in tasks:
        meta = task["metadata"]
        ret = market.equal_weight_return(
            meta["stock_pool"], meta["entry_date"], meta["exit_date"]
        )
        ew_nav *= 1.0 + ret
        ew_months.append((meta["as_of"], ret))
        ew_metric_months.append(
            {
                "as_of": meta["as_of"],
                "net": ret,
                "capital": INITIAL_CAPITAL * ew_nav,
            }
        )
    ew_total = ew_nav - 1.0

    core_alloc = {
        task["metadata"]["as_of"]: core_only_allocation() for task in tasks
    }
    core_total, core_months = replay(market, tasks, core_alloc)

    deterministic_alloc: dict[str, dict[str, float]] = {}
    deterministic_picks: dict[str, list[str]] = {}
    deterministic_coverage: dict[str, int] = {}
    alpha_universes: dict[str, list[str]] = {}
    for task in tasks:
        meta = task["metadata"]
        as_of = meta["as_of"]
        alpha_universes[as_of] = market.v4_alpha_candidates(meta["entry_date"])
        picks, coverage = market.deterministic_growth_top3(meta["entry_date"])
        deterministic_picks[as_of] = picks
        deterministic_coverage[as_of] = coverage
        deterministic_alloc[as_of] = hybrid_allocation(picks)
    deterministic_total, deterministic_months = replay(
        market, tasks, deterministic_alloc
    )

    v1_counterfactual_alloc: dict[str, dict[str, float]] | None = None
    v1_counterfactual_picks: dict[str, list[str]] = {}
    for label, _, allocations in run_allocations:
        if "v1" not in label.lower():
            continue
        v1_counterfactual_alloc = {}
        for task in tasks:
            as_of = task["metadata"]["as_of"]
            weights = allocations.get(as_of) or {}
            picks = [
                code
                for code, _ in sorted(
                    (
                        (code, weight)
                        for code, weight in weights.items()
                        if code != "CASH" and code not in V4_CORE_WEIGHTS
                    ),
                    key=lambda item: (-item[1], item[0]),
                )
            ][:V4_ALPHA_COUNT]
            v1_counterfactual_picks[as_of] = picks
            v1_counterfactual_alloc[as_of] = hybrid_allocation(picks)
        break
    v1_counterfactual_total: float | None = None
    v1_counterfactual_months: list[dict] = []
    if v1_counterfactual_alloc is not None:
        v1_counterfactual_total, v1_counterfactual_months = replay(
            market, tasks, v1_counterfactual_alloc
        )

    random_totals: list[float] = []
    random_drawdowns: list[float] = []
    for seed in range(max(0, int(args.random_seeds))):
        random_alloc = random_alpha_allocations(
            tasks,
            seed,
            alpha_universes,
        )
        random_total, random_months = replay(market, tasks, random_alloc)
        random_totals.append(random_total)
        random_drawdowns.append(
            replay_metrics(random_total, random_months)["max_drawdown"]
        )

    def percentile(values: list[float], quantile: float) -> float:
        if not values:
            return float("nan")
        ordered = sorted(values)
        position = (len(ordered) - 1) * quantile
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        fraction = position - lower
        return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction

    baseline_rows: list[tuple[str, dict[str, float]]] = [
        ("沪深300", replay_metrics(index_total, index_months)),
        ("rel20动量top4", replay_metrics(mom_total, mom_months)),
        ("低PE top4", replay_metrics(pe_total, pe_months)),
        ("90% ETF等权核心+10%现金", replay_metrics(core_total, core_months)),
        (
            "60% ETF+缓存候选成长规则+现金",
            replay_metrics(deterministic_total, deterministic_months),
        ),
        ("全市场全等权（不可交易参考）", replay_metrics(ew_total, ew_metric_months)),
    ]
    if v1_counterfactual_total is not None:
        baseline_rows.append(
            (
                "60% ETF+v1前三选股反事实",
                replay_metrics(
                    v1_counterfactual_total,
                    v1_counterfactual_months,
                ),
            )
        )

    run_results: list[
        tuple[str, dict[str, dict[str, float] | None], float, list[dict]]
    ] = []
    for label, _, alloc in run_allocations:
        total, months = replay(market, tasks, alloc)
        run_results.append((label, alloc, total, months))

    def _fmt_sharpe(value: float | None) -> str:
        return f"{value:.2f}" if value is not None else "—"

    lines.append("## 统一指标")
    lines.append(
        "| 策略 | 总收益% | 指数收益% | 超额pp | 最大回撤% | Sharpe(年化) "
        "| 相对ETF核心pp | 最差月% | 胜率% | 费用 | 有效月 |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, metrics in baseline_rows:
        lines.append(
            f"| {name} | {metrics['total'] * 100:+.2f} | "
            f"{index_total * 100:+.2f} | "
            f"{(metrics['total'] - index_total) * 100:+.2f} | "
            f"{metrics['max_drawdown'] * 100:.2f} | "
            f"{_fmt_sharpe(metrics['annualized_sharpe'])} | "
            f"{(metrics['total'] - core_total) * 100:+.2f} | "
            f"{metrics['worst_month'] * 100:+.2f} | "
            f"{metrics['win_rate'] * 100:.1f} | "
            f"{metrics['fees']:,.0f} | {int(metrics['valid_months'])}/12 |"
        )
    for label, _, total, months in run_results:
        metrics = replay_metrics(total, months)
        lines.append(
            f"| {label} | {metrics['total'] * 100:+.2f} | "
            f"{index_total * 100:+.2f} | "
            f"{(metrics['total'] - index_total) * 100:+.2f} | "
            f"{metrics['max_drawdown'] * 100:.2f} | "
            f"{_fmt_sharpe(metrics['annualized_sharpe'])} | "
            f"{(metrics['total'] - core_total) * 100:+.2f} | "
            f"{metrics['worst_month'] * 100:+.2f} | "
            f"{metrics['win_rate'] * 100:.1f} | "
            f"{metrics['fees']:,.0f} | {int(metrics['valid_months'])}/12 |"
        )
    lines.append("")
    if random_totals:
        lines.append(
            f"- 随机流动性 Alpha（{len(random_totals)} 个固定种子）总收益："
            f"P10 {percentile(random_totals, 0.10) * 100:+.2f}%，"
            f"中位 {statistics.median(random_totals) * 100:+.2f}%，"
            f"P90 {percentile(random_totals, 0.90) * 100:+.2f}%；"
            f"最大回撤中位 {statistics.median(random_drawdowns) * 100:.2f}%。"
        )
    lines.append(
        "- “缓存候选成长规则”只在运行期间已按点时查询并缓存财务的股票中排序；"
        "覆盖率会逐月披露，不能当作完整全市场规则基线。"
    )
    lines.append("")

    for label, alloc, total, months in run_results:
        metrics = replay_metrics(total, months)
        n_valid = int(metrics["valid_months"])
        lines.append(
            f"## {label}: 总收益 {total * 100:+.2f}%"
            f"（有效月 {n_valid}/{len(tasks)}，最大回撤 "
            f"{metrics['max_drawdown'] * 100:.2f}%）"
        )
        lines.append(
            "| 月份 | 净收益% | 指数% | 成交持仓 | 目标现金 | 有效现金 | 费用 | 异常 |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
        for m in months:
            abnormal = (
                m.get("note", "")
                or ",".join(m.get("missing", []))
                or (
                    "整手未成交:" + ",".join(m.get("unfilled", []))
                    if m.get("unfilled")
                    else ""
                )
            )
            lines.append(
                f"| {m['as_of']} | {m['net'] * 100:+.2f} | "
                f"{m.get('index', 0) * 100:+.2f} | {m.get('holdings', 0)} | "
                f"{m.get('cash_w', 1.0):.2f} | "
                f"{m.get('effective_cash_w', 1.0):.2f} | "
                f"{m.get('fees', 0):,.0f} | {abnormal} |"
            )
        lines.append("")

    lines.append("## 规则基线逐月")
    lines.append(
        "| 月份 | ETF核心% | 成长规则% | 成长规则选股/缓存覆盖 | "
        "动量top4% | 低PE top4% | 全等权参考% |"
    )
    lines.append("|---|---:|---:|---|---:|---:|---:|")
    ew_by_month = dict(ew_months)
    for task, core_m, growth_m, mom_m, pe_m in zip(
        tasks,
        core_months,
        deterministic_months,
        mom_months,
        pe_months,
    ):
        as_of = mom_m["as_of"]
        growth_codes = ",".join(deterministic_picks[as_of]) or "none"
        lines.append(
            f"| {as_of} | {core_m['net'] * 100:+.2f} | "
            f"{growth_m['net'] * 100:+.2f} | "
            f"{growth_codes} / {deterministic_coverage[as_of]} | "
            f"{mom_m['net'] * 100:+.2f} | {pe_m['net'] * 100:+.2f} | "
            f"{ew_by_month.get(as_of, 0.0) * 100:+.2f} |"
        )

    report = "\n".join(lines)
    print(report)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report + "\n", encoding="utf-8")
        print(f"\nwritten -> {out_path}")
    conn.close()


if __name__ == "__main__":
    main()
