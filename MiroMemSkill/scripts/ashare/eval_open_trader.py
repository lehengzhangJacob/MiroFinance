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
import re
import sqlite3
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

    def stock_window_return(self, ts_code: str, entry: str, exit_: str) -> float | None:
        rows = self.conn.execute(
            "SELECT pct_chg FROM market_daily WHERE ts_code=? "
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


def replay(
    market: Market,
    tasks: list[dict],
    allocations: dict[str, dict[str, float] | None],
) -> tuple[float, list[dict]]:
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
            months.append({**detail, "net": 0.0, "note": "invalid->cash"})
            continue
        invested = 0.0
        fees = 0.0
        missing: list[str] = []
        for code, w in weights.items():
            if code == "CASH" or w <= 0:
                continue
            amt = start_cap * w
            r = market.stock_window_return(code, entry, exit_)
            if r is None:
                missing.append(code)
                invested += amt  # untradable -> stays as cash-equivalent
                continue
            buy_fee = max(MIN_FEE, amt * BUY_FEE)
            end_val = (amt - buy_fee) * (1.0 + r)
            sell_fee = max(MIN_FEE, end_val * SELL_FEE)
            invested += end_val - sell_fee
            fees += buy_fee + sell_fee
        cash_amt = start_cap * weights.get("CASH", 0.0)
        capital = invested + cash_amt
        months.append(
            {
                **detail,
                "net": capital / start_cap - 1.0,
                "fees": round(fees, 2),
                "holdings": sum(
                    1 for k, v in weights.items() if k != "CASH" and v > 0
                ),
                "cash_w": round(weights.get("CASH", 0.0), 4),
                "missing": missing,
                "index": meta["index_return"],
            }
        )
    return capital / INITIAL_CAPITAL - 1.0, months


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
    args = parser.parse_args()

    tasks = load_tasks()
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.execute("PRAGMA busy_timeout=30000")
    market = Market(conn)

    lines = ["# 开放池（全A股）交易员评测", ""]

    index_nav = 1.0
    for task in tasks:
        meta = task["metadata"]
        index_nav *= 1.0 + market.index_window_return(
            meta["entry_date"], meta["exit_date"]
        )
    lines.append(f"- 沪深300（12窗口复利）: {(index_nav - 1) * 100:+.2f}%")

    mom_alloc: dict[str, dict[str, float]] = {}
    for task in tasks:
        meta = task["metadata"]
        top = market.momentum_top4(meta["entry_date"])
        mom_alloc[meta["as_of"]] = (
            {**{c: 1.0 / TOP_K for c in top}, "CASH": 0.0} if top else None
        )
    mom_total, mom_months = replay(market, tasks, mom_alloc)
    lines.append(
        f"- 全市场rel20动量top4基线（流动性过滤后）: {mom_total * 100:+.2f}%"
    )

    pe_alloc: dict[str, dict[str, float]] = {}
    for task in tasks:
        meta = task["metadata"]
        top = market.low_pe_top4(meta["entry_date"])
        pe_alloc[meta["as_of"]] = (
            {**{c: 1.0 / TOP_K for c in top}, "CASH": 0.0} if top else None
        )
    pe_total, pe_months = replay(market, tasks, pe_alloc)
    lines.append(
        f"- 全市场低PE top4基线（pe_ttm>0 升序，同流动性过滤）: {pe_total * 100:+.2f}%"
    )

    ew_nav = 1.0
    ew_months: list[tuple[str, float]] = []
    for task in tasks:
        meta = task["metadata"]
        ret = market.equal_weight_return(
            meta["stock_pool"], meta["entry_date"], meta["exit_date"]
        )
        ew_nav *= 1.0 + ret
        ew_months.append((meta["as_of"], ret))
    lines.append(
        f"- 可交易池全等权基线（比例费用，无最低费）: {(ew_nav - 1) * 100:+.2f}%"
    )
    lines.append("")

    for spec in args.run:
        label, _, path = spec.partition("=")
        run_dir = Path(path)
        alloc = extract_run_allocations(run_dir)
        total, months = replay(market, tasks, alloc)
        n_valid = sum(1 for v in alloc.values() if v)
        lines.append(f"## {label}: 总收益 {total * 100:+.2f}%（有效月 {n_valid}/{len(tasks)}）")
        lines.append(
            "| 月份 | 净收益% | 指数% | 持仓数 | 现金 | 费用 | 异常 |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---|")
        for m in months:
            lines.append(
                f"| {m['as_of']} | {m['net'] * 100:+.2f} | "
                f"{m.get('index', 0) * 100:+.2f} | {m.get('holdings', 0)} | "
                f"{m.get('cash_w', 1.0):.2f} | {m.get('fees', 0):,.0f} | "
                f"{m.get('note', '') or (','.join(m.get('missing', [])) or '')} |"
            )
        lines.append("")

    lines.append("## 规则基线逐月")
    lines.append("| 月份 | 动量top4% | 动量持仓 | 低PE top4% | 低PE持仓 | 等权% |")
    lines.append("|---|---:|---|---:|---|---:|")
    ew_by_month = dict(ew_months)
    for task, mom_m, pe_m in zip(tasks, mom_months, pe_months):
        as_of = mom_m["as_of"]
        mom_picks = mom_alloc.get(as_of) or {}
        pe_picks = pe_alloc.get(as_of) or {}
        mom_codes = ",".join(c for c in mom_picks if c != "CASH")
        pe_codes = ",".join(c for c in pe_picks if c != "CASH")
        lines.append(
            f"| {as_of} | {mom_m['net'] * 100:+.2f} | {mom_codes} | "
            f"{pe_m['net'] * 100:+.2f} | {pe_codes} | "
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
