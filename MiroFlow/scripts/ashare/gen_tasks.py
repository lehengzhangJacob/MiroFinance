# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

"""Generate the A-share prediction benchmark JSONL from the local cache.

Task design (see plan): at each monthly rebalance date D the agent must
predict, using ONLY point-in-time data (<= D) served by the ashare MCP
tools, whether the stock will out- or under-perform CSI 300 over the next
20 trading days. Ground truth is computed here from qfq prices.

Usage:
    uv run python scripts/ashare/gen_tasks.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]  # MiroFlow/
DATA_DIR = REPO_ROOT / "data" / "ashare"
OUT_DIR = REPO_ROOT / "data" / "ashare_pred"

REBALANCE_START = "2024-07"
REBALANCE_END = "2025-06"  # inclusive, 12 monthly periods
HORIZON = 20  # trading days
OUTPERFORM, UNDERPERFORM = "跑赢", "跑输"

QUESTION_TEMPLATE = """\
你是一名 A 股量化研究员。当前日期为 {as_of}（收盘后）。

请基于截至 {as_of}（含当日收盘数据）的信息，预测股票 {name}（{ts_code}，{industry}行业）从 {as_of} 收盘价买入、持有 {horizon} 个交易日后（按收盘价计算、前复权口径），其区间收益率相对沪深300指数（000300.SH）同期收益率是「跑赢」还是「跑输」。

数据使用规则（务必遵守）：
- 只能使用 ashare-market 工具集（ashare_price_history、ashare_index_history、ashare_valuation、ashare_financials、ashare_stock_info）获取数据，调用时 as_of 参数必须设为 {as_of_compact}；
- 禁止使用任何网络搜索或你记忆中 {as_of} 之后的市场信息（如后续涨跌、新闻、财报），这是一个严格的点时（point-in-time）预测任务；
- 建议综合考察：近期动量与波动（如 20/60 日相对指数表现）、估值水平（PE/PB 历史分位）、最近一期已公告财务指标（注意公告日期必须早于 {as_of}）、成交与换手变化。

输出要求：先用中文简要给出 3-5 条核心依据，最后一行只输出最终结论：\\boxed{{跑赢}} 或 \\boxed{{跑输}}（二选一，不得输出其他内容）。"""


def load_prices() -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict]:
    meta = json.loads((DATA_DIR / "meta.json").read_text(encoding="utf-8"))
    stocks: dict[str, pd.DataFrame] = {}
    for ts_code in meta["stock_pool"]:
        df = pd.read_csv(DATA_DIR / f"daily_{ts_code}.csv", dtype={"trade_date": str})
        stocks[ts_code] = df.sort_values("trade_date").reset_index(drop=True)
    idx = pd.read_csv(
        DATA_DIR / f"index_{meta['index_code']}.csv", dtype={"trade_date": str}
    ).sort_values("trade_date").reset_index(drop=True)
    return stocks, idx, meta


def monthly_first_trading_days(any_daily: pd.DataFrame) -> list[str]:
    dates = any_daily["trade_date"]  # YYYYMMDD strings, trading days only
    months = dates.str[:6]
    firsts = dates.groupby(months).min()
    lo = REBALANCE_START.replace("-", "")
    hi = REBALANCE_END.replace("-", "")
    return [d for m, d in firsts.items() if lo <= m <= hi]


def window_return(df: pd.DataFrame, col: str, entry_date: str, horizon: int) -> tuple[float, str] | None:
    pos_list = df.index[df["trade_date"] == entry_date].tolist()
    if not pos_list:
        return None
    pos = pos_list[0]
    exit_pos = pos + horizon
    if exit_pos >= len(df):
        return None
    entry, exit_ = df[col].iloc[pos], df[col].iloc[exit_pos]
    return exit_ / entry - 1.0, df["trade_date"].iloc[exit_pos]


def fmt_date(yyyymmdd: str) -> str:
    return f"{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:]}"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stocks, idx, meta = load_prices()
    pool = meta["stock_pool"]

    rebalance_days = monthly_first_trading_days(idx)
    print(f"rebalance dates ({len(rebalance_days)}): {rebalance_days}")

    tasks = []
    label_counts = {OUTPERFORM: 0, UNDERPERFORM: 0}
    for ts_code, info in pool.items():
        df = stocks[ts_code]
        for day in rebalance_days:
            stock_r = window_return(df, "close_qfq", day, HORIZON)
            index_r = window_return(idx, "close", day, HORIZON)
            if stock_r is None or index_r is None:
                print(f"  skip {ts_code} @ {day}: window out of range")
                continue
            (s_ret, exit_date), (i_ret, _) = stock_r, index_r
            excess = s_ret - i_ret
            label = OUTPERFORM if excess > 0 else UNDERPERFORM
            label_counts[label] += 1

            as_of = fmt_date(day)
            task = {
                "task_id": f"ashare_{ts_code.split('.')[0]}_{as_of}",
                "task_question": QUESTION_TEMPLATE.format(
                    as_of=as_of,
                    as_of_compact=day,
                    ts_code=ts_code,
                    name=info["name"],
                    industry=info["industry"],
                    horizon=HORIZON,
                ),
                "ground_truth": label,
                "file_path": None,
                "metadata": {
                    "dataset_name": "ashare-pred",
                    "ts_code": ts_code,
                    "stock_name": info["name"],
                    "industry": info["industry"],
                    "as_of": as_of,
                    "entry_date": day,
                    "exit_date": exit_date,
                    "horizon_trading_days": HORIZON,
                    "stock_return": round(s_ret, 6),
                    "index_return": round(i_ret, 6),
                    "excess_return": round(excess, 6),
                    "index_code": meta["index_code"],
                },
            }
            tasks.append(task)

    out_file = OUT_DIR / "standardized_data.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"\nwrote {len(tasks)} tasks -> {out_file}")
    print(f"label balance: {label_counts}")

    # Deterministic smoke subset: one task per stock, staggered months.
    smoke = []
    for i, ts_code in enumerate(pool):
        day = rebalance_days[(i * 2) % len(rebalance_days)]
        smoke.append(f"ashare_{ts_code.split('.')[0]}_{fmt_date(day)}")
    (OUT_DIR / "smoke_whitelist.json").write_text(
        json.dumps(smoke, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"smoke whitelist: {smoke}")


if __name__ == "__main__":
    main()
