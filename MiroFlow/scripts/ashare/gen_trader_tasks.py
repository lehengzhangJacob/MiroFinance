#!/usr/bin/env python3
"""Generate monthly unified A-share portfolio-allocation tasks."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.ashare.gen_tasks import (  # noqa: E402
    HORIZON,
    fmt_date,
    load_prices,
    monthly_first_trading_days,
    window_return,
)

OUT_DIR = ROOT / "data" / "ashare_trader"

QUESTION_TEMPLATE = """\
你是同一名持续管理资金的 A 股交易员。当前日期为 {as_of}（收盘后）。

你必须在一次统一决策中同时比较下面 16 只股票，并为同一个资金账户决定从 {as_of} 收盘买入、持有 {horizon} 个交易日后按收盘卖出的组合仓位。回测从 100 万元开始并逐期复利；目标是在控制回撤和交易成本的同时，提高相对沪深300指数（000300.SH）的净收益。

{stock_list}

组合约束（务必遵守）：
- 仅做多，不得做空或使用杠杆；每只股票权重必须在 0 到 0.25 之间；
- 可以持有现金；股票权重与 CASH 权重之和必须严格等于 1；
- 未写出的股票视为 0 仓位，CASH 必须显式写出；
- 本任务按买入费率 0.05%、卖出费率 0.15%、每笔最低 5 元评价，20 个交易日后全部卖出。

数据使用规则（务必遵守）：
- 只能使用 ashare-market 工具，禁止网络搜索，禁止使用 {as_of} 之后的行情、新闻、财报、真实收益或事后排名；
- 本地 Tushare 缓存有多少就看多少：lookback_days 不传或传 0 即返回截至 {as_of} 的全部可用数据；不限只数、不限次数、不限工具类型；
- 可用工具包括 ashare_trader_universe_context、ashare_price_history、ashare_index_history、ashare_valuation、ashare_financials、ashare_ml_signal、ashare_stock_info、ashare_cross_section_snapshot 等，按需自由组合；
- 必须先评估全池相对机会、风险和信号冲突，再决定选股、集中度与现金比例，不能把 16 只股票拆成互不相关的独立结论。

输出要求：
- 正文简要说明全池比较、入选理由、主要风险和现金决策；
- 最后一行严格输出 \\boxed{{代码:权重,代码:权重,...,CASH:权重}}；
- 权重使用 0 到 1 的小数，例如 \\boxed{{603259.SH:0.25,601899.SH:0.20,CASH:0.55}}，不得在 boxed 内容中加入解释文字。
"""


def build_tasks() -> list[dict[str, Any]]:
    stocks, index, meta = load_prices()
    pool = meta["stock_pool"]
    planned_rebalance_days = monthly_first_trading_days(index)
    index_positions = {
        str(date): position
        for position, date in enumerate(index["trade_date"].tolist())
    }
    stock_list = "\n".join(
        f"- {code} {info['name']}（{info['industry']}）"
        for code, info in pool.items()
    )

    tasks: list[dict[str, Any]] = []
    previous_exit = ""
    for planned_day in planned_rebalance_days:
        # A single cash account cannot fund overlapping 20-session portfolios.
        # If a holiday-shortened month starts before the prior liquidation,
        # rebalance at that liquidation close instead.  The adjusted date stays
        # in the intended calendar month for the current benchmark range.
        day = max(planned_day, previous_exit)
        if day[:6] != planned_day[:6]:
            raise ValueError(
                f"20-session schedule drifted outside month: "
                f"{planned_day} -> {day}"
            )
        index_result = window_return(index, "close", day, HORIZON)
        if index_result is None:
            continue
        index_return, exit_date = index_result
        previous_exit = exit_date
        stock_returns: dict[str, float] = {}
        excess_returns: dict[str, float] = {}
        for ts_code in pool:
            stock_result = window_return(
                stocks[ts_code], "close_qfq", day, HORIZON
            )
            if stock_result is None:
                raise ValueError(
                    f"missing {HORIZON}-day window for {ts_code} @ {day}"
                )
            stock_return, stock_exit = stock_result
            if stock_exit != exit_date:
                raise ValueError(
                    f"calendar mismatch for {ts_code} @ {day}: "
                    f"{stock_exit} != {exit_date}"
                )
            stock_returns[ts_code] = round(float(stock_return), 8)
            excess_returns[ts_code] = round(
                float(stock_return - index_return), 8
            )

        ground_truth_rank = sorted(
            pool,
            key=lambda code: (-excess_returns[code], code),
        )
        as_of = fmt_date(day)
        tasks.append(
            {
                "task_id": f"ashare_trader_{as_of}",
                "task_question": QUESTION_TEMPLATE.format(
                    as_of=as_of,
                    as_of_compact=day,
                    horizon=HORIZON,
                    stock_list=stock_list,
                ),
                # There is no unique optimal allocation.  The framework judge
                # checks constraints; financial quality is evaluated offline.
                "ground_truth": "VALID_PORTFOLIO",
                "file_path": None,
                "metadata": {
                    "dataset_name": "ashare-trader",
                    "task_type": "portfolio_allocation",
                    "as_of": as_of,
                    "scheduled_month": planned_day[:6],
                    "planned_entry_date": planned_day,
                    "entry_date": day,
                    "exit_date": exit_date,
                    "entry_shift_sessions": (
                        index_positions[day] - index_positions[planned_day]
                    ),
                    "horizon_trading_days": HORIZON,
                    "index_code": meta["index_code"],
                    "index_return": round(float(index_return), 8),
                    "pool_size": len(pool),
                    "stock_pool": list(pool),
                    "stock_info": pool,
                    "stock_returns": stock_returns,
                    "excess_returns": excess_returns,
                    "ground_truth_rank": ground_truth_rank,
                    "max_stock_weight": 0.25,
                    "allow_cash": True,
                    "allow_short": False,
                    "open_cost": 0.0005,
                    "close_cost": 0.0015,
                    "min_cost": 5.0,
                    "initial_capital": 1_000_000.0,
                },
            }
        )
    return tasks


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tasks = build_tasks()
    output = OUT_DIR / "standardized_data.jsonl"
    output.write_text(
        "".join(
            json.dumps(task, ensure_ascii=False) + "\n" for task in tasks
        ),
        encoding="utf-8",
    )
    smoke = [task["task_id"] for task in tasks[:1]]
    (OUT_DIR / "smoke_whitelist.json").write_text(
        json.dumps(smoke, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    dates = [task["metadata"]["entry_date"] for task in tasks]
    print(f"wrote {len(tasks)} trader tasks -> {output}")
    print(f"pool size: {tasks[0]['metadata']['pool_size'] if tasks else 0}; dates: {dates}")


if __name__ == "__main__":
    main()
