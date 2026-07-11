#!/usr/bin/env python3
"""Generate 12 monthly A-share cross-sectional ranking tasks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.ashare.gen_tasks import (  # noqa: E402
    HORIZON,
    fmt_date,
    load_prices,
    monthly_first_trading_days,
    window_return,
)

OUT_DIR = ROOT / "data" / "ashare_rank"

QUESTION_TEMPLATE = """\
你是一名 A 股量化研究员。当前日期为 {as_of}（收盘后）。

请仅使用截至 {as_of} 的点时信息，将下面 16 只股票按未来 {horizon} 个交易日相对沪深300指数（000300.SH）的预期超额收益从高到低完整排序：

{stock_list}

数据使用规则（务必遵守）：
- 首先调用 ashare_cross_section_snapshot(as_of={as_of_compact}) 获取全池动量、估值、Qlib与最近已公告财务特征；
- 批量快照是本任务唯一允许使用的数据工具；不得再调用单股工具，确保所有方案使用完全相同的信息集；
- 禁止使用网络搜索和 {as_of} 之后的任何信息；
- Qlib 排名、相对动量、估值与基本面只能作为点时证据，不得查询未来收益或真实排序。

输出要求：
- 正文简要说明排序依据；
- 最后一行仅输出 16 个股票代码，严格按预测超额收益从高到低排列；
- 格式必须为 \\boxed{{代码1,代码2,...,代码16}}，不得重复、遗漏或加入池外代码。
"""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stocks, index, meta = load_prices()
    pool = meta["stock_pool"]
    rebalance_days = monthly_first_trading_days(index)
    stock_list = "\n".join(
        f"- {code} {info['name']}（{info['industry']}）"
        for code, info in pool.items()
    )

    tasks = []
    for day in rebalance_days:
        index_result = window_return(index, "close", day, HORIZON)
        if index_result is None:
            continue
        index_return, exit_date = index_result
        excess_returns: dict[str, float] = {}
        stock_returns: dict[str, float] = {}
        for ts_code in pool:
            stock_result = window_return(stocks[ts_code], "close_qfq", day, HORIZON)
            if stock_result is None:
                raise ValueError(f"missing {HORIZON}-day window for {ts_code} @ {day}")
            stock_return, stock_exit = stock_result
            if stock_exit != exit_date:
                raise ValueError(
                    f"calendar mismatch for {ts_code} @ {day}: {stock_exit} != {exit_date}"
                )
            stock_returns[ts_code] = round(stock_return, 8)
            excess_returns[ts_code] = round(stock_return - index_return, 8)

        ground_truth_rank = sorted(
            pool,
            key=lambda code: (-excess_returns[code], code),
        )
        as_of = fmt_date(day)
        tasks.append(
            {
                "task_id": f"ashare_rank_{as_of}",
                "task_question": QUESTION_TEMPLATE.format(
                    as_of=as_of,
                    as_of_compact=day,
                    horizon=HORIZON,
                    stock_list=stock_list,
                ),
                # The framework judge treats this as a format-validity task;
                # continuous ranking quality is computed by eval_rank.py.
                "ground_truth": ",".join(ground_truth_rank),
                "file_path": None,
                "metadata": {
                    "dataset_name": "ashare-rank",
                    "task_type": "cross_section_rank",
                    "as_of": as_of,
                    "entry_date": day,
                    "exit_date": exit_date,
                    "horizon_trading_days": HORIZON,
                    "index_code": meta["index_code"],
                    "index_return": round(index_return, 8),
                    "pool_size": len(pool),
                    "stock_pool": list(pool),
                    "stock_returns": stock_returns,
                    "excess_returns": excess_returns,
                    "ground_truth_rank": ground_truth_rank,
                },
            }
        )

    output = OUT_DIR / "standardized_data.jsonl"
    output.write_text(
        "".join(json.dumps(task, ensure_ascii=False) + "\n" for task in tasks),
        encoding="utf-8",
    )
    smoke = [task["task_id"] for task in tasks[:1]]
    (OUT_DIR / "smoke_whitelist.json").write_text(
        json.dumps(smoke, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {len(tasks)} rank tasks -> {output}")
    print(f"pool size: {len(pool)}; dates: {[task['metadata']['entry_date'] for task in tasks]}")


if __name__ == "__main__":
    main()
