#!/usr/bin/env python3
"""Generate monthly OPEN-UNIVERSE A-share trader tasks (whole market).

Reuses the exact 12 monthly windows of the legacy 16-stock trader benchmark
(entry/exit dates, fees, horizon) but removes the hand-picked pool: the agent
may buy ANY stock that actually traded at the entry close, is not currently
named ST/退, and was not already delisted at entry.  That tradable set
(~5,000 codes) is embedded in task metadata as stock_pool so the framework's
deterministic judge and the repair loop keep working unchanged.

Known PIT caveat: stock_basic names are current (2026) names, so the ST flag
is not historical; this slightly over- and under-excludes a few names in both
directions and is documented as a benchmark limitation.

Requires the full-market SQLite mirror (scripts/ashare/fetch_full_market.py).
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DB_PATH = ROOT / "data" / "ashare_pools.db"
LEGACY_TASKS = ROOT / "data" / "ashare_trader" / "standardized_data.jsonl"
DEFAULT_VERSION = "v2"
SUPPORTED_VERSIONS = ("v2", "v3", "v4")
DEFAULT_OUT_DIRS = {
    version: ROOT / "data" / f"ashare_trader_open_{version}"
    for version in SUPPORTED_VERSIONS
}
BREADTH_WINDOW = 20
CASH_BANDS = {
    "risk_on": (0.10, 0.30),
    "neutral": (0.20, 0.40),
    "defensive": (0.35, 0.55),
}
V4_FIXED_CORE_WEIGHTS = {
    "510300.SH": 0.20,
    "510500.SH": 0.20,
    "512100.SH": 0.20,
}
V4_ALPHA_COUNT = 3
V4_ALPHA_WEIGHT = 0.10
V4_CASH_WEIGHT = 0.10

# v2 template: the tool menu is deliberately neutral (no momentum-first
# suggestion), adds the empirically observed A-share short-horizon reversal
# warning, and repeats the literal plain-text boxed format to avoid the
# LaTeX-corrupted final answers seen in the v1 run.
QUESTION_TEMPLATE_V2 = """\
你是同一名持续管理资金的 A 股交易员。当前日期为 {as_of}（收盘后）。

你可以从【全部 A 股约 {pool_size} 只可交易股票】中自由选股，为同一个资金账户决定从 {as_of} 收盘买入、持有 {horizon} 个交易日后按收盘卖出的组合仓位。回测从 100 万元开始并逐期复利；目标是在控制回撤和交易成本的同时，提高相对沪深300指数（000300.SH）的净收益。

组合约束（务必遵守，系统将确定性复核）：
- 仅做多，不得做空或使用杠杆；每只股票权重必须在 0 到 0.25 之间；
- 可以持有现金；股票权重与 CASH 权重之和必须严格等于 1；CASH 必须显式写出；
- 建议持有 4-10 只股票；只能买入 {as_of} 当日实际有成交、非 ST、且未退市的股票；
- 本任务按买入费率 0.05%、卖出费率 0.15%、每笔最低 5 元评价，{horizon} 个交易日后全部卖出。

数据使用规则（务必遵守）：
- 只能使用 ashare-open 市场工具，禁止网络搜索；所有工具都以 {as_of} 为点时截断，禁止推测或使用 {as_of} 之后的行情、新闻、财报或真实收益；
- 筛选口径：ashare_screen_market 支持按 pe_ttm、pb、total_mv、amount、turnover_rate、momentum、rel_momentum 排序（窗口 5-250，可升序、可按行业过滤），不预设任何单一口径更优；个股用 ashare_price_history、ashare_valuation、ashare_financials 深挖；用 ashare_market_breadth 与 ashare_index_history 判断市场环境；
- 经验警示：A 股在约 20 个交易日的持有期上，近期涨幅最大的一组股票历史上存在显著短期反转，机械追买全市场 20 日动量领涨股的组合大幅跑输指数；近 20 日涨幅超过 50% 的垂直拉升股，追高买入的期望收益极差；
- 纪律要求：候选股需经估值、基本面、流动性与市场广度交叉验证，单一信号不足以建仓；日均成交额过小的股票按收盘价难以成交，必须检查流动性；没有足够把握时，提高 CASH 权重是合理选择；
- 必须综合全市场相对机会、风险与信号一致性后再定仓位与现金比例。

输出要求：
- 正文简要说明筛选路径、入选理由、主要风险和现金决策；
- 最后一行严格输出 \\boxed{{代码:权重,代码:权重,...,CASH:权重}}；
- boxed 内容必须是纯文本的「代码:权重」逗号分隔列表：禁止 LaTeX 命令（\\text、\\mathrm、\\; 等）、禁止 w= 写法、禁止百分号、禁止解释文字；
- 权重使用 0 到 1 的小数且总和严格等于 1，例如 \\boxed{{600519.SH:0.20,300750.SZ:0.15,CASH:0.65}}。
"""

QUESTION_TEMPLATE_V3 = """\
你是同一名持续管理资金的 A 股交易员。当前日期为 {as_of}（收盘后）。

你可以从【全部 A 股约 {pool_size} 只可交易股票】中自由选股，为同一个资金账户决定从 {as_of} 收盘买入、持有 {horizon} 个交易日后按收盘卖出的组合仓位。回测从 100 万元开始并逐期复利；目标是在控制回撤和交易成本的同时，提高相对沪深300指数（000300.SH）的净收益。

组合硬约束（系统将确定性复核）：
- 仅做多，不得做空或使用杠杆；必须持有 4-8 只股票，每只有效持仓权重必须在 0.05 到 0.25 之间；
- 当前点时全市场近20日正收益占比为 {breadth_pct:.1f}%，简易状态为 {regime}；CASH 权重必须在 {cash_min:.2f} 到 {cash_max:.2f} 之间；
- 股票权重与 CASH 权重之和必须严格等于 1；CASH 必须显式写出；
- 只能买入 {as_of} 当日实际有成交、非 ST、且未退市的股票；
- 本任务按买入费率 0.05%、卖出费率 0.15%、每笔最低 5 元评价，{horizon} 个交易日后全部卖出。

组合结构与交易纪律：
- 先调用 ashare_market_breadth 和 ashare_index_history 复核市场状态；简易状态只决定仓位区间，不能代替选股判断；
- 建立“价值/质量核心 + 非抛物线卫星”：核心占股票仓位 60%-80%，选择 3-5 只流动性充足、估值可解释、盈利质量可核实的股票，尽量覆盖至少 3 个行业；不得机械照抄最低 PE，必须排除一次性收益、盈利恶化等价值陷阱；
- 卫星占股票仓位 20%-40%，选择 1-3 只有相对强度但未垂直拉升、且有估值或基本面确认的股票；20 日涨幅超过 50% 的标的原则上不得追买；
- risk_on 也不能追涨，defensive 也不能只持现金；目标是相对沪深300取得超额收益，因此应在风险区间内保持足够、分散的股票敞口；
- 候选股必须经估值、财务、流动性、价格结构与市场广度交叉验证；单一动量、单一低 PE 或单一题材均不足以建仓。

数据使用规则（务必遵守）：
- 只能使用 ashare-open 市场工具，禁止网络搜索；所有工具都以 {as_of} 为点时截断，禁止推测或使用 {as_of} 之后的行情、新闻、财报或真实收益；
- 必须至少成功使用 ashare_market_breadth、ashare_index_history、ashare_screen_market、ashare_financials；先看市场，再按低估值/高流动性/相对强度等不同口径构建候选，最后深挖个股；
- A 股在约 20 个交易日的持有期上存在显著短期反转风险；机械追买全市场短期涨幅最大的股票不是可接受策略；
- 日均成交额过小的股票按收盘价难以成交，必须检查流动性。

输出要求：
- 正文简要列明市场状态、核心与卫星划分、交叉验证、主要风险和现金决策；
- 最后一行严格输出 \\boxed{{代码:权重,代码:权重,...,CASH:权重}}；
- boxed 内容必须是纯文本的“代码:权重”逗号分隔列表：禁止 LaTeX 命令（\\text、\\mathrm、\\; 等）、禁止 w= 写法、禁止百分号、禁止解释文字；
- 权重使用 0 到 1 的小数且总和严格等于 1，例如 \\boxed{{600519.SH:0.20,300750.SZ:0.15,600036.SH:0.15,601318.SH:0.10,CASH:0.40}}。
"""

QUESTION_TEMPLATE_V4 = """\
你是同一名持续管理资金的 A 股交易员。当前日期为 {as_of}（收盘后）。

本策略采用不可更改的“宽基 ETF 核心 + 成长质量 Alpha”结构：
- 固定核心：510300.SH 沪深300ETF 20%、510500.SH 中证500ETF 20%、512100.SH 中证1000ETF 20%；
- 固定现金：CASH 10%；
- 你只负责从【全部 A 股约 {pool_size} 只可交易股票】中选择恰好 3 只 Alpha 股票，每只固定 10%；
- 所有仓位从 {as_of} 收盘买入，持有 {horizon} 个交易日后按收盘卖出。回测从 100 万元逐期复利，按 100 股/份整手成交，并计买入 0.05%、卖出 0.15%、每笔最低 5 元。

Alpha 硬纪律（必须全部遵守）：
- 只能选择 {as_of} 当日实际有成交、非 ST、未退市的股票；ETF 不能作为 Alpha；
- 先排除近20日日均成交额低于 200000 千元、PE_TTM 非正或高于 60、近20日涨幅超过 50% 或低于 -20% 的股票；
- 排除最新已公告财务中营收与净利润同时恶化的股票；财务数据必须按 ann_date<={as_of} 截断；
- 三只 Alpha 尽量覆盖至少两个行业，禁止仅凭题材、单一低 PE 或单一短期动量建仓。

必须执行的点时筛选路径：
1. 调用 ashare_market_breadth 和 ashare_index_history 了解环境，但不得据此改动固定 ETF/现金权重；当前预计算近20日正收益占比 {breadth_pct:.1f}%，状态 {regime}；
2. 至少用 ashare_screen_market 的流动性、适度相对强度等不同口径构建候选；使用 min_pe_ttm=5、max_pe_ttm=60、min_window_return=-20、max_window_return=50 约束追高与亏损股；
3. 将 5-20 只候选一次性传给 ashare_compare_growth_quality，并只从硬过滤为 YES 的候选中选择最终三只；
4. 优先选择“已公告营收/净利增长或加速 + ROE/毛利率确认 + PE 可解释或利润驱动的 PE 压缩 + 高流动性 + 非极端相对强度”交叉成立的股票。

输出要求：
- 正文简要列出候选构建、成长质量交叉验证、淘汰原因和三只入选股；
- 最后一行必须完整且严格输出：
  \\boxed{{510300.SH:0.20,510500.SH:0.20,512100.SH:0.20,股票1:0.10,股票2:0.10,股票3:0.10,CASH:0.10}}
- boxed 内容只能是纯文本“代码:权重”逗号分隔列表；禁止 LaTeX 命令、w=、百分号或解释文字。
"""

QUESTION_TEMPLATES = {
    "v2": QUESTION_TEMPLATE_V2,
    "v3": QUESTION_TEMPLATE_V3,
    "v4": QUESTION_TEMPLATE_V4,
}


def tradable_pool(conn: sqlite3.Connection, entry_date: str) -> list[str]:
    rows = conn.execute(
        """
        SELECT d.ts_code FROM market_daily d
        JOIN stock_basic_all b USING(ts_code)
        WHERE d.trade_date = ?
          AND b.name NOT LIKE '%ST%'
          AND b.name NOT LIKE '%退%'
          AND (b.delist_date IS NULL OR b.delist_date = '' OR b.delist_date > ?)
        ORDER BY d.ts_code
        """,
        (entry_date, entry_date),
    ).fetchall()
    return [r[0] for r in rows]


def market_breadth_regime(
    conn: sqlite3.Connection, entry_date: str
) -> tuple[float, str]:
    dates = [
        row[0]
        for row in conn.execute(
            "SELECT cal_date FROM trade_cal "
            "WHERE is_open=1 AND cal_date<=? ORDER BY cal_date DESC LIMIT ?",
            (entry_date, BREADTH_WINDOW + 1),
        )
    ][::-1]
    if len(dates) < BREADTH_WINDOW + 1:
        raise ValueError(f"not enough breadth history at {entry_date}")
    placeholders = ",".join("?" * BREADTH_WINDOW)
    cumulative: dict[str, float] = {}
    for code, pct_chg in conn.execute(
        f"SELECT ts_code, pct_chg FROM market_daily "
        f"WHERE trade_date IN ({placeholders}) ORDER BY ts_code, trade_date",
        dates[1:],
    ):
        cumulative[code] = cumulative.get(code, 1.0) * (
            1.0 + float(pct_chg or 0.0) / 100.0
        )
    if not cumulative:
        raise ValueError(f"empty breadth sample at {entry_date}")
    positive_share = sum(nav > 1.0 for nav in cumulative.values()) / len(cumulative)
    regime = (
        "risk_on"
        if positive_share > 0.60
        else "defensive"
        if positive_share < 0.40
        else "neutral"
    )
    return positive_share * 100.0, regime


def build_tasks(version: str = DEFAULT_VERSION) -> list[dict[str, Any]]:
    if version not in QUESTION_TEMPLATES:
        raise ValueError(f"unsupported strategy version: {version}")
    question_template = QUESTION_TEMPLATES[version]
    legacy = [
        json.loads(line)
        for line in LEGACY_TASKS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.execute("PRAGMA busy_timeout=30000")
    tasks: list[dict[str, Any]] = []
    try:
        for old in legacy:
            meta = old["metadata"]
            entry, exit_ = meta["entry_date"], meta["exit_date"]
            pool = tradable_pool(conn, entry)
            if len(pool) < 1000:
                raise ValueError(
                    f"tradable pool suspiciously small at {entry}: {len(pool)}"
                )
            as_of = meta["as_of"]
            breadth_pct, regime = market_breadth_regime(conn, entry)
            cash_min, cash_max = CASH_BANDS[regime]
            version_metadata: dict[str, Any] = {}
            if version == "v3":
                version_metadata = {
                    "strategy_version": "v3",
                    "market_regime": regime,
                    "positive_ret20_share": round(breadth_pct / 100.0, 6),
                    "min_holdings": 4,
                    "max_holdings": 8,
                    "min_active_stock_weight": 0.05,
                    "min_cash_weight": cash_min,
                    "max_cash_weight": cash_max,
                    "required_tools": [
                        "ashare_market_breadth",
                        "ashare_index_history",
                        "ashare_screen_market",
                        "ashare_financials",
                    ],
                }
            elif version == "v4":
                version_metadata = {
                    "strategy_version": "v4",
                    "market_regime": regime,
                    "positive_ret20_share": round(breadth_pct / 100.0, 6),
                    "asset_pool": [*V4_FIXED_CORE_WEIGHTS, *pool],
                    "fixed_core_weights": dict(V4_FIXED_CORE_WEIGHTS),
                    "alpha_count": V4_ALPHA_COUNT,
                    "alpha_weight": V4_ALPHA_WEIGHT,
                    "cash_weight": V4_CASH_WEIGHT,
                    "min_holdings": 6,
                    "max_holdings": 6,
                    "min_active_stock_weight": V4_ALPHA_WEIGHT,
                    "min_cash_weight": V4_CASH_WEIGHT,
                    "max_cash_weight": V4_CASH_WEIGHT,
                    "required_tools": [
                        "ashare_market_breadth",
                        "ashare_index_history",
                        "ashare_screen_market",
                        "ashare_compare_growth_quality",
                    ],
                }
            tasks.append(
                {
                    "task_id": f"ashare_open_trader_{as_of}",
                    "task_question": question_template.format(
                        as_of=as_of,
                        horizon=meta["horizon_trading_days"],
                        pool_size=len(pool),
                        breadth_pct=breadth_pct,
                        regime=regime,
                        cash_min=cash_min,
                        cash_max=cash_max,
                    ),
                    "ground_truth": "VALID_PORTFOLIO",
                    "file_path": None,
                    "metadata": {
                        "dataset_name": "ashare-trader-open",
                        "task_type": "portfolio_allocation",
                        "as_of": as_of,
                        "scheduled_month": meta["scheduled_month"],
                        "planned_entry_date": meta["planned_entry_date"],
                        "entry_date": entry,
                        "exit_date": exit_,
                        "entry_shift_sessions": meta["entry_shift_sessions"],
                        "horizon_trading_days": meta["horizon_trading_days"],
                        "index_code": meta["index_code"],
                        "index_return": meta["index_return"],
                        "universe": "all_ashare",
                        "pool_size": len(pool),
                        "stock_pool": pool,
                        "max_stock_weight": 0.25,
                        "allow_cash": True,
                        "allow_short": False,
                        "open_cost": 0.0005,
                        "close_cost": 0.0015,
                        "min_cost": 5.0,
                        "initial_capital": 1_000_000.0,
                        **version_metadata,
                    },
                }
            )
    finally:
        conn.close()
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--version",
        choices=SUPPORTED_VERSIONS,
        default=DEFAULT_VERSION,
        help="prompt/constraint version (default: v2)",
    )
    parser.add_argument(
        "--out",
        default="",
        help="output directory (default: data/ashare_trader_open_<version>)",
    )
    args = parser.parse_args()
    out_dir = Path(args.out) if args.out else DEFAULT_OUT_DIRS[args.version]
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = build_tasks(args.version)
    output = out_dir / "standardized_data.jsonl"
    output.write_text(
        "".join(json.dumps(t, ensure_ascii=False) + "\n" for t in tasks),
        encoding="utf-8",
    )
    (out_dir / "smoke_whitelist.json").write_text(
        json.dumps([tasks[0]["task_id"]], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    sizes = [t["metadata"]["pool_size"] for t in tasks]
    print(f"wrote {len(tasks)} open trader {args.version} tasks -> {output}")
    print(f"tradable pool sizes: min={min(sizes)} max={max(sizes)}")


if __name__ == "__main__":
    main()
