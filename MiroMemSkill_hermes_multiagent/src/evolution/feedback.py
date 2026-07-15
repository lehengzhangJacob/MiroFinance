# SPDX-FileCopyrightText: 2026 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

"""Structured mutation feedback built from settled replay outcomes.

The generator gets facts, not vibes: per-month realized returns versus the
index, portfolio shape (holdings, concentration, cash), execution problems
(missing/unfilled lots), and the exact allocations that were bought. Only
train-split months may be summarized here — the caller is responsible for
passing train-month data exclusively so dev/holdout never leak into mutation.
"""

from __future__ import annotations

import json
from pathlib import Path


def _month_lines(months: list[dict], allocations: dict) -> list[str]:
    lines: list[str] = []
    for month in months:
        as_of = month["as_of"]
        if "note" in month:
            lines.append(
                f"- {as_of}: 输出无效（{month['note']}），当月被强制 100% 现金。"
            )
            continue
        weights = allocations.get(as_of) or {}
        stock_weights = {
            k: v for k, v in weights.items() if k != "CASH" and v > 0
        }
        top = sorted(stock_weights.items(), key=lambda kv: -kv[1])[:3]
        top_txt = ",".join(f"{code}={w:.0%}" for code, w in top)
        max_w = max(stock_weights.values(), default=0.0)
        problems = []
        if month.get("missing"):
            problems.append(f"无法定价:{','.join(month['missing'])}")
        if month.get("unfilled"):
            problems.append(f"整手未成交:{','.join(month['unfilled'])}")
        problem_txt = f"；执行问题：{'；'.join(problems)}" if problems else ""
        lines.append(
            f"- {as_of}: 净收益 {month['net']*100:+.2f}%，指数 {month['index']*100:+.2f}%，"
            f"主动 {(month['net']-month['index'])*100:+.2f}%；"
            f"持仓 {month.get('holdings', 0)} 只，现金 {month.get('cash_w', 0.0):.0%}，"
            f"单股最大权重 {max_w:.0%}，前三重仓 {top_txt or '无'}{problem_txt}"
        )
    return lines


def build_feedback(
    arm: dict,
    allocations: dict,
    label: str = "baseline",
) -> str:
    """Render one arm's train-split outcomes as mutation feedback text."""
    months = arm["months"]
    lines = _month_lines(months, allocations)
    losing = [m for m in months if "note" not in m and float(m["net"]) < 0.0]
    lagging = [
        m
        for m in months
        if "note" not in m and float(m["net"]) < float(m["index"])
    ]
    header = (
        f"### {label} 在训练月份上的已结算表现（确定性回放，含费/整手）\n"
        f"累计收益 {arm['total_return']*100:+.2f}%，同期指数 {arm['index_return']*100:+.2f}%，"
        f"超额 {arm['excess_return']*100:+.2f}%；最大回撤 {arm['max_drawdown']*100:.2f}%，"
        f"最差月 {arm['worst_month']*100:+.2f}%，费用合计 {arm['fees']:.0f} 元。\n"
        f"亏损月 {len(losing)} 个；跑输指数月 {len(lagging)} 个。\n"
    )
    guidance = (
        "\n### 修改要求\n"
        "以上是执行现行 Skill 得到的事实结果。请针对亏损月和跑输月暴露的流程缺陷"
        "改写 Skill 正文：筛选路径、排除规则、行业分散、仓位与现金决策、最终复核。"
        "不得写入任何具体股票代码、具体日期或对未来行情的断言；"
        "不得修改 frontmatter；保持步骤可执行、与现有工具名一致。"
    )
    return header + "\n" + "\n".join(lines) + guidance


def load_arm_allocations(run_dir: str | Path) -> dict:
    """Parse each month's boxed allocation from an arm's attempt files."""
    from src.evolution.fitness import evaluator

    return evaluator().extract_run_allocations(Path(run_dir))


def save_feedback(run_dir: str | Path, text: str) -> Path:
    path = Path(run_dir) / "mutation_feedback.md"
    path.write_text(text, encoding="utf-8")
    return path


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
