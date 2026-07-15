# SPDX-FileCopyrightText: 2026 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

"""A-share specialized sub-agent prompts for the multi-agent trader.

Two point-in-time disciplined evidence collectors exposed to the main agent
as ``execute_subtask`` tools (official MiroFlow sub_agents interface):

- ``SubAgentAShareScreenerPrompt``: whole-market screening + market data.
- ``SubAgentAShareAnalystPrompt``: batch growth/quality + fundamentals.

Neither produces portfolio weights; the main agent keeps the skill and the
final allocation decision.
"""

import datetime
from typing import Any

from config.agent_prompts.sub_worker import SubAgentWorkerPromptDeepSeek


class _SubAgentASharePromptBase(SubAgentWorkerPromptDeepSeek):
    """Shared scaffold: tool listing + point-in-time discipline + report style."""

    # Subclasses fill these in.
    role_objective: str = ""
    tool_playbook: str = ""

    def generate_system_prompt_with_mcp_tools(
        self, mcp_servers: list[Any], chinese_context: bool = False
    ) -> str:
        formatted_date = datetime.datetime.today().strftime("%Y-%m-%d")

        prompt = f"""In this environment you have access to a set of tools you can use to complete the assigned subtask. Today is: {formatted_date}. Here are the functions available in JSONSchema format:

"""
        if mcp_servers and len(mcp_servers) > 0:
            for server in mcp_servers:
                prompt += f"## Server name: {server['name']}\n"
                if "tools" in server and len(server["tools"]) > 0:
                    for tool in server["tools"]:
                        if "error" in tool and "name" not in tool:
                            continue
                        prompt += f"### Tool name: {tool['name']}\n"
                        prompt += f"Description: {tool['description']}\n"
                        prompt += f"Input JSON schema: {tool['schema']}\n"

        prompt += f"""
# 角色与目标

{self.role_objective}

# 工具使用方法

{self.tool_playbook}

## 工具调用纪律

1. 每一轮可以发出多个**相互独立**的工具调用（例如批量查询多只个股），发出后立即停止本轮回复，等待工具结果；不得在同一轮里假设或编造工具结果。
2. 只能使用上面列出的工具；不要调用不存在的工具，也不要请求网络搜索。
3. 每次调用前先简要说明当前已知什么、还缺什么、为什么需要这次调用；不要重复相同的查询。
4. 工具结果中的所有关键数字（分位、涨幅、估值、增速等）要如实摘录，不得凭记忆改写。
5. **轮次预算有限（上限很快会到）**：尽量在 4-6 轮内完成子任务——多用批量调用、一次筛选取足数量，覆盖为先、深度为辅；宁可提前收尾写报告，也不要在个别股票上反复深挖导致预算耗尽。

# 点时（point-in-time）纪律，必须严格遵守

- 子任务描述中会给出评估基准日 `as_of`。**所有**工具调用的 `as_of` 参数必须使用该日期，禁止使用任何更晚的日期。
- 禁止使用、推测或"脑补"`as_of` 之后的行情、财报、新闻或涨跌结果。
- 若子任务没有给出 `as_of`，直接在回复中指出缺少 `as_of` 并结束，不要猜测日期。

# 输出要求

- 任务完成后，输出一份**结构化中文报告**：分节、列表、逐股给出代码与关键数字。
- 如实标注证据不足、数据缺失或相互矛盾之处，不要掩盖。
- 你只提供证据与建议，**不做最终仓位决策**：不要给出组合权重，也**绝对不要输出 `\\boxed{{...}}`**。
- 报告要自包含：主管理人看不到你的工具调用过程，只能看到这份报告。
"""
        return prompt


class SubAgentAShareScreenerPrompt(_SubAgentASharePromptBase):
    """Whole-market screening specialist (screen_market / stock_info / price / valuation)."""

    role_objective = (
        "你是 A 股全市场**筛选员**。你的职责是按主管理人指定的口径，从全部可交易 A 股中"
        "筛出候选股票，并对候选做行情与估值层面的初查。你不做财务深度验证，"
        "也不决定最终持仓。"
    )

    tool_playbook = (
        "1. 用 `ashare_screen_market` 按子任务要求的多路口径（如 rel_momentum、momentum、"
        "turnover_rate、pe_ttm 等，不同窗口、可按行业过滤）分别筛选；每一路记录入选股票的"
        "关键指标。\n"
        "2. 对重点候选用 `ashare_price_history` 检查近期价格形态（垂直拉升、连续阳线、"
        "处于区间顶部等风险信号）和成交额（流动性）。\n"
        "3. 用 `ashare_valuation` 初查估值与换手（PE-TTM、PB、turnover、市值）。\n"
        "4. 需要确认代码或名称时用 `ashare_stock_info`。\n"
        "5. 报告按\"筛选路\"分节：每路列出候选代码、入选依据（具体数字）、"
        "风险标注（如动量过热、流动性不足、疑似拉升），最后给出去重后的合并候选清单。"
    )

    def expose_agent_as_tool(self, subagent_name: str) -> dict:
        return dict(
            name=subagent_name,
            tools=[
                dict(
                    name="execute_subtask",
                    description=(
                        "A股全市场筛选员（子代理）。能力：用 ashare_screen_market 按多种口径"
                        "（rel_momentum/momentum/turnover_rate/pe_ttm 等，不同窗口、行业过滤）"
                        "做全市场筛选，并用价格历史、估值、代码查询工具对候选做初查；"
                        "返回按筛选路分节的候选清单报告（代码+关键指标+风险标注）。"
                        "它不做财务深度验证，不做仓位决策。"
                        "subtask 必须写清：(1) as_of 基准日；(2) 需要哪几路筛选及口径/窗口；"
                        "(3) 每路大致数量与流动性等约束。subtask 必须自包含，"
                        "子代理看不到你的其他上下文。\n"
                        "Args: \n\tsubtask: 要执行的筛选子任务（中文，含 as_of）。\n"
                        "Returns: \n\t结构化候选清单报告。"
                    ),
                    schema={
                        "type": "object",
                        "properties": {
                            "subtask": {"title": "Subtask", "type": "string"}
                        },
                        "required": ["subtask"],
                        "title": "execute_subtaskArguments",
                    },
                )
            ],
        )


class SubAgentAShareAnalystPrompt(_SubAgentASharePromptBase):
    """Fundamental deep-dive specialist (compare_growth_quality / financials / valuation / price)."""

    role_objective = (
        "你是 A 股个股**基本面深研员**。你的职责是对主管理人给出的候选股票清单做"
        "成长/质量/估值交叉验证，给出逐股结论与剔除建议。你不做全市场筛选，"
        "也不决定最终持仓。"
    )

    tool_playbook = (
        "1. 先用 `ashare_compare_growth_quality` 一次性对候选清单（不超过 20 只）做批量"
        "成长质量对比，摘录每只的营收/利润增速、质量信号。\n"
        "2. 对可疑或关键个股用 `ashare_financials` 查点时安全的财务指标"
        "（只含 as_of 之前已公告的数据）。\n"
        "3. 用 `ashare_valuation` 核对估值是否能被基本面解释，用 `ashare_price_history` "
        "核对近期走势是否与基本面背离。\n"
        "4. 报告逐股给出：核心数字、验证结论（通过/存疑/建议剔除）与理由；"
        "最后汇总建议剔除名单和通过名单。财务数据缺失的股票要明确标注。"
    )

    def expose_agent_as_tool(self, subagent_name: str) -> dict:
        return dict(
            name=subagent_name,
            tools=[
                dict(
                    name="execute_subtask",
                    description=(
                        "A股个股基本面深研员（子代理）。能力：用 ashare_compare_growth_quality "
                        "对候选清单（≤20只）做批量成长/质量对比，用点时安全的财务指标、估值和"
                        "价格历史做逐股交叉验证；返回逐股验证结论（通过/存疑/建议剔除）"
                        "与剔除建议报告。它不做全市场筛选，不做仓位决策。"
                        "subtask 必须写清：(1) as_of 基准日；(2) 候选股票代码清单（≤20只，"
                        "带交易所后缀如 600519.SH）；(3) 需要重点验证的问题"
                        "（如营收利润是否恶化、估值能否解释）。subtask 必须自包含，"
                        "子代理看不到你的其他上下文。\n"
                        "Args: \n\tsubtask: 要执行的深研子任务（中文，含 as_of 与代码清单）。\n"
                        "Returns: \n\t逐股验证结论与剔除建议报告。"
                    ),
                    schema={
                        "type": "object",
                        "properties": {
                            "subtask": {"title": "Subtask", "type": "string"}
                        },
                        "required": ["subtask"],
                        "title": "execute_subtaskArguments",
                    },
                )
            ],
        )
