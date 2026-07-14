# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

"""Prompts for the Mem0-style memory pipeline (extraction phase + update phase).

Extraction distills candidate lessons from a finished task; the update phase
compares each candidate against similar existing memories and decides
ADD / UPDATE / DELETE / NONE (Mem0's consolidation step). Both must return
strict JSON — the caller discards anything unparseable and never stores raw
LLM output.
"""

EXTRACTION_PROMPT = """你是一个金融研究 agent 的元学习助手，负责从已完成的任务中提炼可复用的策略经验。

{outcome_instructions}

硬性要求：
- 必须具体：写明涉及的指标 / 工具 / 阈值 / 步骤（例如「当20日与60日相对动量方向相反时，只有换手率同步放大才信任20日窗口」），禁止「综合考虑多因素」之类的空话。
- 教训必须写成条件式倾向（"当…时，…往往/概率上…"），不是保证成立的规则——单个任务的结果是噪声证据，对市场方向预测尤其如此。
- 不要复述任务协议（点时规则、boxed 输出格式）。
- 不要写入本任务的标准答案或具体涨跌数字。
- 如果本任务没有真正新的可学内容，返回 {{"lessons": []}}。

任务问题（截断）：
{question}

Agent 最终答案：
{answer}

判题结果：{judge_result}

轨迹摘要（截断）：
{trajectory}

只返回合法 JSON，不要任何其他文字：
{{
  "lessons": [
    {{"content": "...", "tags": ["strategy", "..."]}}
  ]
}}
"""

EXTRACTION_CORRECT_INSTRUCTIONS = """该任务 agent 回答正确。最多提炼 2 条可复用的策略经验（不是任务答案本身）。
聚焦决策过程：什么条件下该给哪类证据更高权重、工具使用技巧、计算陷阱。"""

EXTRACTION_INCORRECT_INSTRUCTIONS = """该任务 agent 预测错误。最多提炼 1 条「反事实教训」：agent 依赖的启发式在什么条件下会失效、当时应该检查哪个反证。

关键要求：不要把失败的推理反过来写成正向规则（例如 agent 基于弱动量预测跑输而错了，教训不能写成「弱动量意味着跑赢」——必须写成「仅凭弱动量预测跑输不可靠，当出现<本例中的具体反证条件>时」）。教训必须编码这条推理链失败的原因。"""


MONTHLY_REFLECTION_PROMPT = """你是一个 A 股量化研究 agent 的元学习助手。下面是 {month} 的月度截面复盘表：
池内 {n} 只股票在决策日（当月首个交易日收盘后）可见的特征，以及随后 20 个交易日相对沪深300的实际结果和 agent 当时的预测。

字段说明：rel5/rel20/rel60 = 决策日前5/20/60个交易日相对沪深300的超额收益%；pe_pct/pb_pct/turn_pct = PE(TTM)/PB/换手率的近120日分位%；ml_rank = qlib机器学习截面排名（1=最看多，共{n}只）；pred = agent当时的预测；correct = 预测是否正确；label = 实际结果。

{table}

任务：从这个 n={n} 的横截面提炼至多 2 条「区分本月赢家与输家」的条件模式。

硬性要求：
- 只用表中决策日可见的特征构造条件（rel*/pe_pct/pb_pct/turn_pct/ml_rank），绝不能用 label/pred/correct 作为条件。
- 模式必须真的在表中成立：条件至少覆盖 4 只股票，且其中方向一致率 >= 75%。写出教训时给出观察到的大致比例（如「6只中5只跑赢」）。
- 教训写成条件式倾向（"当…时，跑赢/跑输概率偏高（本月截面 x/y）"），并保留"单月截面证据"的措辞，禁止绝对化。
- 如果本月截面没有清晰模式（特征与结果基本无关），必须返回 {{"lessons": []}}，宁缺毋滥。
- 不要复述任务协议，不要提任何个股代码未来的方向。

只返回合法 JSON，不要任何其他文字：
{{"lessons": [{{"content": "...", "tags": ["monthly", "..."]}}]}}
"""


UPDATE_PROMPT = """你是记忆库管理器。给定一条新提炼的候选教训和记忆库中与之最相似的已有记忆，决定如何整合。

候选教训：
{candidate}

相似的已有记忆（id 与内容）：
{existing}

决策规则：
- "ADD"：候选包含已有记忆都没有覆盖的新条件模式 → 作为新记忆加入。
- "UPDATE"：候选与某条已有记忆描述同一模式（相同指标组合/相似条件），但补充了新的条件、边界或反例 → 把两者合并成一条更完整的教训写入 new_content（保留双方有效信息，条件式表述）。
- "DELETE"：候选提供的新证据直接反驳某条已有记忆，且该记忆已无保留价值（例如它断言的模式被证明系统性失效）→ 删除该条并把候选作为新记忆加入。
- "NONE"：候选与已有记忆实质重复，没有新增信息 → 什么都不做。

注意：
- UPDATE / DELETE 必须给出 target_id（从上面的已有记忆中选）。
- 合并后的 new_content 仍必须是条件式倾向表述，禁止绝对化。
- 宁可 NONE 也不要为微小差异新增条目——记忆库要精不要多。

只返回合法 JSON，不要任何其他文字：
{{"action": "ADD|UPDATE|DELETE|NONE", "target_id": "...", "new_content": "..."}}
（action 为 ADD/NONE 时 target_id 与 new_content 可为空字符串；DELETE 时 new_content 填候选教训原文）
"""
