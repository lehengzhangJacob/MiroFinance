---
name: ashare_cross_section_ranking
description: 全A股开放池的点时批量筛选与候选集排序，不依赖固定股票池或单一因子
version: "2.0"
applies_to: [open_universe_screening, candidate_ranking]
stock_universe: all_ashare_point_in_time
dependencies: [ashare_universe_stats, ashare_screen_market, ashare_compare_growth_quality]
triggers:
  - 全A股
  - 开放池
  - 横截面
  - 批量筛选
  - 候选排序
---

## 标准流程

1. **确认点时股票池**：用 `ashare_universe_stats(as_of)` 核对可交易数量；所有筛选、
   财务和估值数据必须截断到 `as_of`。
2. **建立多来源候选集**：分别调用 `ashare_screen_market` 查看估值、流动性、规模、
   换手和非极端趋势。每个口径只取少量候选并去重，不把任何一次排序直接当答案。
3. **过滤不可交易与极端样本**：剔除 ST/退市风险、历史不足、成交额不足、垂直拉升、
   估值无意义或数据缺失严重的股票。
4. **批量比较成长质量**：将不超过 20 只候选交给
   `ashare_compare_growth_quality(ts_codes, as_of)`，比较已公告增长、ROE、利润率、
   估值变化、流动性和非极端价格状态。
5. **形成候选排序**：先看硬过滤是否通过，再看证据覆盖和一致性；同分时优先财务
   公告更新、流动性更好、行业重复暴露更低的股票。排名只代表当前候选集的研究优先级。

## 纪律

- 不要求完整枚举数千只股票；工具负责全市场筛选，Agent 负责比较候选集。
- 不使用固定股票数量、固定行业或固定因子权重。
- 近期收益只能用于排除极端追高和识别价格状态，不作为强制动量锚。
- 不使用决策日之后的行情、公告、标签或事后排名。
