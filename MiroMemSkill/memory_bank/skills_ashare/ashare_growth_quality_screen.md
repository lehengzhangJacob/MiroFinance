---
name: ashare_growth_quality_screen
description: 全A股成长质量候选筛选，联合已公告增长、盈利质量、估值、流动性和非极端价格状态
version: "1.0"
applies_to: [open_universe_screening, growth_quality_research]
stock_universe: all_ashare_point_in_time
dependencies: [ashare_screen_market, ashare_compare_growth_quality, ashare_financials, ashare_valuation]
triggers:
  - 成长质量
  - 成长股
  - 质量筛选
  - 盈利增长
  - 全市场筛选
  - growth quality
---

## 筛选流程

1. **生成候选而非直接下结论**：用 `ashare_screen_market` 从合理 PE/PB、高流动性、
   适中规模和非极端价格状态中各取少量股票，去重后保留不超过 20 只。
2. **批量比较**：调用 `ashare_compare_growth_quality(ts_codes, as_of)`。硬过滤优先于
   规则分；规则分只是可解释的研究排序，不是未来收益预测。
3. **验证增长来源**：检查营收、净利、ROE、毛利率、利润率和增速变化。利润增长若
   主要来自一次性项目、利润率恶化或缺少收入支持，降低质量判断。
4. **验证估值匹配**：优先“增长改善且估值未失控”，避免把最低 PE 等同于质量，
   也避免为高增长叙事忽略估值扩张。
5. **检查可交易性与价格状态**：成交额必须支持计划仓位；近期垂直拉升是短期反转
   风险。温和相对强度可以确认，但不是必须持有的动量锚。
6. **保留反证**：对每只入围股票记录至少一条主要风险；证据缺失与证据为负要分开处理。

## 点时纪律

- 财务必须满足 `ann_date <= as_of`；
- 行情和估值只能使用 `trade_date <= as_of`；
- 不使用持有期真实收益、事后排名或当前已知公司结局；
- 跨行业比较时使用行业适配指标，不机械比较 PE 绝对值。
