---
name: ashare_valuation_fundamentals
description: A股估值与已公告基本面的点时交叉验证，识别成长质量、估值匹配和价值陷阱
version: "2.0"
applies_to: [open_universe_screening, portfolio_research, security_validation]
stock_universe: all_ashare_point_in_time
dependencies: [ashare_valuation, ashare_financials, ashare_compare_growth_quality]
triggers:
  - 估值
  - PE
  - PB
  - 财务
  - 财报
  - 基本面
  - 成长质量
---

## 步骤

1. **先检查可比口径**：金融、公用事业、周期和成长行业的 PE/PB 含义不同；亏损、
   一次性收益或强周期顶部会让低 PE 失真。
2. **坚持公告日点时纪律**：只使用 `ann_date <= as_of` 的财务数据，同时记录报告期，
   不把尚未公告的数据或当前已知结果带回历史决策。
3. **比较增长质量**：联合观察营收和净利增速、ROE、毛利率/净利率、增速变化与利润率
   变化。利润增长但营收、利润率和现金创造能力同步恶化时，降低可信度。
4. **检查估值与增长匹配**：增长加速且估值未同步扩张优于单纯最低 PE；低估值伴随
   营收、利润持续恶化时视为价值陷阱候选。
5. **结合价格与流动性**：20 日任务中估值是慢变量，不设置固定权重。近期垂直拉升是
   风险证据，温和趋势或止跌只是确认信号；成交额不足直接影响可实现性。
6. **输出条件化结论**：明确支持建仓、需要复核和否决证据，不把某个估值阈值机械
   转换为仓位。

## 常见陷阱

- 跨行业直接比较 PE 绝对值；
- 把低 PE 当作短期上涨保证；
- 引用公告日晚于 `as_of` 的财务数据；
- 用固定“动量/估值权重”代替当前证据判断。
