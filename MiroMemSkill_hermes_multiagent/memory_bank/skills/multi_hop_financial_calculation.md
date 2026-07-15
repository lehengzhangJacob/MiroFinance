---
name: multi_hop_financial_calculation
description: 多跳金融计算——先分解子问题、逐步聚合、单位与舍入规范
version: "2.0"
applies_to: [financial_calculation, multi_entity_aggregation]
stock_universe: not_applicable
dependencies: [search_tool, code_tool]
triggers:
  - 计算
  - 平均
  - 合计
  - 增长率
  - 个月
  - 年
  - 差额
  - difference
  - ratio
  - 复杂
  - T3
---

## 步骤

1. **分解子问题**：列出所有需要独立检索/计算的中间量(如 N 个实体各一个数值)。
2. **逐步收集**：每个子问题单独搜索验证，记录来源；禁止跳步心算。
3. **统一单位**：全部换算到题目要求的单位后再运算。
4. **执行计算**：用 code 工具做精确运算；展示中间结果便于核对。
5. **舍入规则**：严格按题目要求的小数位/整数位；注意"四舍五入"与"允许误差"说明。
6. **完整性检查**：多要点题必须全部答出才得分；缺一项即 0 分。

## 常见陷阱

- 只算了部分实体就取平均
- 时间跨度算错(月 vs 年)
- 最终答案格式不符(如要求整数却给小数)
