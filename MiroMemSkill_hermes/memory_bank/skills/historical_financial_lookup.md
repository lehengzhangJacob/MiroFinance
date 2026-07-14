---
name: historical_financial_lookup
description: 历史金融数据单点查询——权威源优先级、日期核对与单位规范
version: "2.0"
applies_to: [historical_financial_lookup, point_in_time_research]
stock_universe: global_financial_data
dependencies: [search_tool, document_reader]
triggers:
  - 历史
  - historical
  - 截至
  - as of
  - 十亿美元
  - billion
  - 亿元
  - 季度
  - 年报
---

## 步骤

1. **解析题目约束**：提取时间点(as-of date)、数据口径(seasonally adjusted / 非季调)、单位(十亿美元/亿元/%)、小数位数。
2. **权威源优先级**：
   - 全球宏观/央行：FRED、BIS、IMF、各国央行官网、SEC EDGAR
   - 中国：国家统计局、人民银行、外汇管理局、Wind/东方财富公告
   - 公司财务：年报/季报 PDF、投资者关系页面
3. **日期核对**：确认数据发布日期覆盖题目要求的 as-of 时点；历史数据用 Wayback 或官方 archive。
4. **单位归一**：回答前统一单位；允许题目声明的 rounding error(如 ±1%)。
5. **输出格式**：只输出最终数值+单位，不写冗长解释；数值格式与标准答案一致。

## 常见陷阱

- 混淆 seasonally adjusted vs not seasonally adjusted
- 用了最新页面数据而非题目指定历史时点
- 单位差 10 倍(百万 vs 十亿)
