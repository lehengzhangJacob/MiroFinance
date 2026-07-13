---
name: ashare_prediction_protocol
description: 单只A股相对沪深300跑赢/跑输二分类流程；不适用于组合构建
version: "2.0"
applies_to: [binary_excess_return_prediction]
stock_universe: task_provided_security
dependencies: [ashare_price_history, ashare_index_history, ashare_valuation, ashare_financials]
triggers:
  - 预测
  - 单只A股
  - 跑赢
  - 跑输
  - 二分类
  - 量化研究员
  - 超额收益
  - 沪深300
  - boxed
  - 交易日
---

## 标准流程（每题必做）

本技能只用于最终答案为“跑赢/跑输”的单股二分类任务。组合选股、仓位和 CASH 决策
应使用 `ashare_open_portfolio`，不要套用本技能的二分类输出格式。

1. **工具调用清单**（as_of 一律用题目给定日期，格式 YYYYMMDD）：
   - ashare_price_history(ts_code, as_of, lookback_days=250) → 个股动量（覆盖约 1 年交易日）
   - ashare_index_history(as_of, lookback_days=250) → 指数动量
   - ashare_valuation(ts_code, as_of, lookback_days=250) → 估值分位
   - ashare_financials(ts_code, as_of) → 最近已披露财报
   - 若任务环境明确提供经过严格 walk-forward 验证的 `ashare_ml_signal`，可把它作为一条独立证据；没有可靠全市场模型时不得强制使用。
2. **证据打分**：把每条证据标注为 [+看多超额] / [-看空超额] / [0中性]，明确主要矛盾；不预设固定因子权重，近期极端上涨应作为反转风险而不是自动看多。
3. **禁止事项**：
   - 不使用任何工具之外的市场记忆（你可能记得该股票后来的走势——**必须当作不知道**）；
   - 不预测指数或个股的绝对涨跌，只判断**相对沪深300 的超额方向**。
4. **输出硬约束**：
   - 正文给 3-5 条核心依据（每条一行，标注 +/-/0）；
   - 最后一行**只写** `\boxed{跑赢}` 或 `\boxed{跑输}`，不得写"可能/大概率"等模糊词，不得同时给两个答案；
   - 不输出置信区间、免责声明等冗余内容。

## 常见陷阱

- 最终行写成 `\boxed{跑赢沪深300}`、`\boxed{outperform}` 等变体（判分按「跑赢/跑输」精确语义）
- 花大量轮次反复拉同一数据（每个工具一次调用足够，max_turns 有限）
- 因证据不足而拒绝给结论（本任务必须二选一，宁可低置信也要明确方向）
- 全部题目押同一个方向（把「跑输」当默认答案）——池内约半数股票跑赢半数跑输，两类都必须有召回
