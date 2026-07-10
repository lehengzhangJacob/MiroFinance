---
name: ashare_prediction_protocol
description: A股超额收益方向预测的标准作业流程与输出规范——工具调用清单、证据加权、格式硬约束
triggers:
  - 预测
  - 量化研究员
  - 超额收益
  - boxed
  - 交易日
---

## 标准流程（每题必做）

1. **工具调用清单**（as_of 一律用题目给定日期，格式 YYYYMMDD）：
   - ashare_price_history(ts_code, as_of, 60) → 个股动量
   - ashare_index_history(as_of, 60) → 指数动量
   - ashare_valuation(ts_code, as_of) → 估值分位
   - ashare_financials(ts_code, as_of) → 最近已披露财报
   - ashare_ml_signal(ts_code, as_of) → qlib 逐月 walk-forward 机器学习信号（**必查**：给出该股在池内的截面排名，rank 越靠前越可能跑赢）
2. **证据打分**：把每条证据标注为 [+看多超额] / [-看空超额] / [0中性]，明确主要矛盾；证据冲突时优先近端动量与量价，其次 ML 截面排名与估值分位。
3. **禁止事项**：
   - 不使用任何工具之外的市场记忆（你可能记得该股票后来的走势——**必须当作不知道**）；
   - 不预测指数或个股的绝对涨跌，只判断**相对沪深300 的超额方向**。
4. **输出硬约束**：
   - 正文给 3-5 条核心依据（每条一行，标注 +/-/0）；
   - 最后一行**只写** `\boxed{跑赢}` 或 `\boxed{跑输}`，不得写"可能/大概率"等模糊词，不得同时给两个答案；
   - 不输出置信区间、免责声明等冗余内容。

## 配套技能（按需 skill_load 加载）

- **qlib_skill**：解释 ashare_ml_signal 分数的产生方式与解读口径（LightGBM+Alpha158、逐月 walk-forward、防泄漏）；对分数含义有疑问时加载；
- **tushare_skill**：本任务所有 ashare 工具的数据都来自 Tushare 缓存，字段口径（前复权、公告日截断、换手率单位等）有疑问时加载。

## 常见陷阱

- 最终行写成 `\boxed{跑赢沪深300}`、`\boxed{outperform}` 等变体（判分按「跑赢/跑输」精确语义）
- 花大量轮次反复拉同一数据（每个工具一次调用足够，max_turns 有限）
- 因证据不足而拒绝给结论（本任务必须二选一，宁可低置信也要明确方向）
- 全部题目押同一个方向（把「跑输」当默认答案）——池内约半数股票跑赢半数跑输，两类都必须有召回
