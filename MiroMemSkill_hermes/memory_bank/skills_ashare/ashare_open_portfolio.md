---
name: ashare_open_portfolio
description: 全A股开放池的月度组合构建流程，覆盖多路筛选、成长质量验证、分散和动态现金
version: "1.0"
applies_to: [open_universe_portfolio_allocation]
stock_universe: all_ashare_point_in_time
dependencies: [ashare_universe_stats, ashare_screen_market, ashare_compare_growth_quality, ashare_market_breadth, ashare_index_history]
triggers:
  - 全A股
  - 开放池
  - 组合仓位
  - 统一交易员
  - CASH
  - 自由选股
---

## 决策流程

1. **确认市场与约束**：核对 `as_of`、持有期、交易成本、单股上限、可交易池和输出格式。
   用 `ashare_universe_stats`、`ashare_market_breadth`、`ashare_index_history` 判断市场环境。
2. **从不同来源建立候选集**：用 `ashare_screen_market` 分别寻找成长质量、合理估值、
   高流动性、低波动防御和非极端价格状态的股票。每一路只提供候选，不直接决定持仓。
3. **做批量交叉验证**：去重后将不超过 20 只候选交给
   `ashare_compare_growth_quality`；必要时再查个股财务、估值和价格历史。
4. **先排除再排序**：优先剔除流动性不足、财务数据缺失、营收与利润同时恶化、
   近期垂直拉升、估值无法解释的股票。剩余候选按证据覆盖、一致性和行业互补排序。
5. **构建组合**：遵守任务给定的持仓数和单股上限；避免多个高相关行业伪装成分散。
   不设置固定股票、固定核心或固定因子权重。
6. **确定 CASH**：现金由市场广度、候选质量和组合相关性共同决定。不能因为一般性
   不确定就机械持有高现金；也不能为满仓而纳入证据不足的股票。
7. **最终复核**：检查代码可交易、权重和为 1、费用可承受、没有使用 `as_of` 之后的信息。

## 记忆使用

若提示中有已到期的历史记忆，只用它校准筛选失误、仓位和行业集中风险；不要继承
旧股票名单，也不要把过去一期收益当作当前方向标签。

## 输出

遵循任务要求输出纯文本 `\boxed{代码:权重,...,CASH:权重}`。代码不得带 LaTeX
命令，权重使用 0 到 1 的小数。
