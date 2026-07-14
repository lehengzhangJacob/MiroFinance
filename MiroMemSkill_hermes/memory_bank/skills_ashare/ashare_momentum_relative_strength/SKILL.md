---
name: ashare_momentum_relative_strength
description: A股相对动量的点时诊断与消融基线；用于比较和识别极端反转风险，不是强制选股锚
version: "2.0"
applies_to: [factor_diagnostic, baseline_evaluation, reversal_research]
stock_universe: configured_point_in_time_universe
dependencies: [ashare_momentum_baseline_or_cli]
triggers:
  - 动量
  - momentum
  - rel20
  - 动量基线
  - 因子消融
  - 短期反转
  - top4
---

## 运行时用途

只有任务明确要求动量诊断，且环境提供对应工具时，才调用：

```
ashare_momentum_baseline(as_of, window=20, top_k=4)
```

该工具用于可复现地计算：

1. 将个股前复权日线与沪深300都截断到 `as_of`；
2. 计算 `rel20 = 个股近20日收益 - 沪深300近20日收益`；
3. 对全池按 `rel20` 降序、股票代码升序稳定排名；
4. 返回 top4 诊断组合和全池多窗口风险特征，供基线比较。

**禁止把 top4 直接变成强制持仓。** 全市场短周期实验中，近期涨幅最大的股票可能
发生显著反转。动量结果只能用于描述价格状态、识别过热和建立消融基线；组合仍需
独立验证成长质量、估值、流动性、波动和市场环境。

## 字段解读

- `rel20`：基线排序字段，单位为百分点；
- `rel5/rel60/rel120`：短期反转与中长期趋势确认；
- `vol20_ann/max_dd120/from_high250`：追高与尾部风险；
- `amount20_vs120`：近20日成交额相对近120日均值。

## 开发与复现

在本技能目录执行同口径 CLI：

```bash
python run.py --as-of 2024-07-01
python run.py --as-of 2024-07-01 --window 20 --top-k 4 --format csv
```

CLI 与 MCP 工具共用 `src/utils/ashare_momentum.py`。研究报告必须同时给出样本区间、
股票池、流动性过滤和交易成本，不能把某个旧小股票池结论外推到全市场。

## 防泄漏纪律

- 只能使用 `trade_date <= as_of` 的数据；不得使用持有期真实收益或事后排名；
- 个股收益用 `close_qfq`，指数收益用 `close`；
- 动量排名是决策日可见信息，不等于未来收益排名；
- `oracle_top4` 是不可交易的事后上界，不能作为本技能输入。
