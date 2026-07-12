---
name: ashare_momentum_relative_strength
description: A股相对沪深300动量软锚——评测内用 ashare_momentum_baseline 计算严格点时的全池排名与 top4 参考组合；开发侧提供同口径 CLI
triggers:
  - 跑赢
  - 跑输
  - 沪深300
  - 相对
  - 动量
  - momentum
  - 软锚
  - top4
---

## 评测任务内怎么用

评测环境没有终端，不要运行 CLI。统一组合决策应先调用：

```
ashare_momentum_baseline(as_of, window=20, top_k=4)
```

工具用代码完成以下计算：

1. 将个股前复权日线与沪深300都截断到 `as_of`；
2. 计算 `rel20 = 个股近20日收益 - 沪深300近20日收益`；
3. 对全池按 `rel20` 降序、股票代码升序稳定排名；
4. 返回 top4 各 25%、现金 0% 的软锚，以及全池多窗口和风险诊断。

软锚不是必须照抄的答案。证据弱时靠近它；只有 5 日反转或过热、60/120 日趋势
不确认、高波动/回撤、量价异常，或估值、财务、Qlib 明显冲突时才替换成分、
降低权重或提高现金，并明确说明偏离理由。

## 字段解读

- `rel20`：基线排序字段，单位为百分点；
- `rel5/rel60/rel120`：短期反转与中长期趋势确认；
- `vol20_ann/max_dd120/from_high250`：追高与尾部风险；
- `amount20_vs120`：近20日成交额相对近120日均值。

## 开发/复现用法

在本技能目录执行同口径 CLI：

```bash
python run.py --as-of 2024-07-01
python run.py --as-of 2024-07-01 --window 20 --top-k 4 --format csv
```

CLI 与 MCP 工具共用 `src/utils/ashare_momentum.py`，避免评测与离线复现出现两套公式。

## 防泄漏纪律

- 只能使用 `trade_date <= as_of` 的数据；不得使用持有期真实收益或事后排名；
- 个股收益用 `close_qfq`，指数收益用 `close`；
- 动量排名是决策日可见信息，不等于未来收益排名；
- `oracle_top4` 是不可交易的事后上界，不能作为本技能输入。
