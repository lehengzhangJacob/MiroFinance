---
name: tushare_skill
description: 规范化 Tushare Pro 数据技能——点时安全的A股行情/指数/估值/财务/日历 CLI，统一 token 解析与前复权口径
version: "2.0"
applies_to: [data_research, cache_maintenance, field_convention_reference]
stock_universe: all_ashare
dependencies: [Tushare_Pro, local_cache]
triggers:
  - tushare
  - 数据口径
  - pct_chg
  - 数据获取
  - 复权
  - 缓存更新
  - 交易日历
---

## 评测任务内怎么用（预测跑赢/跑输时）

评测环境**没有终端**，不要尝试运行任何 CLI。评测里所有 ashare-market 工具（price_history / index_history / valuation / financials / ml_signal / stock_info）的数据**都来自本技能维护的 Tushare 缓存**，字段口径疑问按下表解读：

| 工具字段 | 口径 |
|----------|------|
| `close` | 未复权收盘价，只用于展示或与同口径 `pre_close` 核对，不直接跨除权日相除 |
| `close_qfq` 等 | CLI 查询内以前复权价格展示，基准为该次查询 `as_of` 的复权因子；只能在同一次查询窗口内相除 |
| `pct_chg` | Tushare 日涨跌幅，基于交易所 `pre_close`；当前全市场 SQLite 筛选和回测统一按 `prod(1+pct_chg/100)-1` 复合 |
| `vol` / `amount` | 成交量（手，1手=100股）/ 成交额（千元） |
| `pe_ttm` / `pb` | 滚动市盈率/市净率，工具附带的分位是**近 N 日窗口内**分位，不是历史全样本 |
| `turnover_rate` | 换手率%（流通股本口径） |
| `ann_date` / `end_date` | 公告日 / 报告期；financials 按 `ann_date <= as_of` 截断，引用时必须核对 ann_date |

点时纪律：所有工具都按 `as_of` 硬截断。不要混用 `close_qfq` 比值和 `pct_chg`
复合来拼接同一段收益，也不要跨不同 `as_of` 查询拼接 qfq 价格。

## 开发/数据准备用法（评测外，需终端）

需要绕过预置缓存、直接从 Tushare Pro 拉取任意 A 股数据时使用（例如：新股票池、更长历史窗口、临时数据核验）。
已有本地缓存的回测任务优先用 ashare-market MCP 工具（同样点时安全且不耗积分）。

## CLI 速查（在本技能目录下执行）

```bash
python run.py daily --ts-code 600519.SH --start 20240101 --as-of 20240701      # 日线 + 前复权
python run.py index --start 20240101 --as-of 20240701                          # 沪深300 指数日线
python run.py valuation --ts-code 600519.SH --start 20240101 --as-of 20240701  # PE/PB/PS/换手/市值
python run.py financials --ts-code 600519.SH --as-of 20240701                  # 财务指标（按公告日截断）
python run.py stock-info --ts-code 600519.SH                                   # 股票档案（当前快照）
python run.py trade-cal --start 20240101 --end 20241231                        # 交易日历
```

通用参数：`--format json|csv`、`--out FILE`（写 CSV 文件）、`--fields` 覆盖默认字段。
日期同时接受 `YYYYMMDD` 与 `YYYY-MM-DD`。

## 点时纪律（防未来函数，必须遵守）

1. 回测/预测场景**必须**传 `--as-of=决策日`：行情/估值/日历按 `trade_date <= as_of` 截断，财务按**公告日** `ann_date <= as_of` 截断。
2. 引用财务数据时必须核对返回的 `ann_date`；`stock-info` 是当前快照、无点时保证，不得用于历史归因。
3. 前复权基准 = 窗口末端（as_of）复权因子：同一次查询窗口内价格可比；不同
   `as_of` 的两次查询结果不可直接拼接。
4. 当前全市场开放池的筛选、市场广度和回放以 SQLite 中 `pct_chg` 复合为唯一收益
   口径；CLI 的 qfq 价格主要用于人工核验，二者不得混算。

## 输出格式

stdout 输出 JSON envelope：`{"api", "params", "as_of", "count", "items": [...]}`；
`--out FILE` 时 items 写入 CSV，stdout 只打印摘要（含 `out` 路径）。
字段模型见 `schema.py`（DailyBar / IndexBar / Valuation / FinIndicator / StockInfo / TradeCalDay）。

## Token 解析顺序

`TUSHARE_TOKEN` 环境变量 → `config.yaml` 的 `token.file` → 从技能目录逐级向上查找 `tushare_token` 文件（兼容 `KEY=值` 与裸 token 两种写法）。
