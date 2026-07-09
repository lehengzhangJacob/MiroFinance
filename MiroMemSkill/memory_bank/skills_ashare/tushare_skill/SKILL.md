---
name: tushare_skill
description: 规范化 Tushare Pro 数据技能——点时安全的A股行情/指数/估值/财务/日历 CLI，统一 token 解析与前复权口径
triggers:
  - tushare
  - 行情
  - 日线
  - 估值
  - 财务
  - 数据获取
  - 复权
  - 交易日历
---

## 适用场景

需要绕过预置缓存、直接从 Tushare Pro 拉取任意 A 股数据时使用（例如：新股票池、更长历史窗口、临时数据核验）。
已有本地缓存的回测任务优先用 ashare-market MCP 工具（同样点时安全且不耗积分）；本技能面向数据准备与自由查询。

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
3. 前复权基准 = 窗口末端（as_of 当日）复权因子：窗口内价格可比；不同 as_of 的两次查询结果不可直接拼接。

## 输出格式

stdout 输出 JSON envelope：`{"api", "params", "as_of", "count", "items": [...]}`；
`--out FILE` 时 items 写入 CSV，stdout 只打印摘要（含 `out` 路径）。
字段模型见 `schema.py`（DailyBar / IndexBar / Valuation / FinIndicator / StockInfo / TradeCalDay）。

## Token 解析顺序

`TUSHARE_TOKEN` 环境变量 → `config.yaml` 的 `token.file` → 从技能目录逐级向上查找 `tushare_token` 文件（兼容 `KEY=值` 与裸 token 两种写法）。
