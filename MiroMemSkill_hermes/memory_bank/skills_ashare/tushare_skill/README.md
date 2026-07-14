# tushare_skill

规范化封装 [Tushare Pro](https://tushare.pro)（[waditu/tushare](https://github.com/waditu/tushare) 的 Pro 版 HTTP API）为**点时安全**的命令行数据技能。

与直接 `pip install tushare` 相比：

- 直连 `http://api.tushare.pro`，仅依赖 `requests/pandas/pyyaml`（免费旧接口已停止维护，不引入 tushare 包）；
- 内置 `--as-of` 点时截断：行情按交易日、财务按**公告日**过滤，杜绝回测未来函数；
- CLI 提供前复权（qfq）价格用于单次查询窗口核验；当前全市场 SQLite 筛选与回测
  统一使用 `pct_chg` 复合收益，两种口径不得混算；
- 统一 token 解析、限频重试与输出 envelope，agent 与人都能直接用。

## 目录

```
tushare_skill/
├── SKILL.md           # agent 入口（frontmatter + 用法 + 点时纪律）
├── README.md          # 本文件
├── requirements.txt   # requests / pandas / pyyaml
├── config.yaml        # API 地址、token、重试限频、各接口默认字段
├── schema.py          # 输入校验 + 输出 dataclass 模型 + envelope
├── run.py             # CLI 总入口（6 个子命令）
├── examples/
│   └── examples.sh    # 每个子命令一条可跑示例
└── tests/
    └── test_tushare_skill.py  # 离线 mock 测试 + 可选 live smoke
```

## 安装与 Token

```bash
pip install -r requirements.txt   # conda env Miro 已满足，无需重复安装

# 三选一（按解析优先级）：
export TUSHARE_TOKEN=你的token                 # 1. 环境变量
# 2. config.yaml 里设置 token.file: /abs/path/to/token
# 3. 仓库根目录放 tushare_token 文件（本仓库已有，KEY=值 或裸 token 均可）
```

## 子命令与 Tushare 接口对照

| 子命令 | api_name | 主要字段 | as_of 语义 |
|--------|----------|----------|-----------|
| `daily` | `daily` + `adj_factor` | OHLC、pct_chg、vol、amount、`*_qfq` | `trade_date <= as_of` |
| `index` | `index_daily` | OHLC、vol、amount | `trade_date <= as_of` |
| `valuation` | `daily_basic` | pe_ttm、pb、ps_ttm、turnover_rate、市值 | `trade_date <= as_of` |
| `financials` | `fina_indicator` | eps、roe、毛利率、净利率、营收/净利同比、`ann_date` | **`ann_date <= as_of`** |
| `stock-info` | `stock_basic` | 名称、行业、市场、上市日期 | 当前快照（无点时保证） |
| `trade-cal` | `trade_cal` | cal_date、is_open | `cal_date <= as_of` |

## 使用示例

```bash
cd MiroMemSkill/memory_bank/skills_ashare/tushare_skill

# 茅台 2024H1 前复权日线，决策日 2024-07-01
python run.py daily --ts-code 600519.SH --start 20240101 --as-of 20240701

# 结果落盘 CSV
python run.py valuation --ts-code 600519.SH --start 20240101 --as-of 20240701 \
    --out /tmp/moutai_valuation.csv

# 点时财务：只返回 2024-07-01 前已公告的报告
python run.py financials --ts-code 600519.SH --as-of 20240701
```

stdout 统一返回 JSON envelope：

```json
{"api": "daily", "params": {"ts_code": "600519.SH", ...}, "as_of": "20240701", "count": 116, "items": [...]}
```

## 点时（point-in-time）设计

本技能把 A 股回测中踩过的坑固化为默认行为：

1. **行情/估值/日历**：请求端把 `end_date` 压到 `as_of`，返回后再按 `trade_date <= as_of` 二次过滤；
2. **财务**：`fina_indicator` 的 `end_date` 参数是报告期而非公告日，因此**客户端按 `ann_date <= as_of` 过滤**，并丢弃缺失 `ann_date` 的行；
3. **前复权**：以窗口末端（as_of）复权因子为基准，同一次查询窗口内价格可比；
   不同 as_of 查询结果不可跨窗口拼接；
4. **全市场收益**：开放池 SQLite 管线使用
   `prod(1 + pct_chg / 100) - 1` 计算窗口收益。`close` 是未复权展示价，
   `close_qfq` 是 CLI 查询内的复权价，不得与 `pct_chg` 拼接或混算。

## 测试

```bash
python -m unittest discover -s tests -v          # 离线（mock HTTP，不耗积分）
TUSHARE_LIVE=1 python -m unittest discover -s tests -v   # 追加真实 API smoke
```
