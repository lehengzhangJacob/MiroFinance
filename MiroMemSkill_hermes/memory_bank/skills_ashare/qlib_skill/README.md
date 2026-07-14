# qlib_skill

封装 [Microsoft Qlib](https://github.com/microsoft/qlib)（AI-oriented quantitative investment platform）为规范化命令行技能：

**本地缓存 → qlib bin 数据 → LightGBM+Alpha158 训练 → RankIC 信号评估 → TopkDropout 组合回测 → markdown 报告**。

定位：给本仓库的 LLM Agent A 股预测实验提供一个**传统 ML 量化基线**（同一批股票、同一 20 交易日 horizon），
也可独立做因子模型实验。

## 目录

```
qlib_skill/
├── SKILL.md           # agent 入口（frontmatter + 用法 + 防泄漏纪律）
├── README.md          # 本文件
├── requirements.txt   # pyqlib / lightgbm / pyyaml
├── config.yaml        # 数据路径、股票池、段切分、label horizon、回测参数
├── schema.py          # 输入校验 + SignalMetrics/BacktestMetrics + envelope
├── qlib_dump.py       # 最小 CSV→qlib bin 转换器（不依赖官方 dump_bin 脚本）
├── run.py             # CLI: convert | train | predict | signal | backtest | report
├── examples/examples.sh
└── tests/test_qlib_skill.py
```

## 环境（独立 conda env）

pyqlib 对 numpy/pandas 钉版本，与仓库 Miro 环境（numpy 2.3）冲突，因此用独立环境：

```bash
./deploy/conda/setup_qlib.sh          # 创建 conda env "Qlib"（python 3.11 + pyqlib + lightgbm）
conda activate Qlib                   # 或直接用 /home/msj_team/.conda/envs/Qlib/bin/python
```

## 数据两条路径

**路径 A（默认，零下载）**：复用本仓库 Tushare 缓存 `MiroMemSkill/data/ashare/`（约 18 只 A 股 + 沪深300，
2023-01 ~ 2025-07，前复权），由 `convert` 子命令转成 qlib bin 格式，输出到 `MiroMemSkill/data/qlib_ashare/`：

```
calendars/day.txt                 # 交易日历（源自 trade_cal.csv）
instruments/all.txt               # 每行: CODE<TAB>start<TAB>end
features/sh600519/close.day.bin   # float32: [起始日历下标, v0, v1, ...]
```

代码映射 `600519.SH → SH600519`；价格用 qfq 前复权列、`factor=1`；沪深300 转成 `SH000300` 作回测 benchmark。

**路径 B（可选，全市场）**：下载社区数据包（数 GB，本仓库默认不用）：

```bash
wget https://github.com/chenditc/investment_data/releases/latest/download/qlib_bin.tar.gz
mkdir -p ~/.qlib/qlib_data/cn_data
tar -zxvf qlib_bin.tar.gz -C ~/.qlib/qlib_data/cn_data --strip-components=1
# 然后把 config.yaml 的 provider_uri 指到该目录、instruments 改 csi300
```

## 快速开始

```bash
cd MiroMemSkill/memory_bank/skills_ashare/qlib_skill
PY=/home/msj_team/.conda/envs/Qlib/bin/python

$PY run.py convert                      # 一次性：CSV -> qlib bin
$PY run.py train    --run-name demo     # 训练 + 测试段预测（秒级）
$PY run.py signal   --run-name demo     # IC / RankIC / ICIR
$PY run.py backtest --run-name demo     # 组合回测 vs SH000300
$PY run.py report   --run-name demo     # 汇总 outputs/demo/report.md
```

## 实验设定（config.yaml 可改）

| 项 | 默认 | 说明 |
|----|------|------|
| label | `Ref($close,-21)/Ref($close,-1)-1` | 20 交易日前瞻收益，与 agent 任务同 horizon |
| 特征 | Alpha158 | qlib 标准 158 因子 |
| 模型 | LGBModel | LightGBM 回归 |
| segments | train 2023-01~2024-06 / valid 2024-07~2024-12 / test 2025-01~2025-06 | test 对齐 pool3 后半段 |
| 策略 | TopkDropout topk=3, n_drop=1 | 约 18 只池子里持有 3 只 |
| 费率 | open 0.0005 / close 0.0015 / min 5 | qlib cn 常用设置 |

## 指标口径

- **signal**（`qlib.contrib.eva.alpha.calc_ic`）：逐日 Pearson IC 与 Spearman RankIC 的
  mean / std / ICIR(=mean/std) / 正率。小池子（约 18 只/日）横截面噪声大，看整段均值。
- **backtest**（`qlib.contrib.evaluate.risk_analysis`）：策略与超额（相对 SH000300）的
  年化收益、信息比率、最大回撤；另报换手。

## 与 agent 任务的口径差异（写报告必读）

qlib 是**批式**流程：一次训练、整段测试；agent benchmark 是**逐题点时**（每题 as_of 截断、独立预测）。
两者可比的是同 horizon 的方向判别能力与组合表现，但训练信息量不同，对比时必须注明。

## 测试

```bash
# 任意环境（qlib 相关用例自动 skip）
python -m unittest discover -s tests -v
# Qlib 环境全量（含转换器回读 + e2e 小训练）
/home/msj_team/.conda/envs/Qlib/bin/python -m unittest discover -s tests -v
```
