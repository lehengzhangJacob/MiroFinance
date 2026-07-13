---
name: qlib_skill
description: Qlib多年walk-forward机器学习研发工具；当前全市场样本不足时不得作为正式组合信号
version: "2.0"
applies_to: [ml_research, walk_forward_training, factor_evaluation]
stock_universe: explicitly_configured_research_universe
dependencies: [Qlib_conda_environment, point_in_time_dataset]
triggers:
  - qlib
  - 机器学习
  - 信号
  - 回测
  - 因子
  - 组合评估
  - rankic
  - lightgbm
  - alpha158
  - 量化实验
---

## 当前定位

本技能是研发工具，不是当前全A股开放组合的运行时信号。只有同时满足以下条件，
模型分数才可进入正式决策：

1. 具有覆盖多个市场阶段的多年点时数据；
2. 每个决策月只训练标签已完全结算的样本；
3. 股票池、退市/ST、停牌和流动性过滤都按历史时点重建；
4. 样本外 RankIC、分组收益、换手和含费组合结果稳定；
5. 与简单价值、质量、反转和等权基线做过同口径比较。

旧16股实验生成的 `ashare_ml_signal` 和 demo 输出只用于复现，不代表全市场有效性。

## 开发与复现（需终端）

数据来自本仓库 Tushare 缓存（可用 `tushare_skill` 更新）。
必须用 **Qlib conda 环境**运行：`/home/msj_team/.conda/envs/Qlib/bin/python`（缺环境时先跑 `deploy/conda/setup_qlib.sh`）；首次使用先执行 `convert`。

```bash
python run.py convert                                   # CSV 缓存 -> data/qlib_ashare bin 数据（按 meta.json 股票池过滤）
python run.py walkforward                               # 逐月 walk-forward 打分 -> data/ashare/qlib_signal.csv（ashare_ml_signal 的数据源）
python run.py train --run-name demo                     # 训练 LGBM+Alpha158，存 model/pred/label
python run.py signal --run-name demo                    # IC / RankIC / ICIR / 正率
python run.py backtest --run-name demo                  # TopkDropout 组合回测 vs SH000300
python run.py report --run-name demo                    # 汇总 -> outputs/demo/report.md
python run.py predict --run-name demo --start 2025-03-01 --end 2025-06-30   # 任意窗口打分
```

产物在 `outputs/<run-name>/`：`model.pkl`、`pred.pkl`、`label.pkl`、`signal.json`、`backtest.json`、`report.md`。

## 防泄漏纪律

1. **walkforward 命令**：第 M 月的模型只训练「标签在 M 月决策日前已完全结算」的样本，逐月重训——这是评测口径的黄金标准；
2. **train 命令段隔离**：test 段不得与 train/valid 重叠（config.yaml 的 segments 已按 2023~2024H1 训练 / 2024H2 验证 / 2025H1 测试切好）；
3. **标签前瞻只存在于训练目标**：label 是 20 交易日前瞻收益（与 A 股 agent 任务同 horizon），特征 Alpha158 只用截至 T 的数据；
4. 小股票池横截面 RankIC 噪声大，不得外推到全市场；结论看多阶段样本外均值、
   ICIR、分组单调性和含费组合，不看单日或单月。

## 输出指标

- signal：`ic_mean / ic_std / icir / rank_ic_mean / rank_icir / positive_rate`
- backtest：年化收益/超额、信息比率 IR、最大回撤、换手率（含费率 open 0.0005 / close 0.0015 / min 5）
