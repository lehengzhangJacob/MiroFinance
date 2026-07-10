---
name: qlib_skill
description: qlib 机器学习信号的使用与解读——评测内经 ashare_ml_signal 查询逐月 walk-forward 截面分数；开发侧提供 LightGBM+Alpha158 训练/信号评估/组合回测 CLI
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

## 评测任务内怎么用（预测跑赢/跑输时）

评测环境**没有终端**，不要尝试运行任何 CLI；直接调用 MCP 工具：

```
ashare_ml_signal(ts_code, as_of)
```

返回该股最近若干个月度决策日的 qlib 分数与**池内截面排名**（rank 1 = 模型预测未来 20 日收益最高）。解读口径：

1. **截面可比**：同一月份内不同股票的 score/rank 可直接比较，rank 前 1/3 视为 [+看多超额]，后 1/3 视为 [-看空超额]，中间为 [0中性]；
2. **点时安全**：每个月的分数由「仅用标签在该决策日前已结算的数据」训练的模型产出，无未来信息，可放心引用；
3. **权重建议**：ML 信号是一条独立证据，与动量/估值/财务并列加权，信号与近端动量一致时可提高置信，冲突时以近端量价为主；
4. 分数绝对值无含义（横截面 z 分布），只看相对排名与近几个月排名变化趋势。

## 开发/复现用法（评测外，需终端）

数据来自本仓库 Tushare 缓存（`data/ashare/`，可用 tushare_skill 更新）。
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

## 防泄漏纪律（必须遵守）

1. **walkforward 命令**：第 M 月的模型只训练「标签在 M 月决策日前已完全结算」的样本，逐月重训——这是评测口径的黄金标准；
2. **train 命令段隔离**：test 段不得与 train/valid 重叠（config.yaml 的 segments 已按 2023~2024H1 训练 / 2024H2 验证 / 2025H1 测试切好）；
3. **标签前瞻只存在于训练目标**：label 是 20 交易日前瞻收益（与 A 股 agent 任务同 horizon），特征 Alpha158 只用截至 T 的数据；
4. 小股票池（16 只）横截面 RankIC 噪声大，结论要看整段均值与 ICIR，不看单日。

## 输出指标

- signal：`ic_mean / ic_std / icir / rank_ic_mean / rank_icir / positive_rate`
- backtest：年化收益/超额、信息比率 IR、最大回撤、换手率（含费率 open 0.0005 / close 0.0015 / min 5）
