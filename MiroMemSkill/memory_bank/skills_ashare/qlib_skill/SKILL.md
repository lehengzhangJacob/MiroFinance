---
name: qlib_skill
description: 封装 Microsoft Qlib 的量化实验技能——本地缓存转 qlib 数据、LightGBM+Alpha158 训练预测、RankIC 信号评估、TopkDropout 组合回测与报告
triggers:
  - qlib
  - 回测
  - 因子
  - 组合评估
  - rankic
  - lightgbm
  - alpha158
  - 量化实验
  - 机器学习基线
---

## 适用场景

需要用**传统 ML 量化流程**（因子模型 + 组合回测）做实验或给 LLM Agent 预测提供对照基线时使用：
训练 LightGBM+Alpha158 因子模型 → 输出预测分数 → RankIC 信号评估 → TopkDropout 组合回测（相对沪深300）。
数据来自本仓库 Tushare 缓存（`data/ashare/`，可用 tushare_skill 更新），无需下载外部数据包。

## 前置条件

- 必须用 **Qlib conda 环境**运行：`/home/msj_team/.conda/envs/Qlib/bin/python`（缺环境时先跑 `deploy/conda/setup_qlib.sh`）；
- 首次使用先执行 `convert` 生成 qlib bin 数据。

## CLI 速查（在本技能目录下执行，python 指 Qlib 环境）

```bash
python run.py convert                                   # CSV 缓存 -> data/qlib_ashare bin 数据
python run.py train --run-name demo                     # 训练 LGBM+Alpha158，存 model/pred/label
python run.py signal --run-name demo                    # IC / RankIC / ICIR / 正率
python run.py backtest --run-name demo                  # TopkDropout 组合回测 vs SH000300
python run.py report --run-name demo                    # 汇总 -> outputs/demo/report.md
python run.py predict --run-name demo --start 2025-03-01 --end 2025-06-30   # 任意窗口打分
```

产物在 `outputs/<run-name>/`：`model.pkl`、`pred.pkl`、`label.pkl`、`signal.json`、`backtest.json`、`report.md`。

## 防泄漏纪律（必须遵守）

1. **段隔离**：test 段不得与 train/valid 重叠（config.yaml 的 segments 已按 2023~2024H1 训练 / 2024H2 验证 / 2025H1 测试切好）；
2. **标签前瞻只存在于训练目标**：label 是 20 交易日前瞻收益（与 A 股 agent 任务同 horizon），特征 Alpha158 只用截至 T 的数据；
3. **口径区别**：qlib 是批式回测（一次训练、整段评估），与 agent 任务的逐题 as_of 点时预测不同，报告对比时必须注明；
4. 小股票池（约 18 只）横截面 RankIC 噪声大，结论要看整段均值与 ICIR，不看单日。

## 输出指标

- signal：`ic_mean / ic_std / icir / rank_ic_mean / rank_icir / positive_rate`
- backtest：年化收益/超额、信息比率 IR、最大回撤、换手率（含费率 open 0.0005 / close 0.0015 / min 5）
