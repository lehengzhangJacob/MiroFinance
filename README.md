# MiroFinance

面向全 A 股开放选股的金融智能体：在 [MiroFlow](https://github.com/MiroMindAI/miroflow) 上引入 **Memory**、**Skill** 与可审计的 **Skill 自进化**，用冻结点时数据和确定性回放器做月度组合回测。

- **评测窗**：2024-07 ~ 2026-06（24 个月）
- **主系统**：Memory + 晋升 Skill R1（`3aebb813bd33`）
- **环境**：Anaconda `Miro`，Python 3.12
- **报告**：`report/`（LaTeX）

> 结果来自 temperature=1.0 的单次 rollout 确定性重放，属于机制验证，不构成投资建议。

---

## 核心结果（24 个月）

| 系统 | 总收益 | 超额（vs 沪深300） | 最大回撤 | Sharpe |
|------|-------:|------------------:|---------:|-------:|
| **MiroFinance（Memory + R1）** | **+104.39%** | **+70.63pp** | **-8.31%** | **1.35** |
| w/o memory（仅 R1） | +112.88% | +79.12pp | -16.54% | 1.44 |
| MiroFlow-Plain | +67.44% | +33.69pp | -13.68% | 1.08 |
| w/o self-evolve | +34.48% | +0.73pp | -16.62% | 0.62 |
| w/o skill | +22.30% | -11.46pp | -26.92% | 0.49 |
| 沪深300 | +33.76% | 0 | -10.81% | 0.79 |

要点：

1. 完整系统相对 Plain 总收益高 36.95pp，最大回撤改善 5.37pp。
2. 同 Skill 下打开 Memory：峰值收益略降，回撤明显收窄（收益–风险权衡）。
3. 相对 w/o self-evolve / w/o skill，R1 技能贡献显著的描述性增量。

消融矩阵与分段指标见 [`ablation/README.md`](ablation/README.md)。

---

## 仓库结构

| 路径 | 说明 |
|------|------|
| `MiroFlow/` | 原版工具智能体（对照） |
| `MiroMemSkill/` | Memory + Skill 运行时 |
| `MiroMemSkill_hermes/` | **MiroFinance 主实现**（自进化 + 回测 fitness） |
| `ablation/` | 24 月 leave-one-out 消融脚本与密封 Skill |
| `shared/ashare_open_stocks_glm52_24m_20260715/` | 主快照（tasks + manifest；DB 需本地放置） |
| `deploy/conda/` | Anaconda `Miro` 环境定义与安装脚本 |
| `report/` | 论文式实验报告 |
| `任务要求.txt` | 研究任务说明 |

---

## 快速开始

```bash
# 1. 创建 / 更新 Miro 环境
source "$(conda info --base)/etc/profile.d/conda.sh"
bash deploy/conda/setup_miro.sh
conda activate Miro

# 2. 配置密钥（勿提交）
cp MiroMemSkill_hermes/.env.template MiroMemSkill_hermes/.env
# 填入 GLM / E2B / Jina / Serper / Tushare 等

# 3. Memory 需要 Qdrant（可选，仅 Memory-on 臂）
docker compose -f MiroMemSkill_hermes/deploy/qdrant/compose.yaml up -d
```

主实验入口（Skill 自进化，Memory 关闭）：

```bash
conda activate Miro
cd MiroMemSkill_hermes
python scripts/ashare/run_skill_evolution.py \
  --snapshot=../shared/ashare_open_stocks_glm52_24m_20260715 \
  --train_months=12 --dev_months=6 --holdout_months=6 \
  full --run_id=formal24m_20260715 --n=3 --cleanup_db=True
```

消融入口：

```bash
conda activate Miro
python ablation/run_skill_ablation.py --run_id=mem_ablation_24m
python ablation/run_skillonly_ablation.py --run_id=skillonly_r1_24m
python ablation/run_memonly_ablation.py --run_id=memonly_ablation_24m
python ablation/run_plain_ablation.py --run_id=plain_ablation_24m
python ablation/build_ablation_matrix.py
```

---

## 评测指标

全部由冻结快照上的确定性回放器（`eval_open_trader.py`）离线复算：

| 指标 | 定义 |
|------|------|
| 总收益 | 整窗逐月复利净收益（含费用、100 股整手） |
| 指数收益 | 同窗沪深300 |
| 超额 | 总收益 − 指数收益 |
| 最大回撤 | 月末净值峰谷最大跌幅 |
| Sharpe | `sqrt(12) × mean(月净收益) / stdev`，\(r_f=0\) |

---

## Skill 自进化（R1）

- 协议：train / dev / holdout = 12 / 6 / 6 月；进化期间 **Memory 关闭**。
- 晋升 Skill：`3aebb813bd33`（dev / holdout 均过硬门禁）。
- 生产文件：`MiroMemSkill_hermes/memory_bank/skills_ashare/ashare_open_portfolio.md`
- 设计文档：`MiroMemSkill_hermes/docs/hermes_evolution_design.md`

| 阶段 | baseline | R1 候选 | 月均配对差 |
|------|---------:|--------:|-----------:|
| Dev（2025-07~12） | +29.35% | **+83.53%** | +6.78pp（5–1） |
| Holdout（2026-01~06） | -1.89% | **+38.95%** | +5.94pp（5–1） |

---

## 许可与声明

基于 MiroFlow（Apache-2.0）改造。本仓库实验代码与结果仅供研究复现，不构成任何投资建议。
