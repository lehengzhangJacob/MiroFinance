# Jacob / Miro 实验记录

本目录是 MiroFlow、MiroMemSkill、MiroMemSkill_hermes 等 A 股开放池实验的工作区。  
冻结快照与完整报告见 [`shared/ashare_open_stocks_glm52_20260714/`](shared/ashare_open_stocks_glm52_20260714/)。

---

## 全 A 股开放池：MiroFlow vs MiroMemSkill（正式结果）

**评测设定（memfix02，有效）**

| 项 | 值 |
|---|---|
| 模型 | GLM-5.2（thinking enabled，temp=1.0） |
| 任务 | 全 A 股开放池月度组合，12 个月（2024-07 ~ 2025-06） |
| 起点 | 100 万，逐月复利，含费、整手 |
| 对照 | MiroFlow-Plain（无 Memory） vs MiroMemSkill-MemSkill（有 Memory + Skill） |
| 快照 | `shared/ashare_open_stocks_glm52_20260714/` |
| 完整报告 | [`shared/.../reports/ashare_open_flow_vs_memskill_20260714_memfix02_full.md`](shared/ashare_open_stocks_glm52_20260714/reports/ashare_open_flow_vs_memskill_20260714_memfix02_full.md) |
| Git 分支 | `MemSkill` @ [MiroFinance](https://github.com/lehengzhangJacob/MiroFinance.git) |

### 12 个月汇总

| 策略 | 总收益 | 相对 ETF 核心 | 最大回撤 | 最差月 | 月胜率 | 费用 |
|------|--------|---------------|----------|--------|--------|------|
| **MiroFlow-Plain** | **+21.28%** | +5.87pp | -14.81% | -12.16% | 50.0% | 16,928 |
| **MiroMemSkill-MemSkill** | **+38.18%** | +22.77pp | -17.01% | -12.08% | 66.7% | 19,934 |
| 沪深300 | +13.98% | — | -10.81% | -8.99% | 58.3% | 0 |
| 90% ETF 等权核心 | +15.41% | 0 | -11.24% | -10.62% | 58.3% | 23,597 |
| 低 PE top4 | +25.10% | +9.69pp | -18.95% | -10.92% | 50.0% | 25,738 |
| 全市场全等权（不可交易参考） | +36.91% | +21.50pp | -13.67% | -13.67% | 66.7% | 0 |
| rel20 动量 top4 | -64.98% | -80.39pp | -72.76% | -36.38% | 41.7% | 10,854 |

**MemSkill 比 Flow 多 +16.90pp**（约 16.9 万元/100 万本金）。

### 逐月净收益（%）

| 月份 | Flow | MemSkill | 指数 |
|------|------|----------|------|
| 2024-07 | -3.02 | +3.48 | -2.51 |
| 2024-08 | -12.16 | -7.24 | -4.14 |
| 2024-09 | +15.00 | +20.41 | +21.16 |
| 2024-10 | -0.98 | +4.59 | +1.73 |
| 2024-11 | -2.15 | -3.88 | -2.33 |
| 2024-12 | -3.51 | +0.74 | -2.80 |
| 2025-01 | +7.77 | -2.54 | +1.89 |
| 2025-02 | +2.59 | +0.01 | +1.32 |
| 2025-03 | -8.17 | -12.08 | -8.99 |
| 2025-04 | +4.95 | +3.93 | +7.34 |
| 2025-05 | +6.57 | +12.95 | +0.55 |
| 2025-06 | +16.61 | +17.34 | +2.79 |

### 解读与注意

- **memfix01 无效**：episode 写入失败（`universe: all_ashare` 与 gate 期望 `all_ashare_point_in_time` 不匹配），Memory 未生效；MemSkill 仅 **-5.99%**。勿用 memfix01 作 Memory 对比结论。
- **memfix02 为正式对比**：修复 episode 后重跑 MemSkill 臂；Flow 臂复用同一次 rollout。
- **归因偏弱**：单次运行、temp=1.0、无固定 seed；MemSkill 7/12 月胜出，统计上不显著；+38.18% 接近全等权参考 +36.91%，不能断言 Memory 一定是主因。
- **大 DB 发布**：Git LFS 配额不足，大快照通过 GitHub Release `ashare-open-20260714` 分发；见 `shared/.../download_assets.sh`。

---

## MiroMemSkill_hermes：Skill 进化（12 月冒烟，非 Flow/Mem 对比）

与上面 Flow vs MemSkill **不是同一条实验线**——Hermes fork 只改 Skill 正文，Memory 全关，用 train/dev/holdout 切分。

| 阶段 | baseline（原版 Skill） | 候选 `cc249cf4da2d` |
|------|------------------------|---------------------|
| train 6 月 | +7.88% | —（用于生成变异） |
| dev 3 月 | -17.39% | -18.44%（配对 -0.59pp） |
| holdout 3 月 | +13.44% | **+26.05%**（配对 +3.64pp） |

- Run ID：`full_20260714_230306`（`MiroMemSkill_hermes/.evolution/runs/`）
- 生产 Skill **未 promote**；holdout 仅 3 月，`sign_p=1.0`，不能当正式结论。
- 设计文档：`MiroMemSkill_hermes/docs/hermes_evolution_design.md`

---

## 目录速查

| 路径 | 说明 |
|------|------|
| `MiroFlow/` | 原版 agent 运行时 |
| `MiroMemSkill/` | Memory + Skill 运行时 |
| `MiroMemSkill_hermes/` | Skill 自进化 fork（Hermes 思路 + 真实回测 fitness） |
| `shared/ashare_open_stocks_glm52_20260714/` | 12 月冻结快照 + 对比报告 |
| `shared/ashare_open_stocks_glm52_24m_20260715/` | 24 月扩展快照（进行中） |
| `任务要求.txt` | 研究任务与 ablation 要求 |

---

*最后更新：2026-07-15*
