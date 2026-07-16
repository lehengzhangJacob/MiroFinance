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

## MiroMemSkill_hermes：Skill 进化（非 Flow/Mem 对比）

与上面 Flow vs MemSkill **不是同一条实验线**——Hermes fork 只改 Skill 正文，Memory 全关，fitness 来自确定性回测回放。设计文档：`MiroMemSkill_hermes/docs/hermes_evolution_design.md`

### 24 月正式实验（formal24m_20260715）✅

**设定**

| 项 | 值 |
|---|---|
| 模型 | GLM-5.2（`own_glm`，thinking enabled，temp=1.0） |
| 快照 | `shared/ashare_open_stocks_glm52_24m_20260715/`（755 交易日，DB 本地/LFS） |
| 切分 | train 12 月（2024-07~2025-06）/ dev 6 月 / holdout 6 月 |
| 候选数 | 3（reflective mutation from train feedback） |
| Run ID | `formal24m_20260715` |
| 报告 | `MiroMemSkill_hermes/.evolution/runs/formal24m_20260715/reports/` |

**三候选 dev 筛选**

| 候选 | dev 门控 | dev 配对均值 | 备注 |
|------|----------|--------------|------|
| `b25723dfcf23` | ❌ | -1.96pp | 最大回撤劣化 11.68pp |
| `cd1ab8517312` | ✅ | +2.46pp | 未进 holdout（dev 分较低） |
| **`3aebb813bd33`** | ✅ | **+6.78pp** | dev 最高 → 唯一 holdout |

**Dev（2025-07 ~ 2025-12，6 月）— baseline vs `3aebb813bd33`**

| | 总收益 | 超额 | 最大回撤 | 月胜率 |
|--|--------|------|----------|--------|
| baseline | +29.35% | +13.83% | -6.52% | 67% |
| candidate | **+83.53%** | **+68.00%** | **-3.12%** | 67% |
| 配对 | 5 胜 1 负，均值 **+6.78pp**，sign_p=0.22 | | | |

**Holdout（2026-01 ~ 2026-06，6 月，一次性封存）**

| | 总收益 | 超额 | 最大回撤 | 月胜率 |
|--|--------|------|----------|--------|
| baseline | -1.89% | -3.47% | -7.41% | 50% |
| candidate | **+38.95%** | **+37.37%** | **-3.84%** | 83% |
| 配对 | 5 胜 1 负，均值 **+5.94pp**，sign_p=0.22 | | | |

Dev 与 holdout 硬门控（无效月、回撤劣化 ≤5pp）均 **PASS**。

**晋升决定：已 promote `3aebb813bd33` → 生产 Skill**

- 时间：2026-07-15，`run_id=formal24m_20260715`
- 生产文件：`MiroMemSkill_hermes/memory_bank/skills_ashare/ashare_open_portfolio.md`
- 备份：`.evolution/backups/20260715_111536_0a931278001c.md`（可 `rollback 0a931278001c`）
- **理由**：24 月正式切分下 dev/holdout 均过硬门控；holdout 配对 +5.94pp 且回撤优于 baseline（-3.84% vs -7.41%）；dev/holdout 方向一致（5/6 月胜出）。

**候选 Skill 主要新增规则**（相对 baseline `0a931278001c`）

- 动量暴露控制：近 20 日动量分位上限、高动量占比 >40% 时补低波/估值候选
- 行业/集中度硬约束：申万一级 ≤30%、二级 ≤20%、前三重仓 ≤50%、最少 5 只持仓
- 广度过热防御：`ashare_market_breadth` 高位时分位触发 CASH ≥30%
- 剔除垂直拉升、高换手率（>90 分位）个股；整手约束复核

### 12 月冒烟（已 supersede，仅供参考）

| 阶段 | baseline | 候选 `cc249cf4da2d` |
|------|----------|---------------------|
| train 6 月 | +7.88% | — |
| dev 3 月 | -17.39% | -18.44%（配对 -0.59pp） |
| holdout 3 月 | +13.44% | +26.05%（配对 +3.64pp） |

Run ID：`full_20260714_230306`。未 promote（holdout 仅 3 月，sign_p=1.0）。

---

## 消融对照：以晋升 Skill `3aebb813bd33` 为 FINAL

Leave-one-out 矩阵：把 **FINAL = Memory + R1 Skill** 当作完整系统，分别拿掉 memory / skill / 两者，看各自贡献。

| 对照命名 | 含义 |
|----------|------|
| **w/o self-evolve** | Memory 开 + 进化前 Skill `0a931278001c`（无 Hermes 自进化） |
| **w/o skill** | Memory 开 + **无 Skill**（trader-episode 记忆，exit 日期到期后可见） |

| 项 | 值 |
|---|---|
| 模型 | GLM-5.2（thinking enabled，temp=1.0） |
| 快照 | `shared/ashare_open_stocks_glm52_24m_20260715/`（24 个月，2024-07 ~ 2026-06） |
| FINAL Skill | `3aebb813bd33`（formal24m R1 已晋升） |
| 进化前 Skill | `0a931278001c` |
| 细节与脚本 | [`ablation/`](ablation/) |

### 矩阵格（memory × skill）

| memory \ skill | none | Skill `0a931278001c`（进化前） | Skill `3aebb813bd33`（R1 / FINAL） |
|---|---|---|---|
| **off** | plain（own_glm6） | — | skill-only R1（own_glm5） |
| **on** | **w/o skill**（own_glm4） | **w/o self-evolve**（own_glm3） | **FINAL** full R1（own_glm3） |

### 已完成对照（24 个月全窗）

#### w/o self-evolve（Memory + 进化前 Skill）

Run：`ablation/runs/mem_ablation_24m/arms/baseline`（24/24，Skill=`0a931278001c`）。

| 分段 | 区间 | 总收益 | 沪深300 | 超额 | 最大回撤 |
|------|------|-------:|--------:|-----:|---------:|
| **full_24m** | 2024-07 ~ 2026-06 | **+34.48%** | +33.76% | **+0.73%** | -16.62% |
| formal_12m | 2024-07 ~ 2025-06 | +18.00% | +13.98% | +4.03% | -16.62% |
| dev_6m | 2025-07 ~ 2025-12 | +22.04% | +15.53% | +6.51% | -11.18% |
| holdout_6m | 2026-01 ~ 2026-06 | -6.18% | +1.58% | -7.76% | -10.99% |

全窗：月胜率 54.2%，费用 ¥37,611，无效月 0。

#### w/o skill（仅 Memory，无 Skill）

Run：`ablation/runs/memonly_ablation_24m/arms/mem_only`（24/24，config=`agent_ashare_trader_open_hermes_memonly_glm`，key=`own_glm4`）。  
后验：0 次技能注入、24 条 episode 写入、23 个月有到期记忆块注入。

| 分段 | 区间 | 总收益 | 沪深300 | 超额 | 最大回撤 |
|------|------|-------:|--------:|-----:|---------:|
| **full_24m** | 2024-07 ~ 2026-06 | **+22.30%** | +33.76% | **-11.46%** | -26.92% |
| formal_12m | 2024-07 ~ 2025-06 | -11.88% | +13.98% | -25.86% | -26.92% |
| dev_6m | 2025-07 ~ 2025-12 | +19.16% | +15.53% | +3.63% | -10.16% |
| holdout_6m | 2026-01 ~ 2026-06 | +16.66% | +1.58% | +15.08% | -2.20% |

全窗：月胜率 54.2%，费用 ¥32,432，无效月 0。

#### 对照表（含主实验 R1）

主实验数字来自 [`ablation/r1_best_3aebb813bd33/`](ablation/r1_best_3aebb813bd33/)（`formal24m_20260715` 密封 fitness）：**Skill=`3aebb813bd33`，Memory 关（skill-only 协议）**。  
消融两格为 **Memory 开**、跨 run、单种子；同一快照与同一确定性回放器，但**协议不同，不能当配对检验**。

**A. 全窗 24 个月（仅消融格；主实验未跑 full_24m）**

| 格子 | 协议 | 总收益 | 超额 | 最大回撤 | 相对 w/o self-evolve |
|------|------|-------:|-----:|---------:|---------------------:|
| **w/o self-evolve** | Memory + Skill`0a931…` | +34.48% | +0.73% | -16.62% | —（消融底座） |
| **w/o skill** | Memory，无 Skill | +22.30% | -11.46% | -26.92% | **-12.18pp** |
| 主实验 R1 | Skill`3aebb…`，Memory 关 | — | — | — | 无全窗密封数字 |

**B. 与主实验对齐的分段（dev / holdout = formal24m 密封窗）**

| 格子 | 协议 | Dev 总收益 (2025-07..12) | Holdout 总收益 (2026-01..06) | Holdout 超额 | Holdout maxDD |
|------|------|-------------------------:|-----------------------------:|-------------:|--------------:|
| **主实验 R1** `3aebb813bd33` | Skill-only（Memory 关） | **+83.53%** | **+38.95%** | **+37.37%** | **-3.84%** |
| 主实验 baseline Skill | Skill-only（Memory 关） | +29.35% | -1.89% | -3.47% | -7.41% |
| **w/o self-evolve** | Memory + Skill`0a931…` | +22.04% | -6.18% | -7.76% | -10.99% |
| **w/o skill** | Memory，无 Skill | +19.16% | +16.66% | +15.08% | -2.20% |

主实验 R1 相对其 skill-only baseline：dev 配对均值 **+6.78pp**（5-1），holdout 配对均值 **+5.94pp**（5-1），硬门控 PASS（详见上文 Hermes 正式实验节与 `fitness_{dev,holdout}.*`）。

**解读**

1. **消融内部**：去掉 Skill（w/o skill vs w/o self-evolve）全窗总收益 **-12.18pp**，说明在 Memory 开的设定下，进化前 Skill 文本仍有实质贡献。
2. **对主实验**：密封窗上主实验 R1（Memory 关 + 进化后 Skill）远强于两消融格（Memory 开 ± 无/旧 Skill）。这不能直接读成「Memory 有害」——协议不同（有无 Skill、是否为 R1 文本、是否 skill-only 选出）；真正的 Memory×R1 交互要等 **FINAL = Memory + `3aebb813bd33`** 与 **skill-only R1 复现臂** 跑完再比。
3. 分段上 w/o skill 在 holdout 看起来不差（+16.66%），但 formal_12m 深度拖累（-11.88%），故以 full_24m 与密封 holdout 对照主实验时分开读。

### 其余格子（进行中，跑完后回填）

| 格子 | 含义 | key | 状态 |
|------|------|-----|------|
| FINAL (full R1) | Memory + `3aebb813bd33` | own_glm3 | 待重开（r1_best 臂启动失败） |
| skill-only R1 | 仅 `3aebb813bd33`，无 Memory | own_glm5 | 进行中 |
| plain | 无 Memory、无 Skill | own_glm6 | 进行中 |

矩阵汇总脚本：`ablation/build_ablation_matrix.py` → `ablation/reports/matrix_24m.md`。

---

## 目录速查

| 路径 | 说明 |
|------|------|
| `MiroFlow/` | 原版 agent 运行时 |
| `MiroMemSkill/` | Memory + Skill 运行时 |
| `MiroMemSkill_hermes/` | Skill 自进化 fork（Hermes 思路 + 真实回测 fitness） |
| `ablation/` | Leave-one-out 消融（FINAL=`3aebb813bd33`；含 w/o self-evolve、w/o skill） |
| `shared/ashare_open_stocks_glm52_20260714/` | 12 月冻结快照 + 对比报告 |
| `shared/ashare_open_stocks_glm52_24m_20260715/` | 24 月扩展快照（train/dev/holdout 任务 + manifest；DB 本地） |
| `任务要求.txt` | 研究任务与 ablation 要求 |

---

*最后更新：2026-07-16（消融表对照主实验 R1 `3aebb813bd33` 密封数字）*
