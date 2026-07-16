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

## 评估与消融：全窗 24 个月（非自进化切分）

**口径说明**

- **自进化阶段**（上文 Hermes 正式实验）：必须用 train/dev/holdout（12/6/6）选 Skill、封存 holdout，防止同一测试窗反复晋升。密封数字仍以 [`ablation/r1_best_3aebb813bd33/`](ablation/r1_best_3aebb813bd33/) 为准。
- **评估 / 消融阶段（本节）**：Skill 已固定为晋升结果 `3aebb813bd33`，**不再做自进化**；整段 **2024-07 ~ 2026-06（24 个月）合在一起评估**，主实验与消融用同一快照、同一确定性回放器比总收益 / 超额 / 回撤。

| 项 | 值 |
|---|---|
| 评估窗 | **full_24m**：2024-07 ~ 2026-06（不分 train/dev/holdout） |
| 模型 | GLM-5.2（thinking enabled，temp=1.0） |
| 快照 | `shared/ashare_open_stocks_glm52_24m_20260715/` |
| 主实验 Skill | `3aebb813bd33`（R1 已晋升） |
| 进化前 Skill | `0a931278001c` |
| 细节与脚本 | [`ablation/`](ablation/) |

### 矩阵格（memory × skill）

| memory \ skill | none | Skill `0a931278001c` | Skill `3aebb813bd33`（主实验 / FINAL） |
|---|---|---|---|
| **off** | plain（own_glm6） | skill-only baseline（formal 拼接） | **主实验 R1** skill-only（own_glm5） |
| **on** | **w/o skill**（own_glm4） | **w/o self-evolve**（own_glm3） | FINAL = Memory+R1（own_glm3，待重开） |

### 全窗 24 个月总表（主实验 + 消融）

评估底座 = **主实验 R1**：Skill `3aebb813bd33`、Memory 关、整段连跑 24 月（`skillonly_r1_24m`，own_glm5）。  
产物：[`ablation/r1_best_3aebb813bd33/fitness_full_24m.{json,md}`](ablation/r1_best_3aebb813bd33/fitness_full_24m.md)。

| 格子 | 协议 | 总收益 | 超额 | 最大回撤 | 相对主实验 R1（总收益） |
|------|------|-------:|-----:|---------:|------------------------:|
| **主实验 R1** `3aebb813bd33` | Skill-only（Memory 关） | **+112.88%** | **+79.12%** | -16.54% | —（评估底座） |
| skill-only baseline `0a931…` | Skill-only（Memory 关） | +80.37% | +46.62% | -13.50% | **-32.50pp** |
| **w/o self-evolve** | Memory + Skill`0a931…` | +34.48% | +0.73% | -16.62% | **-78.39pp** |
| **w/o skill** | Memory，无 Skill | +22.30% | -11.46% | -26.92% | **-90.58pp** |
| plain | 无 Memory、无 Skill | 进行中 | | | |
| FINAL Memory+R1 | Memory + Skill`3aebb…` | 待重开 | | | |

沪深300 同期全窗均为 **+33.76%**。相对差为累计收益百分点差（跨 run、单种子，描述性；非晋升用的配对 sign test）。

**数据来源**

| 格子 | Run / 来源 |
|------|------------|
| 主实验 R1 24m | `ablation/runs/skillonly_r1_24m`（24/24；后验：技能注入 24/24，零 episode） |
| skill-only baseline 24m | `formal24m_20260715` 的 train+dev+holdout 轨迹拼接后同一回放器重放 |
| w/o self-evolve | `ablation/runs/mem_ablation_24m/arms/baseline`（24/24） |
| w/o skill | `ablation/runs/memonly_ablation_24m/arms/mem_only`（24/24） |

自进化时的 **dev/holdout 密封表**（选 Skill 用）仍见上文「24 月正式实验」与 `fitness_{dev,holdout}.*`；**消融主对照改用本节 full_24m**，避免晋升短窗与评估全窗混谈。

**解读**

1. **主实验全窗**：R1 Skill 在 skill-only 协议下 24 月 **+112.88%**（超额 +79.12%），相对同协议进化前 Skill（+80.37%）仍高 **+32.50pp**——全窗评估下自进化 Skill 仍强于未进化文本。
2. **消融**：相对主实验，拿掉 R1 Skill / 换成旧 Skill+Memory / 只留 Memory，全窗分别低约 **33 / 78 / 91pp**；其中 w/o skill 相对 w/o self-evolve 再低 **12.18pp**，说明 Memory 开时 Skill 文本仍有贡献。
3. **注意**：主实验与 Memory 臂协议不同；FINAL（Memory+R1）未跑完前，不要把「Memory 臂更低」直接读成 Memory 有害。temp=1.0、单种子，全窗数字含运行噪声（密封 holdout +38.95% 与本次全窗连跑的 holdout 段不必逐点相等）。

### 其余格子状态

| 格子 | key | 状态 |
|------|-----|------|
| plain | own_glm6 | 进行中 |
| FINAL Memory+R1 | own_glm3 | 待重开（干净技能目录） |

矩阵脚本：`ablation/build_ablation_matrix.py` → `ablation/reports/matrix_24m.md`。

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

*最后更新：2026-07-16（评估/消融改全窗 24m；主实验 R1 full_24m = +112.88%）*
