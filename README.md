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

### 主系统与四个消融

完整主系统定义为 `Memory + R1 Skill 3aebb813bd33`。四个消融分别移除自进化、
Skill、Memory，以及同时移除 Memory 与 Skill：

| 格子 | Memory | Skill | 被移除的组件 |
|------|:------:|-------|---------------|
| **完整主系统（Full R1）** | 开 | R1 `3aebb813bd33` | — |
| **w/o self-evolve** | 开 | 进化前 `0a931278001c` | 自进化后的 Skill 文本 |
| **w/o skill** | 开 | 无 | Skill |
| **w/o memory** | 关 | R1 `3aebb813bd33` | Memory |
| **plain** | 关 | 无 | Memory、Skill 和 self-evolve |

### 全窗 24 个月结果

四个消融均已完成。完整主系统的 `Memory + R1` arm 尚未完成，因此当前不能计算
四个消融相对 Full R1 的严格降幅。下表最后一列暂以已完成的 `w/o memory`
（R1 skill-only）作为描述性评估底座；它不是完整主系统。

| 格子 | 协议 | 总收益 | 超额 | 最大回撤 | 相对 R1 skill-only |
|------|------|-------:|-----:|---------:|-------------------:|
| **完整主系统（Full R1）** | Memory + R1 `3aebb813bd33` | 待运行 | 待运行 | 待运行 | — |
| **w/o self-evolve** | Memory + `0a931278001c` | +34.48% | +0.73% | -16.62% | -78.39pp |
| **w/o skill** | Memory，无 Skill | +22.30% | -11.46% | -26.92% | -90.58pp |
| **w/o memory** | R1 `3aebb813bd33`，无 Memory | **+112.88%** | **+79.12%** | -16.54% | —（临时底座） |
| **plain** | 无 Memory、无 Skill | +67.44% | +33.69% | -13.68% | -45.44pp |

沪深300 同期全窗为 **+33.76%**。相对差是跨独立单次 rollout 的累计收益百分点差，
仅作描述性比较，不是晋升阶段的配对 sign test，也不能替代缺失的 Full R1 对照。

**数据来源**

| 格子 | Run / 来源 |
|------|------------|
| w/o self-evolve | `ablation/runs/mem_ablation_24m/arms/baseline`（24/24） |
| w/o skill | `ablation/runs/memonly_ablation_24m/arms/mem_only`（24/24） |
| w/o memory | `ablation/runs/skillonly_r1_24m/arms/r1_best`（24/24；技能注入 24/24，零 episode） |
| plain | `ablation/runs/plain_ablation_24m/arms/plain`（24/24；Pass@1 100%） |
| 完整主系统 Full R1 | 尚无完成产物 |

自进化时的 **dev/holdout 密封表**（选 Skill 用）仍见上文「24 月正式实验」与 `fitness_{dev,holdout}.*`；**消融主对照改用本节 full_24m**，避免晋升短窗与评估全窗混谈。

**解读**

1. **w/o memory**：R1 Skill 在 skill-only 协议下 24 月为 **+112.88%**（超额 +79.12%）；这是已完成 arm 中最高值，但不是 Full R1。
2. **Skill 的同开关比较**：Memory 关闭时，w/o memory 比 plain 高 **45.44pp**；Memory 开启时，w/o self-evolve 比 w/o skill 高 **12.18pp**。两组都支持 Skill 文本可能有贡献，但仍是不同 key 的单次 rollout。
3. **Memory 尚不能归因**：w/o skill 和 w/o self-evolve 低于 plain 或 w/o memory，不等于 Memory 有害；只有 Full R1 与 w/o memory 使用同一 R1 Skill，才能更直接估计 Memory 增量。
4. **统计边界**：所有全窗数字均来自 temp=1.0、单次 rollout。Full R1 完成前，只报告四个消融的绝对结果和临时底座差，不报告“相对完整系统”的因果降幅。

### 其余格子状态

| 格子 | key | 状态 |
|------|-----|------|
| w/o self-evolve | own_glm3 | **已完成**（24/24） |
| w/o skill | own_glm4 | **已完成**（24/24） |
| w/o memory | own_glm5 | **已完成**（24/24） |
| plain | own_glm6 | **已完成**（24/24） |
| 完整主系统 Full R1 | own_glm3 | 待重开（干净技能目录） |

矩阵脚本：`ablation/build_ablation_matrix.py` → `ablation/reports/matrix_24m.md`。

---

## 目录速查

| 路径 | 说明 |
|------|------|
| `MiroFlow/` | 原版 agent 运行时 |
| `MiroMemSkill/` | Memory + Skill 运行时 |
| `MiroMemSkill_hermes/` | Skill 自进化 fork（Hermes 思路 + 真实回测 fitness） |
| `ablation/` | 四个消融：w/o self-evolve、w/o skill、w/o memory、plain |
| `shared/ashare_open_stocks_glm52_20260714/` | 12 月冻结快照 + 对比报告 |
| `shared/ashare_open_stocks_glm52_24m_20260715/` | 24 月扩展快照（train/dev/holdout 任务 + manifest；DB 本地） |
| `任务要求.txt` | 研究任务与 ablation 要求 |

---

*最后更新：2026-07-16（四个消融均完成；w/o memory / R1 skill-only = +112.88%；Full R1 待运行）*
