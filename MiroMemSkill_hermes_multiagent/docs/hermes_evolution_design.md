# MiroMemSkill_hermes：真实回测驱动的 Skill 自进化

本 fork 在 MiroMemSkill 运行时之上加了一个**离线 Skill 进化控制面**，借鉴
hermes-agent-self-evolution 的闭环纪律（候选生成 → 隔离评估 → 门禁 → 晋升），
但 fitness 完全来自确定性金融回放，不使用任何文本重合或 LLM 打分。

## 核心原则

1. **运行时唯一**：Agent、MCP 工具、点时数据库、组合校验全部沿用 MiroMemSkill；
   进化层只通过 `main.py common-benchmark` 子进程运行臂，不侵入 orchestrator。
2. **候选不可变**：每个候选是内容寻址的快照（`.evolution/candidates/<sha12>/`），
   生产 Skill 文件只能经 `SkillRegistry.promote`（CAS 校验）改写，可回滚。
3. **单变量对照**：baseline 臂与候选臂只有注入的 Skill 正文不同——相同模型、
   相同任务子集、相同快照 DB 副本、memory 全关、Skill 全文注入
   （`skill_preview_max_chars: 0`）。
4. **时间纪律**：12 个月按时间切分 train(6)/dev(3)/holdout(3)；
   probe = train 前 2 月。holdout 由 registry 的一次性租约强制（每候选一次）。
5. **多保真漏斗**：L0 静态门禁（免费）→ probe（2 月）→ dev（3 月）→
   holdout（3 月，封存）。差候选在便宜的层级被淘汰。

## 数据流

```
冻结快照(shared/ashare_open_stocks_glm52_20260714)
  └─ 任务子集(按月) + DB 副本(每臂 cp --reflink)
       └─ ArmRunner: main.py common-benchmark (skill-only 配置)
            └─ out/task_*_attempt_1.json  (boxed 组合)
                 ├─ fitness: eval_open_trader.replay(整手+费用+复利)
                 │    └─ 配对月度差 + 符号检验 + 硬否决 + 排名分
                 └─ feedback: 已结算事实(亏损月/集中度/执行问题)
                      └─ ReflectiveMutationGenerator(GLM) → 新候选正文
                           └─ L0 门禁 → registry 注册 → 下一轮
```

## 组件

| 文件 | 职责 |
|---|---|
| `src/evolution/types.py` | `SkillArtifact`（内容 SHA、frontmatter/正文拆分） |
| `src/evolution/registry.py` | 注册/lineage/CAS 晋升/回滚/holdout 租约 |
| `src/evolution/gates.py` | L0：frontmatter 不变、长度、新增股票代码/日期拦截 |
| `src/evolution/splits.py` | 时间切分与任务子集物化（DATA_DIR 布局） |
| `src/evolution/fitness.py` | 包装 `eval_open_trader` 回放；配对统计与硬否决 |
| `src/evolution/feedback.py` | 从回放结果生成结构化变异反馈 |
| `src/evolution/generators.py` | `CandidateGenerator` 协议 + GLM 反思变异；Hermes GEPA 适配器占位 |
| `src/evolution/controller.py` | 臂隔离、manifest、注入验证、配对评估 |
| `scripts/ashare/run_skill_evolution.py` | CLI：init/status/propose/run_arm/evaluate/smoke/holdout/promote/rollback |
| `config/memory/ashare_trader_hermes_skillonly.yaml` | skill-only 隔离臂配置 |
| `config/agent_ashare_trader_open_hermes_glm.yaml` | GLM-5.2 进化臂 agent 配置 |

## 臂隔离清单

每个臂独立拥有：

- `ASHARE_OPEN_DB` → 臂私有 DB 副本（fina_cache 写入互不可见）
- `DATA_DIR` → 臂私有任务子集
- `HERMES_SKILLS_DIR` → 候选物化目录（只含 1 个 .md，只读）
- `HERMES_STORE_DIR` + run-scoped namespace → 空 memory scratch
- `MEMSKILL_EMBEDDING_ENABLED=false` → 关键词匹配 Skill，确定性路由
- `arm_manifest.json` → skill SHA、月份、config、注入验证结果

回放 fitness 永远读**原始快照 DB**，不读臂副本。

## 硬否决与排名

- 硬否决（任何一条即拒绝）：组合输出无效月 > 0；最大回撤劣化 > 5pp。
- 排名分（仅对幸存者排序）：`mean_paired_diff_pp − 0.25 × max(0, 回撤劣化pp)`。
- 配对统计：逐月差、均值/标准差、精确双侧符号检验。
- probe/dev 结果只用于淘汰与排序；对外结论必须来自 holdout。

## CLI 用法

```bash
cd MiroMemSkill_hermes
python scripts/ashare/run_skill_evolution.py init
python scripts/ashare/run_skill_evolution.py smoke              # probe 闭环
python scripts/ashare/run_skill_evolution.py status
# 完整流程（多候选）：
python scripts/ashare/run_skill_evolution.py run_arm --run_id=r1 --arm=baseline --level=train
python scripts/ashare/run_skill_evolution.py propose --feedback_arm=<out> --level=train --n=4
python scripts/ashare/run_skill_evolution.py run_arm --run_id=r1 --arm=cand_x --candidate=<sha12> --level=dev
python scripts/ashare/run_skill_evolution.py evaluate --run_id=r1 --candidate=<sha12> --level=dev
python scripts/ashare/run_skill_evolution.py holdout --candidate=<sha12>   # 一次性
python scripts/ashare/run_skill_evolution.py promote --candidate=<sha12>
```

## 已知限制（后续工作）

- probe 层单次重复在 temp=1.0 下噪声很大；正式实验每层至少 2–3 次重复，
  并把重复均值作为配对单位。
- 反馈目前来自回放结果与 boxed 组合，尚未解析工具轨迹级失败
  （重复调用、点时参数错误）；`TaskTracer` 数据已在 attempt JSON 中，可扩展。
- `HermesGEPAGenerator` 为协议占位；外部 Hermes 仓库固化后可作为第二生成器
  做对比消融。
- 单一 Skill、单一任务族（open-universe trader）；扩展到 rank/pred 任务需要
  对应 evaluator 适配。
