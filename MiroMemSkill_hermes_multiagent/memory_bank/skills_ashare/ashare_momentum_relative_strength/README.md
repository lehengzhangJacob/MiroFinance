# ashare_momentum_relative_strength

严格点时的 A 股相对动量工具，与统一交易员 benchmark 的
`momentum_top4(20日相对动量)` 使用同一特征口径。

## 目录

```text
ashare_momentum_relative_strength/
├── SKILL.md
├── README.md
├── config.yaml
├── run.py
├── examples/examples.sh
└── tests/test_momentum_skill.py
```

核心计算位于 `src/utils/ashare_momentum.py`，同时供：

- 评测内 MCP 工具 `ashare_momentum_baseline`；
- 本目录离线 CLI；
- 自动化测试。

## 快速开始

```bash
cd MiroMemSkill/memory_bank/skills_ashare/ashare_momentum_relative_strength
python run.py --as-of 2024-07-01
python run.py --as-of 2024-07-01 --format csv
```

默认读取 `MiroMemSkill/data/ashare`，也可传 `--data-dir`。

## 计算口径

1. 所有行情先按 `trade_date <= as_of` 截断；
2. 个股使用 `close_qfq`，沪深300使用 `close`；
3. `rel20 = stock_return_20 - index_return_20`；
4. 按 `rel20` 降序、股票代码升序稳定排序；
5. 默认取前4只、每只25%，输出全池诊断与现金权重。

## 测试

```bash
python -m unittest discover -s tests -v
```
