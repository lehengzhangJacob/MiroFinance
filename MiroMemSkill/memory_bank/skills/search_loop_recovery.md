---
name: search_loop_recovery
description: 搜索循环自救——N次无进展时收敛策略、避免无限检索
triggers:
  - 搜索
  - search
  - 找不到
  - unavailable
  - 无法
  - 不确定
---

## 步骤

1. **设定搜索预算**：同一子问题最多 3 轮不同关键词；无新信息则换策略或换源。
2. **关键词变体**：中英文切换、同义词、官方缩写(如 PBOC/人民银行)、加 site: 限定。
3. **降级策略**：
   - 权威源 → 次级源(新闻/研报) → 明确标注不确定性
   - 无法获取精确值时，给出最接近的可验证估计并说明依据
4. **禁止行为**：
   - 重复相同搜索 >2 次
   - 返回 "Data Unavailable" 而不尝试替代源
   - 超过 max_turns 仍不输出 boxed 答案
5. **收敛输出**：即使不完全确定，也要给出最佳候选答案(符合 FinSearchComp 评分要求)。

## 常见陷阱

- GAIA/FinSearchComp 中因无限搜索被 kill
- 过早放弃(只搜一次就报 unavailable)
- 不调用 reasoning 工具整合已有信息
