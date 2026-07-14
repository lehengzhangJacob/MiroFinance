---
name: ashare_announcement_search
description: 允许联网的A股研究任务中检索公告与财报；封闭点时回测禁止使用
version: "2.0"
applies_to: [online_ashare_research, announcement_lookup]
stock_universe: all_ashare
dependencies: [web_search, browser]
triggers:
  - 公告
  - 财报
  - 年报
  - 季报
  - 巨潮
  - 上交所
  - 深交所
  - 港股
---

## 步骤

仅在任务明确允许联网时使用。若任务要求只能调用本地点时工具、禁止网络搜索，
不要加载或执行本技能。

1. **确定实体**：股票代码/公司全称/Wind ticker；中文名用全称搜索。
2. **检索路径**：
   - cninfo.com.cn (巨潮资讯) → 公告/定期报告
   - sse.com.cn / szse.cn 交易所披露
   - 公司 IR 页面投资者关系
3. **筛选报告类型**：年报/半年报/季报/临时公告；按题目要求的时间范围过滤。
4. **提取关键字段**：营收、净利润、EPS、分红、持股比例等；记录报告期与公告日期。
5. **交叉验证**：至少两个来源一致后再作答。

## 常见陷阱

- 用了预告/快报而非正式年报数据
- 混淆报告期与公告发布日期
- 港股/A股代码混用
