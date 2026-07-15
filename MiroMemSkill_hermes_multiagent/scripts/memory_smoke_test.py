#!/usr/bin/env python3
"""Unit smoke tests for the Mem0-style memory rebuild (offline: no network).

Covers the failure modes that sank the previous implementation:
  - unparseable reflection output must be DISCARDED, never stored raw
  - retrieval enforces exact label exit dates (with before_month fallback)
  - stance quota uses metadata-derived functional stance, not keyword scans
  - concurrent writers cannot corrupt the JSONL (fcntl lock)
  - consolidation (ADD/UPDATE/DELETE/NONE) keeps the bank non-redundant
  - rolling rules need large samples and temporal validation, and resume cleanly
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("GLM_API_KEY", "smoke-test-key")

from src.memory.context import (  # noqa: E402
    build_memory_context_block,
    task_as_of_date,
    task_before_month,
)
from src.memory.feature_evidence import (  # noqa: E402
    build_feature_evidence_block,
    compute_task_feature_row,
    nearest_neighbors,
)
from src.memory.memory import (  # noqa: E402
    Mem0Memory,
    functional_stance,
    parse_task_month,
)
from src.memory.monthly_reflection import (  # noqa: E402
    _trailing_percentile,
    compute_month_features,
)
from src.memory.rolling_reflection import (  # noqa: E402
    OUTPERFORM,
    RollingRuleConfig,
    mine_rolling_rules,
)
from src.memory.skills import SkillLibrary  # noqa: E402
from src.memory.vector_store import VectorStore  # noqa: E402
from src.core.orchestrator import _filter_memskill_tool_definitions  # noqa: E402

# ---------------------------------------------------------------- fakes

_KEYWORD_AXES = ["动量", "估值", "换手", "财报", "信号"]


def fake_embed(self: VectorStore, text: str) -> list[float]:
    """Deterministic embedding: one axis per keyword family + a bias axis.

    Identical texts -> cosine 1.0; same-family texts -> high similarity;
    different families -> low similarity. Good enough to drive consolidation.
    """
    vec = [0.05] * (len(_KEYWORD_AXES) + 1)
    for i, kw in enumerate(_KEYWORD_AXES):
        vec[i] += 2.0 * text.count(kw)
    vec[-1] += 0.1 * (len(text) % 7)
    return vec


class ScriptedLLM:
    """Queue of canned _call_llm responses for Mem0Memory."""

    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls: list[str] = []

    def __call__(self, prompt: str, max_tokens: int = 2048, json_mode: bool = True) -> str:
        self.calls.append(prompt)
        if not self.responses:
            raise AssertionError("ScriptedLLM exhausted")
        return self.responses.pop(0)


def make_memory(tmp: str, responses: list[str]) -> tuple[Mem0Memory, ScriptedLLM]:
    store = VectorStore(store_dir=tmp, namespace="smoke")
    memory = Mem0Memory(store=store, api_key="smoke-test-key")
    llm = ScriptedLLM(responses)
    memory._call_llm = llm  # type: ignore[method-assign]
    return memory, llm


# ---------------------------------------------------------------- tests

def test_vector_store_crud_and_history() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = VectorStore(store_dir=tmp, namespace="crud")
        rec = store.add("动量教训一", metadata={"entry_month": "2024-07"}, source_task="t1")
        store.update(rec.id, "动量教训一(修订)", source_task="t2")
        assert store.get(rec.id).content == "动量教训一(修订)"
        assert store.delete(rec.id, source_task="t3")
        assert store.count() == 0

        history = [json.loads(line) for line in open(store.history_path)]
        assert [h["op"] for h in history] == ["ADD", "UPDATE", "DELETE"]
        assert history[1]["old_content"] == "动量教训一"
        print("[OK] vector store CRUD + history audit")


def test_concurrent_writes() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = VectorStore(store_dir=tmp, namespace="conc")

        def writer(k: int) -> None:
            for j in range(5):
                store.add(f"并发教训 {k}-{j} 动量" * 3, metadata={"entry_month": "2024-08"})

        threads = [threading.Thread(target=writer, args=(k,)) for k in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        lines = [line for line in open(store.memories_path) if line.strip()]
        assert len(lines) == 40, f"expected 40 lines, got {len(lines)}"
        for line in lines:
            json.loads(line)  # every line must be valid JSON
        print("[OK] 40 concurrent writes, zero corrupted lines")


def test_month_filter_and_stance_quota() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = VectorStore(store_dir=tmp, namespace="filt")
        memory = Mem0Memory(store=store, api_key="smoke-test-key")

        months = ["2024-07", "2024-08", "2024-09", "2025-01", ""]
        stances = ["bearish", "bearish", "bearish", "bullish", "neutral"]
        for i, (m, s) in enumerate(zip(months, stances)):
            store.add(
                f"动量教训{i} 关于相对沪深300动量的判断",
                metadata={"entry_month": m, "functional_stance": s},
            )

        # strict before-month: 2024-09 task sees only 07/08; missing-month hidden
        res = memory.search("动量 相对沪深300", top_k=5, before_month="2024-09")
        got_months = {r.metadata["entry_month"] for r, _ in res}
        assert got_months == {"2024-07", "2024-08"}, got_months

        # tight quota (v2): 3 bearish + 1 bullish + 1 neutral, top_k=3 ->
        # at most 1 per directional stance, so bullish must survive.
        res = memory.search("动量 相对沪深300", top_k=3, before_month="")
        got_stances = [r.metadata["functional_stance"] for r, _ in res]
        assert got_stances.count("bearish") <= 1, got_stances
        assert "bullish" in got_stances, got_stances
        print(f"[OK] before_month filter {sorted(got_months)}; tight quota stances={got_stances}")


def test_functional_stance_mapping() -> None:
    assert functional_stance("跑赢", "CORRECT") == "bullish"
    assert functional_stance("跑输", "CORRECT") == "bearish"
    assert functional_stance("跑输", "INCORRECT") == "bullish"  # counter-lesson pushes bullish
    assert functional_stance("跑赢", "INCORRECT") == "bearish"
    assert functional_stance("", "CORRECT") == "neutral"
    assert parse_task_month("ashare_300012_2025-04-01") == "2025-04"
    assert task_before_month("当前日期为 2025-04-01（收盘后）") == "2025-04"
    assert task_as_of_date("当前日期为 2025-04-01（收盘后）") == "20250401"
    print("[OK] functional stance + month parsing")


def test_exact_date_visibility() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = VectorStore(store_dir=tmp, namespace="dates")
        memory = Mem0Memory(store=store, api_key="smoke-test-key")
        store.add(
            "当近20日动量显著为正时使用已验证的大样本条件规则",
            metadata={
                "entry_month": "2024-07",
                "available_after": "20240805",
                "functional_stance": "neutral",
            },
        )
        assert memory.search(
            "动量", top_k=3, before_month="2024-08", before_date="20240801"
        ) == []
        assert len(
            memory.search(
                "动量", top_k=3, before_month="2024-08", before_date="20240805"
            )
        ) == 1

        memory.log_outcome(
            "t1", "2024-07", "跑赢", "CORRECT", available_after="20240805"
        )
        assert memory.calibration_block(
            "2024-08", min_samples=1, before_date="20240801"
        ) == ""
        assert "1 次" in memory.calibration_block(
            "2024-08", min_samples=1, before_date="20240805"
        )
        print("[OK] exact exit-date embargo for memories and calibration")


def test_unparseable_extraction_discarded() -> None:
    truncated = '{ "lessons": [ { "content": "当20日和60日相对沪深300超额收益均处'
    with tempfile.TemporaryDirectory() as tmp:
        memory, llm = make_memory(tmp, [truncated, truncated])  # initial + one retry
        ops = memory.add(
            question="预测…", answer="跑输", judge_result="INCORRECT",
            task_id="ashare_300012_2024-07-01",
        )
        assert ops == []
        assert memory.store.count() == 0, "raw text must never be stored"
        assert len(llm.calls) == 2
        print("[OK] truncated extraction JSON discarded after retry (nothing stored)")


def _extraction(content: str) -> str:
    return json.dumps({"lessons": [{"content": content, "tags": ["strategy"]}]}, ensure_ascii=False)


def test_consolidation_add_none_update_delete() -> None:
    lesson_a = "当20日相对动量为负且换手率萎缩时，动量延续跑输的概率上升，不宜仅凭低估值抄底"
    lesson_dup = lesson_a  # identical -> NONE
    lesson_b = "当20日相对动量为负但换手率放大时，动量教训需要结合估值分位交叉验证，单一动量不可靠"
    lesson_c = "当动量与换手率背离时，此前的动量延续规则被证明系统性失效，应优先检查财报"

    with tempfile.TemporaryDirectory() as tmp:
        # Task 1 (bank empty): direct ADD, no update-LLM call
        memory, llm = make_memory(tmp, [_extraction(lesson_a)])
        ops = memory.add("q", "跑输", "CORRECT", task_id="ashare_600418_2024-07-01")
        assert len(ops) == 1 and ops[0].startswith("ADD"), ops
        assert memory.store.count() == 1

        # Task 2: duplicate -> update LLM says NONE
        memory, llm = make_memory(
            tmp, [_extraction(lesson_dup), json.dumps({"action": "NONE", "target_id": "", "new_content": ""})]
        )
        ops = memory.add("q", "跑输", "CORRECT", task_id="ashare_600418_2024-08-01")
        assert ops and ops[0].startswith("NONE"), ops
        assert memory.store.count() == 1

        # Task 3: same family -> UPDATE merges into target
        target_id = memory.store.all_records()[0].id
        merged = lesson_a + "；但换手率放大时该规则不适用"
        memory, llm = make_memory(
            tmp,
            [
                _extraction(lesson_b),
                json.dumps({"action": "UPDATE", "target_id": target_id, "new_content": merged}, ensure_ascii=False),
            ],
        )
        ops = memory.add("q", "跑输", "INCORRECT", task_id="ashare_600418_2024-09-02")
        assert ops and ops[0].startswith("UPDATE"), ops
        recs = memory.store.all_records()
        assert len(recs) == 1 and recs[0].content == merged
        # merged metadata: later month wins, disagreeing stances -> mixed
        assert recs[0].metadata["entry_month"] == "2024-09"
        assert recs[0].metadata["functional_stance"] == "mixed"

        # Task 4: contradiction -> DELETE old + ADD candidate
        target_id = recs[0].id
        memory, llm = make_memory(
            tmp,
            [
                _extraction(lesson_c),
                json.dumps({"action": "DELETE", "target_id": target_id, "new_content": lesson_c}, ensure_ascii=False),
            ],
        )
        ops = memory.add("q", "跑赢", "CORRECT", task_id="ashare_600418_2024-10-08")
        assert ops and ops[0].startswith("DELETE"), ops
        recs = memory.store.all_records()
        assert len(recs) == 1 and recs[0].content == lesson_c

        history_ops = [json.loads(line)["op"] for line in open(memory.store.history_path)]
        assert history_ops == ["ADD", "UPDATE", "DELETE", "ADD"], history_ops
        print("[OK] consolidation pipeline: ADD -> NONE -> UPDATE(merge) -> DELETE+ADD; history complete")


def test_context_block() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = VectorStore(store_dir=tmp, namespace="ctx")
        memory = Mem0Memory(store=store, api_key="smoke-test-key")
        store.add(
            "当动量为负时的教训（来自2024-07任务）",
            metadata={"entry_month": "2024-07", "functional_stance": "bearish", "tags": ["momentum"]},
        )
        store.add(
            "当动量为正时的教训（来自2025-05任务，未来月份不得注入）",
            metadata={"entry_month": "2025-05", "functional_stance": "bullish", "tags": ["momentum"]},
        )
        skills_dir = ROOT / "memory_bank" / "skills_ashare"
        lib = SkillLibrary(skills_dir=skills_dir)
        block = build_memory_context_block(
            task_description=(
                "你是一名 A 股量化研究员。当前日期为 2025-04-01（收盘后）。"
                "请预测股票的动量表现相对沪深300是「跑赢」还是「跑输」。数据使用规则（务必遵守）：…"
            ),
            store=memory,
            skill_lib=lib,
            inject_top_k=3,
            skill_top_k=2,
        )
        assert "2024-07" in block, "earlier-month lesson missing"
        assert "2025-05" not in block, "future-month lesson leaked into injection"
        assert "NOT direction priors" in block
        assert "Recommended Skills" in block
        print(f"[OK] context block: temporal filter + framing + skills ({len(block)} chars)")


def test_skills() -> None:
    skills_dir = ROOT / "memory_bank" / "skills_ashare"
    lib = SkillLibrary(skills_dir=skills_dir)
    assert len(lib.list_skills()) >= 3, "expected ashare skills"
    matches = lib.match("预测股票相对沪深300指数是跑赢还是跑输，考察动量与估值", top_k=2)
    assert matches, "skill matching should return results"
    text = lib.load_skill_text(matches[0][0].name)
    assert len(text) > 100
    print(f"[OK] skills: {len(lib.list_skills())} loaded, top match={matches[0][0].name}")


# ------------------------------------------------------------- v2 additions

def test_outcome_ledger_and_calibration() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = VectorStore(store_dir=tmp, namespace="cal")
        memory = Mem0Memory(store=store, api_key="smoke-test-key")
        i = 0
        for pred, judge, n in [
            ("跑输", "CORRECT", 6), ("跑输", "INCORRECT", 4),
            ("跑赢", "CORRECT", 2), ("跑赢", "INCORRECT", 4),
        ]:
            for _ in range(n):
                memory.log_outcome(f"t{i}", "2024-07", pred, judge)
                i += 1
        memory.log_outcome("t0", "2024-07", "跑输", "CORRECT")  # dup: ignored
        memory.log_outcome("bad", "2024-07", "", "CORRECT")  # invalid: ignored
        lines = [line for line in open(memory.outcomes_path) if line.strip()]
        assert len(lines) == 16, f"ledger rows {len(lines)}"

        block = memory.calibration_block("2024-08")
        assert "16 次" in block, block
        assert "跑输 10 次（62%）" in block, block
        assert "命中 60%（6/10）" in block, block
        assert "命中 33%（2/6）" in block, block
        assert "跑输 62%、跑赢 38%" in block, block  # labels recovered from pred x judge
        assert memory.calibration_block("2024-07") == "", "no earlier months -> empty"
        print("[OK] outcome ledger idempotent; calibration numbers exact; first month empty")


def test_monthly_reflection_pipeline() -> None:
    lesson_a = "当决策日 rel20 相对超额低于 -5% 且换手率分位低于 20% 时，本月截面 5/6 跑输（单月截面证据）"
    lesson_b = "当 ml_rank 处于前 4 且财报增速为正时，本月截面 4/5 跑赢（单月截面证据）"
    extraction = json.dumps(
        {"lessons": [{"content": lesson_a, "tags": ["monthly"]}, {"content": lesson_b, "tags": ["monthly"]}]},
        ensure_ascii=False,
    )
    add_decision = json.dumps({"action": "ADD", "target_id": "", "new_content": ""})
    with tempfile.TemporaryDirectory() as tmp:
        memory, llm = make_memory(tmp, [extraction, add_decision, add_decision])
        ops = memory.add_monthly("2024-07", "ts_code,rel20\nA,1.0", 16)
        assert len(ops) == 2 and all("ADD" in o or "UPDATE" in o for o in ops), ops
        recs = memory.store.all_records()
        assert len(recs) == 2
        for r in recs:
            assert r.metadata["entry_month"] == "2024-07"
            assert r.metadata["functional_stance"] == "neutral"
            assert r.metadata["source"] == "monthly_reflection"
        # visible to later months only
        assert len(memory.search("rel20 换手率", top_k=3, before_month="2024-08")) == 2
        assert memory.search("rel20 换手率", top_k=3, before_month="2024-07") == []

        # empty-lessons month stores nothing
        memory2, _ = make_memory(tmp, [json.dumps({"lessons": []})])
        assert memory2.add_monthly("2024-08", "t", 16) == []
        print("[OK] monthly reflection: 2 cross-sectional lessons stored, neutral stance, temporal visibility")


def test_month_grouping_order() -> None:
    ids = [
        "ashare_x_2024-08-01", "ashare_y_2024-07-01",
        "ashare_z_2024-07-01", "ashare_w_2025-01-02",
    ]
    groups: dict[str, list[str]] = {}
    for tid in ids:
        groups.setdefault(parse_task_month(tid), []).append(tid)
    order = [tid for m in sorted(groups) for tid in sorted(groups[m])]
    assert order == [
        "ashare_y_2024-07-01", "ashare_z_2024-07-01",
        "ashare_x_2024-08-01", "ashare_w_2025-01-02",
    ], order
    print("[OK] month grouping: chronological barrier order")


def test_month_features_table() -> None:
    stocks = [
        {"ts_code": "300012.SZ", "stock_name": "华测检测", "label": "跑赢",
         "predicted": "跑输", "judge_result": "INCORRECT"},
        {"ts_code": "600418.SH", "stock_name": "江淮汽车", "label": "跑输",
         "predicted": "跑输", "judge_result": "CORRECT"},
    ]
    csv_str, n = compute_month_features("20240701", stocks)
    assert n == 2
    header = csv_str.splitlines()[0]
    for col in ("rel5", "rel20", "rel60", "pe_pct", "pb_pct", "turn_pct", "ml_rank", "label"):
        assert col in header, header
    assert "华测检测" in csv_str and "江淮汽车" in csv_str
    assert _trailing_percentile(pd.Series([1.0, 2.0, None])) is None
    assert _trailing_percentile(pd.Series([1.0, 2.0, 3.0])) == 100.0
    print("[OK] month feature table computed from local cache")


def _synthetic_rolling_samples() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for month in range(1, 9):
        entry_date = f"2024{month:02d}01"
        exit_date = f"2024{month:02d}20"
        for stock in range(16):
            condition = stock < 8
            # The condition is perfect; outside it the labels are balanced.
            label = OUTPERFORM if condition or stock % 2 == 0 else "跑输"
            rows.append(
                {
                    "task_id": f"synthetic_{month:02d}_{stock:02d}",
                    "entry_month": f"2024-{month:02d}",
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "ts_code": f"{stock:06d}.SZ",
                    "rel20": 10.0 if condition else -1.0,
                    "label": label,
                    "pred": label,
                    "correct": "Y",
                }
            )
    return rows


def test_rolling_large_sample_and_idempotency() -> None:
    samples = _synthetic_rolling_samples()
    assert mine_rolling_rules(samples[:16], "20240201") == []

    inverted_validation = []
    for row in samples:
        changed = dict(row)
        if changed["entry_month"] in ("2024-07", "2024-08") and changed["rel20"] == 10.0:
            changed["label"] = "跑输"
        inverted_validation.append(changed)
    assert mine_rolling_rules(inverted_validation, "20240901") == []

    rules = mine_rolling_rules(samples, "20240901", RollingRuleConfig())
    assert len(rules) == 1, [rule.rule_id for rule in rules]
    rule = rules[0]
    assert rule.metadata["source"] == "rolling_statistical"
    assert rule.metadata["validation_support"] >= 8
    assert rule.metadata["available_after"] == "20240820"
    assert "时间验证样本" in rule.content

    # A future, inverted cross-section must not affect a September-1 refresh.
    future = []
    for stock in range(16):
        future.append(
            {
                "task_id": f"future_{stock}",
                "entry_month": "2024-09",
                "entry_date": "20240901",
                "exit_date": "20240920",
                "ts_code": f"{stock:06d}.SZ",
                "rel20": 10.0,
                "label": "跑输",
            }
        )
    assert [r.rule_id for r in mine_rolling_rules(samples + future, "20240901")] == [
        rule.rule_id
    ]

    with tempfile.TemporaryDirectory() as tmp:
        store = VectorStore(store_dir=tmp, namespace="rolling")
        memory = Mem0Memory(store=store, api_key="smoke-test-key")
        assert memory.log_samples(samples) == (128, 0)
        assert memory.log_samples(samples) == (0, 0)

        first_ops = memory.refresh_rolling("20240901")
        assert any(op.startswith("ADD") for op in first_ops), first_ops
        history_count = len([line for line in open(store.history_path) if line.strip()])
        second_ops = memory.refresh_rolling("20240901")
        assert second_ops == ["UNCHANGED: 1 validated rolling rules"], second_ops
        assert len([line for line in open(store.history_path) if line.strip()]) == history_count

        memory.log_samples(future)
        assert memory.refresh_rolling("20240901") == [
            "UNCHANGED: 1 validated rolling rules"
        ]
        assert len(
            memory.search(
                "动量", top_k=3, before_month="2024-09", before_date="20240901"
            )
        ) == 1

        # Rewinding a completed namespace must remove future-state rules.
        rewind_ops = memory.refresh_rolling("20240101")
        assert any(op.startswith("DELETE") for op in rewind_ops), rewind_ops
        assert store.count() == 0

    with tempfile.TemporaryDirectory() as tmp:
        store = VectorStore(store_dir=tmp, namespace="rolling_guard")
        memory = Mem0Memory(store=store, api_key="smoke-test-key")
        memory.log_samples(inverted_validation)
        assert memory.refresh_rolling("20240901") == []
        block = build_memory_context_block(
            task_description=(
                "当前日期为 2024-09-01（收盘后）。预测股票相对沪深300表现。"
                "数据使用规则（务必遵守）：…"
            ),
            store=memory,
            skill_lib=None,
            memory_enabled=True,
            skill_enabled=False,
            rolling_status_enabled=True,
            rolling_min_samples=64,
            rolling_min_months=6,
        )
        assert "没有任何条件规则" in block, block
        assert "不得自行从历史样本编造方向规则" in block, block
        print("[OK] rolling rules: large sample, temporal holdout, idempotent resume")


def test_context_block_with_calibration() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = VectorStore(store_dir=tmp, namespace="ctx2")
        memory = Mem0Memory(store=store, api_key="smoke-test-key")
        for i in range(16):
            memory.log_outcome(f"t{i}", "2024-07", "跑输" if i < 12 else "跑赢",
                               "CORRECT" if i % 2 == 0 else "INCORRECT")
        store.add(
            "当 rel20 低于 -5% 时本月截面多数跑输（单月截面证据）",
            metadata={"entry_month": "2024-07", "functional_stance": "neutral"},
        )
        block = build_memory_context_block(
            task_description="当前日期为 2024-08-01（收盘后）。预测股票相对沪深300动量表现。数据使用规则（务必遵守）：…",
            store=memory,
            skill_lib=None,
            inject_top_k=3,
            skill_enabled=False,
            calibration_enabled=True,
        )
        assert "自我校准统计" in block, "calibration section missing"
        assert "rel20" in block, "monthly lesson missing"
        # first month: no calibration, no lessons
        block_m1 = build_memory_context_block(
            task_description="当前日期为 2024-07-01（收盘后）。预测股票相对沪深300动量表现。数据使用规则（务必遵守）：…",
            store=memory,
            skill_lib=None,
            inject_top_k=3,
            skill_enabled=False,
            calibration_enabled=True,
        )
        assert "自我校准统计" not in block_m1
        print("[OK] context block: calibration injected for later months only")


def test_direction_free_reliability_and_tool_filter() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = VectorStore(store_dir=tmp, namespace="reliability")
        memory = Mem0Memory(store=store, api_key="smoke-test-key")
        for i in range(16):
            memory.log_outcome(
                f"t{i}",
                "2024-07",
                "跑输" if i < 15 else "跑赢",
                "CORRECT" if i % 2 == 0 else "INCORRECT",
                available_after="20240731",
            )
        block = memory.reliability_block(
            "2024-08", min_samples=16, before_date="20240801"
        )
        assert "历史总体命中率" in block
        assert "预测分布" not in block
        assert "跑输" not in block and "跑赢" not in block

        definitions = [
            {
                "name": "tool-memskill",
                "tools": [
                    {"name": "memory_search"},
                    {"name": "memory_save"},
                    {"name": "skill_load"},
                ],
            }
        ]
        filtered = _filter_memskill_tool_definitions(
            definitions,
            expose_memory_search=False,
            expose_memory_save=False,
        )
        assert [tool["name"] for tool in filtered[0]["tools"]] == ["skill_load"]
        assert len(definitions[0]["tools"]) == 3, "filter must not mutate cached definitions"
        print("[OK] reliability block has no direction anchor; unavailable memory tools hidden")


def test_feature_conditioned_evidence() -> None:
    current = {
        "rel5": 1.0,
        "rel20": 4.0,
        "rel60": 8.0,
        "pe_pct": 50.0,
        "pb_pct": 40.0,
        "turn_pct": 60.0,
        "ml_rank": 3,
    }
    samples = []
    for i in range(12):
        row = dict(current)
        for offset, key in enumerate(("rel5", "rel20", "rel60", "pe_pct")):
            row[key] = float(row[key]) + (i - 5.5) * (offset + 1) / 20
        row.update(
            {
                "task_id": f"n{i}",
                "entry_month": "2024-07",
                "entry_date": f"202407{i + 1:02d}",
                "exit_date": "20240801",
                "ts_code": f"{i:06d}.SZ",
                "label": OUTPERFORM,
            }
        )
        samples.append(row)
    neighbors = nearest_neighbors(current, samples, k=12)
    assert len(neighbors) == 12
    assert all(row["common_features"] >= 3 for row in neighbors)

    with tempfile.TemporaryDirectory() as tmp:
        store = VectorStore(store_dir=tmp, namespace="feature")
        memory = Mem0Memory(store=store, api_key="smoke-test-key")
        real_current = compute_task_feature_row("20250401", "300012.SZ", "华测检测")
        synthetic = []
        for i in range(12):
            row = {
                "task_id": f"feature_{i}",
                "entry_month": "2024-07",
                "entry_date": f"202407{i + 1:02d}",
                "exit_date": "20240801",
                "ts_code": f"{i:06d}.SZ",
                "label": OUTPERFORM,
            }
            for offset, key in enumerate(
                ("rel5", "rel20", "rel60", "pe_pct", "pb_pct", "turn_pct", "ml_rank")
            ):
                value = real_current.get(key)
                row[key] = (
                    float(value) + (i - 5.5) * (offset + 1) / 100
                    if value not in ("", None)
                    else ""
                )
            synthetic.append(row)
        memory.log_samples(synthetic)
        evidence, audit = build_feature_evidence_block(
            memory,
            entry_date="20250401",
            ts_code="300012.SZ",
            stock_name="华测检测",
            k=12,
            min_neighbors=8,
        )
        assert "相似历史样本" in evidence
        assert "跑赢" in evidence
        assert audit["eligible_samples"] == 12

        for i, row in enumerate(synthetic):
            row["label"] = OUTPERFORM if i % 2 == 0 else "跑输"
        memory.log_samples(synthetic)
        evidence, audit = build_feature_evidence_block(
            memory,
            entry_date="20250401",
            ts_code="300012.SZ",
            stock_name="华测检测",
            k=12,
            min_neighbors=8,
        )
        assert evidence == ""
        assert audit["neighbor_direction"] == "inconclusive"
        print("[OK] post-tool feature evidence is point-in-time and withholds noisy majorities")


def test_official_mem0_qdrant_integration() -> None:
    """Opt-in live integration test; requires Qdrant and a real GLM key."""
    if os.environ.get("MEM0_QDRANT_INTEGRATION", "").lower() not in {
        "1",
        "true",
        "yes",
    }:
        print("[SKIP] official Mem0/Qdrant integration (set MEM0_QDRANT_INTEGRATION=1)")
        return
    if os.environ.get("GLM_API_KEY") in {"", "smoke-test-key", None}:
        raise RuntimeError("MEM0_QDRANT_INTEGRATION requires a real GLM_API_KEY")

    from src.memory.official_mem0_store import OfficialMem0Store
    from src.memory.rolling_reflection import ValidatedRule

    left = OfficialMem0Store(store_dir=ROOT / "memory_bank", namespace="__qdrant_smoke_left__")
    right = OfficialMem0Store(store_dir=ROOT / "memory_bank", namespace="__qdrant_smoke_right__")
    left.reset_namespace()
    right.reset_namespace()
    try:
        old = left.add(
            "已到期规则：相对动量证据必须结合估值确认。",
            metadata={
                "source": "rolling_statistical",
                "rule_id": "old",
                "entry_month": "2024-07",
                "available_after": "20240729",
                "functional_stance": "neutral",
                "q_value": 0.05,
                "validation_lift": 0.2,
                "validation_support": 10,
            },
        )
        left.add(
            "未来规则：在到期日前绝不能被检索。",
            metadata={
                "source": "rolling_statistical",
                "rule_id": "future",
                "entry_month": "2024-09",
                "available_after": "20240929",
                "functional_stance": "neutral",
                "q_value": 0.01,
                "validation_lift": 0.3,
                "validation_support": 10,
            },
        )
        assert left.count() == 2
        assert right.count() == 0
        assert right.get(old.id) is None
        updated = left.update(
            old.id,
            old.content,
            metadata_patch={"audit_revision": 2},
        )
        assert updated and updated.metadata["audit_revision"] == 2
        assert {"ADD", "UPDATE"}.issubset(
            {item.get("event") for item in left.history(old.id)}
        )

        memory = Mem0Memory(left)
        visible = memory.search(
            "条件规则",
            top_k=5,
            before_month="2024-08",
            before_date="20240801",
        )
        assert [record.metadata.get("rule_id") for record, _ in visible] == ["old"]

        if os.environ.get("MEM0_QDRANT_INFERENCE_TEST", "").lower() in {
            "1",
            "true",
            "yes",
        }:
            inferred_memory = Mem0Memory(right)
            inferred_ops = inferred_memory.add(
                question=(
                    "预测某股票未来20日相对沪深300表现；"
                    "证据为相对动量为正但估值处于高位。"
                ),
                answer="依据不足，最终判断：跑赢",
                judge_result="INCORRECT",
                task_id="ashare_test_2024-07-01",
                ts_code="000001.SZ",
                available_after="20240729",
            )
            inferred_records = right.all_records()
            assert inferred_ops and inferred_records
            assert all(
                record.metadata.get("available_after")
                == "2024-07-29T00:00:00Z"
                for record in inferred_records
            )

        metadata = {
            "rule_id": "stable",
            "condition": {"feature": "rel20", "operator": ">", "threshold": 2.0},
            "direction": OUTPERFORM,
            "entry_month": "2024-07",
            "available_after": "20240729",
            "generated_for_date": "20240801",
            "source": "rolling_statistical",
            "source_months": ["2024-07"],
            "source_tasks": [],
            "functional_stance": "neutral",
            "tags": ["rolling", "validated", "rel20"],
            "train_support": 20,
            "train_accuracy": 0.7,
            "validation_support": 10,
            "validation_accuracy": 0.7,
            "validation_lift": 0.2,
            "q_value": 0.05,
            "total_support": 30,
            "total_accuracy": 0.7,
            "support_months": 3,
        }
        rule = ValidatedRule(
            "stable",
            "当近20日相对收益大于2%时，历史验证更常跑赢，仅作辅助。",
            metadata,
        )
        assert memory._sync_rolling_rules([rule], "20240801")
        assert memory._sync_rolling_rules([rule], "20240802") == [
            "UNCHANGED: 1 validated rolling rules"
        ]
        print("[OK] official Mem0/Qdrant CRUD, namespace, date filter, rolling idempotency")
    finally:
        left.reset_namespace()
        right.reset_namespace()


if __name__ == "__main__":
    VectorStore.embed = fake_embed  # offline: deterministic embeddings
    test_vector_store_crud_and_history()
    test_concurrent_writes()
    test_month_filter_and_stance_quota()
    test_functional_stance_mapping()
    test_exact_date_visibility()
    test_unparseable_extraction_discarded()
    test_consolidation_add_none_update_delete()
    test_context_block()
    test_skills()
    test_outcome_ledger_and_calibration()
    test_monthly_reflection_pipeline()
    test_month_grouping_order()
    test_month_features_table()
    test_rolling_large_sample_and_idempotency()
    test_context_block_with_calibration()
    test_direction_free_reliability_and_tool_filter()
    test_feature_conditioned_evidence()
    test_official_mem0_qdrant_integration()
    print("\n=== All memory smoke tests passed ===")
