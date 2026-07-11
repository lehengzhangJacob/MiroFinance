#!/usr/bin/env python3
"""Unit smoke tests for the Mem0-style memory rebuild (offline: no network).

Covers the failure modes that sank the previous implementation:
  - unparseable reflection output must be DISCARDED, never stored raw
  - retrieval is embedding-only with a strict before_month temporal filter
  - stance quota uses metadata-derived functional stance, not keyword scans
  - concurrent writers cannot corrupt the JSONL (fcntl lock)
  - consolidation (ADD/UPDATE/DELETE/NONE) keeps the bank non-redundant
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("GLM_API_KEY", "smoke-test-key")

from src.memory.context import build_memory_context_block, task_before_month  # noqa: E402
from src.memory.memory import (  # noqa: E402
    Mem0Memory,
    functional_stance,
    parse_task_month,
)
from src.memory.monthly_reflection import compute_month_features  # noqa: E402
from src.memory.skills import SkillLibrary  # noqa: E402
from src.memory.vector_store import VectorStore  # noqa: E402

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

        history = [json.loads(l) for l in open(store.history_path)]
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

        lines = [l for l in open(store.memories_path) if l.strip()]
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
    print("[OK] functional stance + month parsing")


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

        history_ops = [json.loads(l)["op"] for l in open(memory.store.history_path)]
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
        lines = [l for l in open(memory.outcomes_path) if l.strip()]
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
    print("[OK] month feature table computed from local cache")


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


if __name__ == "__main__":
    VectorStore.embed = fake_embed  # offline: deterministic embeddings
    test_vector_store_crud_and_history()
    test_concurrent_writes()
    test_month_filter_and_stance_quota()
    test_functional_stance_mapping()
    test_unparseable_extraction_discarded()
    test_consolidation_add_none_update_delete()
    test_context_block()
    test_skills()
    test_outcome_ledger_and_calibration()
    test_monthly_reflection_pipeline()
    test_month_grouping_order()
    test_month_features_table()
    test_context_block_with_calibration()
    print("\n=== All memory smoke tests passed ===")
