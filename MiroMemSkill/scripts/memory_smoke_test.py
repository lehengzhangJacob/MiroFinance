#!/usr/bin/env python3
"""Unit smoke tests for MiroMemSkill memory/skill modules."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.memory.store import MemoryStore, stance_of
from src.memory.skills import SkillLibrary
from src.memory.context import build_memory_context_block


def test_store() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = MemoryStore(store_dir=tmp, namespace="test")
        store.add(
            "When querying US bank credit, prefer FRED and check seasonally adjusted flag.",
            kind="episodic",
            tags=["finance", "strategy"],
            compute_embedding=False,
        )
        store.add(
            "PBOC publishes China external debt statistics quarterly.",
            kind="semantic",
            tags=["china", "source"],
            compute_embedding=False,
        )
        results = store.search("China external debt source", top_k=2)
        assert results, "BM25 search should return results"
        print(f"[OK] store: {len(results)} hits, counts={store.count()}")


def test_skills() -> None:
    skills_dir = ROOT / "memory_bank" / "skills"
    lib = SkillLibrary(skills_dir=skills_dir)
    assert len(lib.list_skills()) >= 5, "Expected at least 5 seed skills"
    matches = lib.match("2024年中国外债 historical lookup", top_k=2)
    assert matches, "Skill matching should return results"
    text = lib.load_skill_text(matches[0][0].name)
    assert len(text) > 100
    print(f"[OK] skills: {len(lib.list_skills())} loaded, top match={matches[0][0].name}")


def test_context_block() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = MemoryStore(store_dir=tmp, namespace="ctx")
        store.add(
            "For FinSearchComp T2, verify as-of date and unit before answering.",
            kind="episodic",
            tags=["finsearch"],
            compute_embedding=False,
        )
        skills_dir = ROOT / "memory_bank" / "skills"
        lib = SkillLibrary(skills_dir=skills_dir)
        block = build_memory_context_block(
            task_description="2024年全年中国外债净流入是多少？",
            store=store,
            skill_lib=lib,
            inject_top_k=2,
            skill_top_k=1,
        )
        assert "Relevant Experience" in block or "Recommended Skills" in block
        print(f"[OK] context block: {len(block)} chars")


def test_stance_of() -> None:
    assert stance_of("当20日动量为负时，倾向于预测该股票跑输指数") == "bearish"
    assert stance_of("多窗口动量全面占优时，倾向于预测跑赢指数") == "bullish"
    assert stance_of("X情况时跑赢，Y情况时跑输，需结合换手率判断") == "neutral"
    assert stance_of("调用 ashare_financials 时注意公告日必须早于 as_of") == "neutral"
    print("[OK] stance_of: bullish/bearish/neutral classification")


def test_search_balanced() -> None:
    """5 bearish + 1 bullish lessons: balanced top-3 must not be all-bearish."""
    with tempfile.TemporaryDirectory() as tmp:
        store = MemoryStore(store_dir=tmp, namespace="bal")
        bearish = [
            "当股票短期动量为负时，倾向于预测该股票跑输沪深300指数",
            "当股票估值高位且盈利下滑时，倾向于预测该股票跑输指数",
            "当股票换手率萎缩且动量走弱时，预测该股票跑输指数概率高",
            "当股票中长期动量透支时，均值回归导致跑输指数的概率较高",
            "当股票财报增速停滞时，倾向于预测该股票跑输指数",
        ]
        bullish = ["当股票多窗口动量全面占优且换手放大时，倾向于预测该股票跑赢指数"]
        for content in bearish + bullish:
            store.add(content, kind="episodic", compute_embedding=False)

        results = store.search_balanced("预测股票相对沪深300指数动量表现", top_k=3)
        assert len(results) == 3, f"expected 3 results, got {len(results)}"
        stances = [stance_of(e.content) for e, _ in results]
        assert stances.count("bearish") <= 2, f"bearish quota violated: {stances}"
        assert "bullish" in stances, f"bullish lesson missing from top-3: {stances}"
        print(f"[OK] search_balanced: stances={stances}")


def test_context_block_neutral_framing() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = MemoryStore(store_dir=tmp, namespace="frame")
        store.add(
            "当股票动量为负时，倾向于预测该股票跑输指数",
            kind="episodic",
            compute_embedding=False,
        )
        block = build_memory_context_block(
            task_description="预测股票600519相对沪深300指数表现",
            store=store,
            skill_lib=None,
            inject_top_k=2,
            skill_enabled=False,
        )
        assert "NOT direction priors" in block, "neutral framing line missing"
        print("[OK] context block includes neutral framing")


if __name__ == "__main__":
    test_store()
    test_skills()
    test_context_block()
    test_stance_of()
    test_search_balanced()
    test_context_block_neutral_framing()
    print("\n=== All unit smoke tests passed ===")
