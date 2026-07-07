#!/usr/bin/env python3
"""Unit smoke tests for MiroMemSkill memory/skill modules."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.memory.store import MemoryStore
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


if __name__ == "__main__":
    test_store()
    test_skills()
    test_context_block()
    print("\n=== All unit smoke tests passed ===")
