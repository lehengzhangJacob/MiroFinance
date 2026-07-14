# SPDX-FileCopyrightText: 2026 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from src.evolution.registry import (
    HoldoutLeaseError,
    PromotionError,
    RegistryError,
    SkillRegistry,
)

SKILL_REL = "memory_bank/skills_ashare/demo_skill.md"


def make_registry(skill_repo) -> SkillRegistry:
    return SkillRegistry(skill_repo, SKILL_REL)


def candidate_text(baseline_text: str, marker: str) -> str:
    return baseline_text.replace("交叉验证", marker)


def test_init_and_reinit_guard(skill_repo):
    registry = make_registry(skill_repo)
    artifact = registry.init_baseline()
    assert registry.active_digest() == artifact.digest
    assert (registry.candidates_dir / artifact.short_id / "demo_skill.md").exists()
    with pytest.raises(RegistryError):
        registry.init_baseline()
    registry.init_baseline(force=True)  # explicit re-init allowed


def test_register_resolve_and_tamper_detection(skill_repo, baseline_text):
    registry = make_registry(skill_repo)
    baseline = registry.init_baseline()
    cand = registry.register_candidate(
        candidate_text(baseline_text, "批量验证"),
        parent_digest=baseline.digest,
        generator="test",
    )
    assert registry.resolve(cand.short_id)["digest"] == cand.digest
    assert registry.artifact(cand.short_id).body != baseline.body

    # Tampering with the materialized snapshot must be detected.
    path = registry.candidates_dir / cand.short_id / "demo_skill.md"
    path.chmod(0o644)
    path.write_text(path.read_text().replace("批量验证", "篡改"), encoding="utf-8")
    with pytest.raises(RegistryError, match="tamper"):
        registry.artifact(cand.short_id)


def test_register_requires_known_parent(skill_repo, baseline_text):
    registry = make_registry(skill_repo)
    registry.init_baseline()
    with pytest.raises(RegistryError, match="parent"):
        registry.register_candidate(
            candidate_text(baseline_text, "无父版本"),
            parent_digest="deadbeef" * 8,
            generator="test",
        )


def test_promote_cas_and_rollback(skill_repo, baseline_text):
    registry = make_registry(skill_repo)
    baseline = registry.init_baseline()
    cand = registry.register_candidate(
        candidate_text(baseline_text, "批量验证"),
        parent_digest=baseline.digest,
        generator="test",
    )
    registry.promote(cand.short_id, run_id="run1")
    assert registry.active_digest() == cand.digest
    assert "批量验证" in registry.skill_path.read_text(encoding="utf-8")
    assert registry.resolve(cand.short_id)["status"] == "promoted"
    assert list(registry.backups_dir.glob("*.md"))

    registry.rollback(baseline.short_id)
    assert registry.active_digest() == baseline.digest
    assert "交叉验证" in registry.skill_path.read_text(encoding="utf-8")


def test_promote_refuses_out_of_band_edit(skill_repo, baseline_text):
    registry = make_registry(skill_repo)
    baseline = registry.init_baseline()
    cand = registry.register_candidate(
        candidate_text(baseline_text, "批量验证"),
        parent_digest=baseline.digest,
        generator="test",
    )
    # Someone edits the production file outside the registry.
    registry.skill_path.write_text(
        baseline_text.replace("多路筛选", "手工改动"), encoding="utf-8"
    )
    with pytest.raises(PromotionError, match="outside the registry"):
        registry.promote(cand.short_id)


def test_holdout_lease_is_single_use(skill_repo, baseline_text):
    registry = make_registry(skill_repo)
    baseline = registry.init_baseline()
    cand = registry.register_candidate(
        candidate_text(baseline_text, "批量验证"),
        parent_digest=baseline.digest,
        generator="test",
    )
    registry.acquire_holdout_lease(cand.short_id, "holdout_run_1")
    with pytest.raises(HoldoutLeaseError):
        registry.acquire_holdout_lease(cand.short_id, "holdout_run_2")


def test_reports_and_status(skill_repo, baseline_text):
    registry = make_registry(skill_repo)
    baseline = registry.init_baseline()
    cand = registry.register_candidate(
        candidate_text(baseline_text, "批量验证"),
        parent_digest=baseline.digest,
        generator="test",
    )
    registry.attach_report(cand.short_id, "fitness_probe", {"score": 1.5})
    registry.update_status(cand.short_id, "probed")
    rec = registry.resolve(cand.short_id)
    assert rec["status"] == "probed"
    assert rec["reports"]["fitness_probe"]["score"] == 1.5
