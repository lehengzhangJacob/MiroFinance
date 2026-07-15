# SPDX-FileCopyrightText: 2026 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from src.evolution.gates import run_static_gates
from src.evolution.types import SkillArtifact, split_skill_text


def test_split_skill_text_roundtrip(baseline_text):
    frontmatter, body = split_skill_text(baseline_text)
    assert "name: demo_skill" in frontmatter
    assert body.startswith("## 决策流程")


def test_split_requires_frontmatter():
    with pytest.raises(ValueError):
        split_skill_text("no frontmatter here")


def test_artifact_digest_is_content_addressed(baseline_text):
    a = SkillArtifact.from_text(baseline_text)
    b = SkillArtifact.from_text(baseline_text)
    assert a.digest == b.digest
    assert a.name == "demo_skill"
    c = SkillArtifact.from_text(baseline_text.replace("交叉验证", "批量验证"))
    assert c.digest != a.digest


def _candidate(baseline_text: str, new_body: str) -> SkillArtifact:
    frontmatter, _ = split_skill_text(baseline_text)
    return SkillArtifact.from_text(f"---\n{frontmatter}\n---\n\n{new_body}\n")


def test_gates_pass_for_honest_rewrite(baseline_text):
    baseline = SkillArtifact.from_text(baseline_text)
    candidate = _candidate(
        baseline_text,
        "## 决策流程\n\n1. 核对约束。\n2. 建立候选集并做流动性排除。\n3. 行业分散后确定 CASH。",
    )
    result = run_static_gates(baseline, candidate)
    assert result.passed, result.failures


def test_gates_reject_identical(baseline_text):
    baseline = SkillArtifact.from_text(baseline_text)
    result = run_static_gates(baseline, SkillArtifact.from_text(baseline_text))
    assert not result.passed
    assert any("identical" in f for f in result.failures)


def test_gates_reject_frontmatter_change(baseline_text):
    baseline = SkillArtifact.from_text(baseline_text)
    mutated = baseline_text.replace('version: "1.0"', 'version: "2.0"')
    result = run_static_gates(baseline, SkillArtifact.from_text(mutated))
    assert not result.passed
    assert any("frontmatter" in f for f in result.failures)


def test_gates_reject_new_ticker_and_dates(baseline_text):
    baseline = SkillArtifact.from_text(baseline_text)
    with_ticker = _candidate(baseline_text, "优先买入 600519.SH 并持有。")
    result = run_static_gates(baseline, with_ticker)
    assert any("ticker" in f for f in result.failures)

    with_date = _candidate(baseline_text, "在 2025-03-07 减仓。")
    result = run_static_gates(baseline, with_date)
    assert any("date" in f for f in result.failures)


def test_gates_reject_oversized_body(baseline_text):
    baseline = SkillArtifact.from_text(baseline_text)
    huge = _candidate(baseline_text, "步骤。" * 10_000)
    result = run_static_gates(baseline, huge)
    assert not result.passed
    assert any("long" in f or "grew" in f for f in result.failures)
