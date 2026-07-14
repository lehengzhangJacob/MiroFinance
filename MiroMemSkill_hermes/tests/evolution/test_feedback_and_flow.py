# SPDX-FileCopyrightText: 2026 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

"""Feedback rendering plus a mock end-to-end control flow (no API, no arms)."""

import json

from src.evolution.controller import EvolutionController, render_report_markdown
from src.evolution.feedback import build_feedback
from src.evolution.fitness import fitness_report
from src.evolution.gates import run_static_gates
from src.evolution.registry import SkillRegistry
from src.evolution.types import SkillArtifact

SKILL_REL = "memory_bank/skills_ashare/demo_skill.md"


def _arm(months: list[dict], total: float) -> dict:
    return {
        "run_dir": "arm",
        "months": months,
        "total_return": total,
        "index_return": 0.005,
        "excess_return": total - 0.005,
        "max_drawdown": -0.08,
        "worst_month": min(float(m["net"]) for m in months),
        "win_rate": 0.5,
        "fees": 2500.0,
        "invalid_months": [],
    }


def test_build_feedback_mentions_facts():
    months = [
        {
            "as_of": "2024-07-01",
            "net": -0.031,
            "index": -0.025,
            "holdings": 8,
            "cash_w": 0.10,
            "missing": [],
            "unfilled": ["600519.SH"],
        },
        {
            "as_of": "2024-08-01",
            "net": 0.042,
            "index": -0.041,
            "holdings": 5,
            "cash_w": 0.25,
            "missing": [],
            "unfilled": [],
        },
    ]
    allocations = {
        "2024-07-01": {"600519.SH": 0.25, "000001.SZ": 0.20, "CASH": 0.55},
        "2024-08-01": {"300750.SZ": 0.25, "CASH": 0.75},
    }
    text = build_feedback(_arm(months, 0.01), allocations, label="现行 Skill")
    assert "2024-07-01" in text and "整手未成交:600519.SH" in text
    assert "单股最大权重 25%" in text
    assert "亏损月 1 个" in text
    assert "不得写入任何具体股票代码" in text


class MockGenerator:
    """Deterministic stand-in for the GLM mutation generator."""

    name = "mock"

    def __init__(self, bodies: list[str]):
        self.bodies = bodies

    def propose(self, baseline, feedback, n=1):
        return self.bodies[:n]


def test_mock_end_to_end_flow(skill_repo, baseline_text):
    """propose -> L0 -> register -> paired fitness -> attach -> promote."""
    registry = SkillRegistry(skill_repo, SKILL_REL)
    baseline = registry.init_baseline()

    feedback = "训练月亏损集中在追高动量股。"
    generator = MockGenerator(
        ["## 决策流程\n\n1. 核对约束。\n2. 剔除近期垂直拉升股。\n3. 行业分散并确定 CASH。"]
    )
    bodies = generator.propose(registry.artifact(baseline.digest), feedback, n=1)
    text = f"---\n{baseline.frontmatter}\n---\n\n{bodies[0]}\n"
    candidate_artifact = SkillArtifact.from_text(text, parent_digest=baseline.digest)

    gate = run_static_gates(baseline, candidate_artifact)
    assert gate.passed, gate.failures

    registered = registry.register_candidate(
        text, parent_digest=baseline.digest, generator=generator.name
    )

    base_months = [
        {"as_of": "2024-07-01", "net": -0.031, "index": -0.025, "capital": 0.969},
        {"as_of": "2024-08-01", "net": -0.12, "index": -0.041, "capital": 0.85},
    ]
    cand_months = [
        {"as_of": "2024-07-01", "net": -0.005, "index": -0.025, "capital": 0.995},
        {"as_of": "2024-08-01", "net": -0.06, "index": -0.041, "capital": 0.935},
    ]
    report = fitness_report(
        "probe", _arm(base_months, -0.147), _arm(cand_months, -0.0647)
    )
    assert report["gates"]["passed"]
    assert report["score"] > 0

    registry.attach_report(registered.short_id, "fitness_probe", report)
    registry.update_status(registered.short_id, "probed")

    markdown = render_report_markdown({**report, "run_id": "mock", "months": []})
    assert "Fitness report" in markdown and "Hard gates: PASS" in markdown

    registry.promote(registered.short_id, run_id="mock")
    assert registry.active_digest() == registered.digest


def test_verify_injection_checks_prompt(tmp_path, skill_repo, baseline_text):
    """_verify_injection detects both injected and missing skill bodies."""
    skill_file = skill_repo / SKILL_REL
    body_head = baseline_text.split("---", 2)[-1].strip()[:60]

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    good = {
        "main_agent_message_history": {
            "message_history": [
                {
                    "role": "user",
                    "content": f"任务…### Top Skill Preview\n{body_head} …",
                }
            ]
        }
    }
    (out_dir / "task_a_attempt_1.json").write_text(
        json.dumps(good, ensure_ascii=False), encoding="utf-8"
    )
    assert EvolutionController._verify_injection(out_dir, skill_file)

    (out_dir / "task_b_attempt_1.json").write_text(
        json.dumps({"main_agent_message_history": {"message_history": [
            {"role": "user", "content": "no skill here"}
        ]}}, ensure_ascii=False),
        encoding="utf-8",
    )
    assert not EvolutionController._verify_injection(out_dir, skill_file)
