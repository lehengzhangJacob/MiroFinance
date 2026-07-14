# SPDX-FileCopyrightText: 2026 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

"""Skill-evolution CLI: rollout-driven candidate lifecycle.

Typical flow (all paths relative to the fork root)::

    python scripts/ashare/run_skill_evolution.py init
    python scripts/ashare/run_skill_evolution.py smoke            # probe loop
    python scripts/ashare/run_skill_evolution.py status
    python scripts/ashare/run_skill_evolution.py holdout --candidate=<short_id>
    python scripts/ashare/run_skill_evolution.py promote --candidate=<short_id>

Levels: probe (first 2 train months) -> train (6) -> dev (3) -> holdout (3,
one-shot lease per candidate, enforced by the registry).
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import fire  # noqa: E402

from src.evolution.controller import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_SNAPSHOT,
    EvolutionController,
    _load_key_file,
)
from src.evolution.feedback import build_feedback, save_feedback  # noqa: E402
from src.evolution.fitness import evaluate_arm, evaluator  # noqa: E402
from src.evolution.gates import run_static_gates  # noqa: E402
from src.evolution.registry import SkillRegistry  # noqa: E402
from src.evolution.splits import filter_tasks  # noqa: E402

DEFAULT_SKILL = "memory_bank/skills_ashare/ashare_open_portfolio.md"


def _now_tag() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


class EvolutionCLI:
    def __init__(
        self,
        snapshot: str = str(DEFAULT_SNAPSHOT),
        config: str = DEFAULT_CONFIG,
        skill: str = DEFAULT_SKILL,
        python_exe: str = sys.executable,
    ):
        self._registry = SkillRegistry(REPO_ROOT, skill)
        self._controller = EvolutionController(
            repo_root=REPO_ROOT,
            snapshot_dir=snapshot,
            config_name=config,
            python_exe=python_exe,
        )
        for key, value in _load_key_file(REPO_ROOT.parent / "llm_key").items():
            os.environ.setdefault(key, value)

    # ----------------------------------------------------------- lifecycle

    def init(self, force: bool = False):
        artifact = self._registry.init_baseline(force=force)
        print(f"baseline registered: {artifact.name} @ {artifact.short_id}")
        return artifact.short_id

    def status(self):
        data = self._registry.status()
        print(f"skill: {data['skill_name']} ({data['skill_rel_path']})")
        print(f"active: {data['active_digest'][:12]}  baseline: {data['baseline_digest'][:12]}")
        print(f"candidates: {len(data['candidates'])}")
        for rec in data["candidates"].values():
            reports = ",".join(rec["reports"].keys()) or "-"
            print(
                f"  {rec['short_id']}  status={rec['status']:<18} "
                f"gen={rec['generator']:<24} reports={reports}"
            )
        if data["holdout_leases"]:
            print("holdout leases:")
            for digest, lease in data["holdout_leases"].items():
                print(f"  {digest[:12]} -> {lease['run_id']} at {lease['acquired_at']}")

    # ------------------------------------------------------------ pipeline

    def run_arm(
        self,
        run_id: str,
        arm: str,
        candidate: str = "baseline",
        level: str = "probe",
        cleanup_db: bool = False,
    ):
        digest = (
            self._registry.baseline_digest()
            if candidate == "baseline"
            else self._registry.resolve(candidate)["digest"]
        )
        skill_dir = self._registry.skill_dir(digest)
        out = self._controller.run_arm(
            run_id, arm, skill_dir, level=level, cleanup_db=cleanup_db
        )
        print(f"arm done -> {out}")
        return str(out)

    def propose(
        self,
        feedback_arm: str,
        level: str = "probe",
        n: int = 1,
        temperature: float = 0.9,
    ):
        """Generate candidates from a completed arm's settled outcomes."""
        from src.evolution.generators import ReflectiveMutationGenerator

        tasks, splits = self._controller.tasks_and_splits()
        months = splits.level_months(level)
        subset = filter_tasks(tasks, months)
        arm = evaluate_arm(feedback_arm, subset, self._controller.snapshot_db)
        allocations = evaluator().extract_run_allocations(Path(feedback_arm))
        feedback = build_feedback(arm, allocations, label="现行 Skill")
        feedback_path = save_feedback(feedback_arm, feedback)
        print(f"feedback -> {feedback_path}")

        baseline = self._registry.artifact(self._registry.baseline_digest())
        generator = ReflectiveMutationGenerator(temperature=temperature)
        bodies = generator.propose(baseline, feedback, n=n)
        if not bodies:
            raise RuntimeError("generator produced no candidate bodies")

        registered = []
        for body in bodies:
            text = f"---\n{baseline.frontmatter}\n---\n\n{body}\n"
            from src.evolution.types import SkillArtifact

            candidate = SkillArtifact.from_text(
                text, parent_digest=baseline.digest
            )
            gate = run_static_gates(baseline, candidate)
            if not gate.passed:
                print(f"L0 REJECT: {gate.failures}")
                continue
            artifact = self._registry.register_candidate(
                text,
                parent_digest=baseline.digest,
                generator=generator.name,
                rationale=f"reflective mutation from {feedback_arm} ({level})",
            )
            registered.append(artifact.short_id)
            print(f"L0 PASS -> registered candidate {artifact.short_id}")
        return registered

    def evaluate(
        self,
        run_id: str,
        candidate: str,
        baseline_arm: str = "baseline",
        candidate_arm: str = "candidate",
        level: str = "probe",
    ):
        run_dir = self._controller.run_dir(run_id)
        report = self._controller.evaluate_pair(
            run_id,
            level,
            run_dir / "arms" / baseline_arm / "out",
            run_dir / "arms" / candidate_arm / "out",
        )
        self._registry.attach_report(candidate, f"fitness_{level}", report)
        status = "probed" if level == "probe" else f"{level}_evaluated"
        if not report["gates"]["passed"]:
            status = "gates_failed"
        self._registry.update_status(candidate, status)
        print(json.dumps(report["paired"], ensure_ascii=False, indent=2))
        print(f"gates: {report['gates']}")
        print(f"score: {report['score']}")
        return report

    # -------------------------------------------------------------- smoke

    def smoke(
        self,
        run_id: str = "",
        n: int = 1,
        level: str = "probe",
        cleanup_db: bool = True,
    ):
        """Full loop at probe fidelity: baseline arm -> propose -> candidate arm."""
        run_id = run_id or f"smoke_{_now_tag()}"
        print(f"=== smoke run {run_id} (level={level}) ===")

        baseline_out = self.run_arm(
            run_id, "baseline", candidate="baseline", level=level, cleanup_db=cleanup_db
        )
        short_ids = self.propose(feedback_arm=baseline_out, level=level, n=n)
        if not short_ids:
            raise RuntimeError("no candidate survived L0 gates")
        chosen = short_ids[0]
        print(f"=== evaluating candidate {chosen} ===")
        self.run_arm(
            run_id, "candidate", candidate=chosen, level=level, cleanup_db=cleanup_db
        )
        report = self.evaluate(run_id, chosen, level=level)
        print(f"=== smoke run {run_id} complete ===")
        return {"run_id": run_id, "candidate": chosen, "score": report["score"]}

    # ------------------------------------------------------------ sealed

    def holdout(self, candidate: str, run_id: str = "", cleanup_db: bool = True):
        """One-shot sealed evaluation; lease is acquired before anything runs."""
        run_id = run_id or f"holdout_{_now_tag()}"
        self._registry.acquire_holdout_lease(candidate, run_id)
        self.run_arm(
            run_id, "baseline", candidate="baseline", level="holdout", cleanup_db=cleanup_db
        )
        self.run_arm(
            run_id, "candidate", candidate=candidate, level="holdout", cleanup_db=cleanup_db
        )
        return self.evaluate(run_id, candidate, level="holdout")

    def promote(self, candidate: str, run_id: str = ""):
        record = self._registry.promote(candidate, run_id=run_id)
        print(f"promoted {candidate}: {json.dumps(record, ensure_ascii=False)}")
        return record

    def rollback(self, target: str):
        self._registry.rollback(target)
        print(f"rolled back active skill to {target}")


if __name__ == "__main__":
    fire.Fire(EvolutionCLI)
