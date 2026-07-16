# SPDX-FileCopyrightText: 2026 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

"""Skill-evolution CLI: rollout-driven candidate lifecycle.

Typical flow (all paths relative to the fork root)::

    python scripts/ashare/run_skill_evolution.py init
    python scripts/ashare/run_skill_evolution.py smoke            # probe loop
    python scripts/ashare/run_skill_evolution.py full             # train->dev->holdout
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
DEFAULT_CONFIG_DEEPSEEK = "agent_ashare_trader_open_hermes_deepseek"


def _bootstrap_llm_keys(repo_root: Path) -> None:
    """Load llm_key defaults; user-local own_* key files take precedence."""
    for key, value in _load_key_file(repo_root.parent / "llm_key").items():
        os.environ.setdefault(key, value)
    own_glm = repo_root.parent / "own_glm"
    if own_glm.exists():
        token = own_glm.read_text(encoding="utf-8").strip()
        if token:
            os.environ["GLM_API_KEY"] = token
            os.environ["VISION_API_KEY"] = token
    own_deepseek = repo_root.parent / "own_deepseek"
    if own_deepseek.exists():
        token = own_deepseek.read_text(encoding="utf-8").strip()
        if token:
            os.environ["DEEPSEEK_API_KEY"] = token


def _now_tag() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


class EvolutionCLI:
    def __init__(
        self,
        snapshot: str = str(DEFAULT_SNAPSHOT),
        config: str = DEFAULT_CONFIG,
        skill: str = DEFAULT_SKILL,
        python_exe: str = sys.executable,
        train_months: int = 6,
        dev_months: int = 3,
        holdout_months: int = 3,
    ):
        self._registry = SkillRegistry(REPO_ROOT, skill)
        self._controller = EvolutionController(
            repo_root=REPO_ROOT,
            snapshot_dir=snapshot,
            config_name=config,
            python_exe=python_exe,
            train_months=train_months,
            dev_months=dev_months,
            holdout_months=holdout_months,
        )
        _bootstrap_llm_keys(REPO_ROOT)

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

    def _resolve_ref(self, ref: str) -> str:
        """Resolve 'baseline' / 'active' / digest / short_id to a digest."""
        if ref == "baseline":
            return self._registry.baseline_digest()
        if ref == "active":
            return self._registry.active_digest()
        return self._registry.resolve(ref)["digest"]

    def run_arm(
        self,
        run_id: str,
        arm: str,
        candidate: str = "baseline",
        level: str = "probe",
        cleanup_db: bool = False,
    ):
        digest = self._resolve_ref(candidate)
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
        parent: str = "baseline",
    ):
        """Generate candidates from a completed arm's settled outcomes.

        ``parent`` selects the skill the mutation starts from: ``baseline``,
        ``active`` (current promoted skill), or any registered digest/short_id.
        """
        from src.evolution.generators import ReflectiveMutationGenerator

        tasks, splits = self._controller.tasks_and_splits()
        months = splits.level_months(level)
        subset = filter_tasks(tasks, months)
        arm = evaluate_arm(feedback_arm, subset, self._controller.snapshot_db)
        allocations = evaluator().extract_run_allocations(Path(feedback_arm))
        feedback = build_feedback(arm, allocations, label="现行 Skill")
        feedback_path = save_feedback(feedback_arm, feedback)
        print(f"feedback -> {feedback_path}")

        base = self._registry.artifact(self._resolve_ref(parent))
        generator = ReflectiveMutationGenerator(temperature=temperature)
        bodies = generator.propose(base, feedback, n=n)
        if not bodies:
            raise RuntimeError("generator produced no candidate bodies")

        registered = []
        for body in bodies:
            text = f"---\n{base.frontmatter}\n---\n\n{body}\n"
            from src.evolution.types import SkillArtifact

            candidate = SkillArtifact.from_text(
                text, parent_digest=base.digest
            )
            gate = run_static_gates(base, candidate)
            if not gate.passed:
                print(f"L0 REJECT: {gate.failures}")
                continue
            artifact = self._registry.register_candidate(
                text,
                parent_digest=base.digest,
                generator=generator.name,
                rationale=(
                    f"reflective mutation from {feedback_arm} "
                    f"({level}, parent={base.short_id})"
                ),
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

    # --------------------------------------------------------------- search

    def search(
        self,
        run_id: str = "",
        n: int = 1,
        cleanup_db: bool = True,
        base: str = "active",
    ):
        """Dev-only search: train -> propose -> rank on dev. Never touches holdout.

        Use this for multi-round evolution where sealed holdout is opened once
        after the whole search budget is spent (see holdout / holdout_multiseed).
        """
        run_id = run_id or f"search_{_now_tag()}"
        base_digest = self._resolve_ref(base)
        print(f"=== search run {run_id} (n={n}, base={base}@{base_digest[:12]}) ===")

        baseline_train_out = self.run_arm(
            run_id,
            "baseline_train",
            candidate=base,
            level="train",
            cleanup_db=cleanup_db,
        )
        short_ids = self.propose(
            feedback_arm=baseline_train_out, level="train", n=n, parent=base
        )
        if not short_ids:
            raise RuntimeError("no candidate survived L0 gates after train propose")

        self.run_arm(
            run_id,
            "baseline_dev",
            candidate=base,
            level="dev",
            cleanup_db=cleanup_db,
        )

        dev_results: list[tuple[str, dict]] = []
        for sid in short_ids:
            arm = f"candidate_dev_{sid}"
            print(f"=== dev candidate {sid} ===")
            self.run_arm(
                run_id, arm, candidate=sid, level="dev", cleanup_db=cleanup_db
            )
            report = self.evaluate(
                run_id,
                sid,
                baseline_arm="baseline_dev",
                candidate_arm=arm,
                level="dev",
            )
            dev_results.append((sid, report))

        survivors = [(s, r) for s, r in dev_results if r["gates"]["passed"]]
        result: dict = {
            "run_id": run_id,
            "candidates": short_ids,
            "dev_results": {s: r["score"] for s, r in dev_results},
            "base": base_digest[:12],
        }
        if not survivors:
            result["status"] = "dev_all_failed"
            result["best"] = None
            result["dev_score"] = None
            print("=== search aborted: no candidate passed dev gates ===")
        else:
            best_sid, best_dev = max(survivors, key=lambda x: x[1]["score"])
            result["status"] = "dev_complete"
            result["best"] = best_sid
            result["dev_score"] = best_dev["score"]
            result["dev_mean_paired_pp"] = best_dev["paired"]["mean_diff_pp"]
            result["dev_gates"] = best_dev["gates"]["passed"]
            print(
                f"=== search best={best_sid} "
                f"(dev score={best_dev['score']:+.4f}, "
                f"mean_paired={best_dev['paired']['mean_diff_pp']:+.2f}pp) ==="
            )

        # Persist a machine-readable summary for chain scripts.
        summary_path = self._controller.run_dir(run_id) / "reports" / "search_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"=== search run {run_id} complete ===")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result

    # --------------------------------------------------------------- full

    def full(
        self,
        run_id: str = "",
        n: int = 1,
        cleanup_db: bool = True,
        promote_best: bool = False,
        base: str = "baseline",
    ):
        """Full fidelity loop: train(6) -> propose -> dev(3) rank -> holdout(3) best.

        ``base`` selects the comparison/parent skill for the whole round:
        ``baseline`` (registry baseline) or ``active`` (current promoted skill).
        Round-2 style continued evolution should pass ``--base=active``.

        Prefer ``search`` + a single final ``holdout_multiseed`` when running
        multi-round chains so the sealed set is not reopened each round.
        """
        search_result = self.search(
            run_id=run_id or f"full_{_now_tag()}",
            n=n,
            cleanup_db=cleanup_db,
            base=base,
        )
        run_id = search_result["run_id"]
        if search_result.get("status") != "dev_complete":
            return search_result

        best_sid = search_result["best"]
        print(
            f"=== holdout best={best_sid} "
            f"(dev score={search_result['dev_score']:+.4f}) ==="
        )
        holdout_report = self.holdout(
            best_sid, run_id=run_id, cleanup_db=cleanup_db, base=base
        )

        result = {
            **search_result,
            "status": "complete",
            "holdout_score": holdout_report["score"],
            "holdout_gates": holdout_report["gates"]["passed"],
        }
        if promote_best and holdout_report["gates"]["passed"]:
            result["promotion"] = self.promote(best_sid, run_id=run_id)
        print(f"=== full run {run_id} complete ===")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result

    # ------------------------------------------------------------ sealed

    def holdout(
        self,
        candidate: str,
        run_id: str = "",
        cleanup_db: bool = True,
        base: str = "baseline",
    ):
        """One-shot sealed evaluation; lease is acquired before anything runs."""
        run_id = run_id or f"holdout_{_now_tag()}"
        self._registry.acquire_holdout_lease(candidate, run_id)
        self.run_arm(
            run_id, "baseline", candidate=base, level="holdout", cleanup_db=cleanup_db
        )
        self.run_arm(
            run_id, "candidate", candidate=candidate, level="holdout", cleanup_db=cleanup_db
        )
        return self.evaluate(run_id, candidate, level="holdout")

    def holdout_multiseed(
        self,
        candidate: str,
        run_id: str = "",
        seeds: int = 3,
        cleanup_db: bool = True,
        base: str = "active",
    ):
        """Sealed holdout with independent rollout seeds; lease acquired once.

        Each seed re-runs baseline and candidate arms (temperature > 0 yields
        different trajectories). Aggregation uses the mean ranking score across
        seeds; hard gates must pass on every seed.
        """
        if seeds < 1:
            raise ValueError("seeds must be >= 1")
        run_id = run_id or f"holdout_ms_{_now_tag()}"
        self._registry.acquire_holdout_lease(candidate, run_id)

        seed_reports: list[dict] = []
        for seed in range(seeds):
            print(f"=== holdout seed {seed + 1}/{seeds} ===")
            base_arm = f"baseline_s{seed}"
            cand_arm = f"candidate_s{seed}"
            self.run_arm(
                run_id, base_arm, candidate=base, level="holdout", cleanup_db=cleanup_db
            )
            self.run_arm(
                run_id,
                cand_arm,
                candidate=candidate,
                level="holdout",
                cleanup_db=cleanup_db,
            )
            run_dir = self._controller.run_dir(run_id)
            report = self._controller.evaluate_pair(
                run_id,
                "holdout",
                run_dir / "arms" / base_arm / "out",
                run_dir / "arms" / cand_arm / "out",
            )
            report["seed"] = seed
            seed_reports.append(report)
            seed_path = run_dir / "reports" / f"fitness_holdout_s{seed}.json"
            seed_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(
                f"seed {seed}: score={report['score']:+.4f} "
                f"gates={report['gates']['passed']} "
                f"paired={report['paired']['mean_diff_pp']:+.2f}pp"
            )

        all_gates = all(r["gates"]["passed"] for r in seed_reports)
        mean_score = sum(r["score"] for r in seed_reports) / len(seed_reports)
        mean_paired = (
            sum(r["paired"]["mean_diff_pp"] for r in seed_reports) / len(seed_reports)
        )
        aggregate = {
            "level": "holdout",
            "run_id": run_id,
            "candidate": candidate,
            "base": self._resolve_ref(base)[:12],
            "seeds": seeds,
            "gates": {
                "passed": all_gates,
                "failures": []
                if all_gates
                else [
                    f"seed {r['seed']} failed: {r['gates']['failures']}"
                    for r in seed_reports
                    if not r["gates"]["passed"]
                ],
            },
            "score": round(mean_score, 6),
            "mean_paired_pp": round(mean_paired, 4),
            "seed_scores": [r["score"] for r in seed_reports],
            "seed_paired_pp": [r["paired"]["mean_diff_pp"] for r in seed_reports],
            "seed_reports": seed_reports,
        }
        self._registry.attach_report(
            candidate,
            "fitness_holdout_multiseed",
            {k: v for k, v in aggregate.items() if k != "seed_reports"},
        )
        status = "holdout_evaluated" if all_gates else "gates_failed"
        self._registry.update_status(candidate, status)

        reports_dir = self._controller.run_dir(run_id) / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        out_path = reports_dir / "fitness_holdout_multiseed.json"
        # Drop nested seed_reports' bulky month lists for the summary file? Keep full.
        out_path.write_text(
            json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({k: v for k, v in aggregate.items() if k != "seed_reports"},
                         ensure_ascii=False, indent=2))
        return aggregate

    def promote(self, candidate: str, run_id: str = ""):
        record = self._registry.promote(candidate, run_id=run_id)
        print(f"promoted {candidate}: {json.dumps(record, ensure_ascii=False)}")
        return record

    def rollback(self, target: str):
        self._registry.rollback(target)
        print(f"rolled back active skill to {target}")


if __name__ == "__main__":
    fire.Fire(EvolutionCLI)
