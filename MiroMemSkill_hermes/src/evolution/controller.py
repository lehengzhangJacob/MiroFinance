# SPDX-FileCopyrightText: 2026 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

"""Evolution controller: isolated arms, manifests, paired evaluation.

An "arm" is one subprocess run of ``main.py common-benchmark`` with:

- its own copy-on-write snapshot DB (tool writes stay private),
- its own memory scratch dir and run-scoped namespace (defense in depth),
- a materialized single-skill directory (the ONLY difference between a
  baseline arm and a candidate arm),
- a month-subset task file generated from the frozen snapshot tasks.

Replay/fitness always reads the pristine snapshot DB, never an arm copy.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from src.evolution import fitness as fitness_mod
from src.evolution.splits import (
    MonthSplits,
    filter_tasks,
    load_tasks,
    make_splits,
    write_task_subset,
)
from src.evolution.types import sha256_text

DEFAULT_SNAPSHOT = Path(
    "/home/msj_team/Jacob/agent/shared/ashare_open_stocks_glm52_20260714"
)
DEFAULT_CONFIG = "agent_ashare_trader_open_hermes_glm"


def _utcnow() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _load_key_file(path: Path) -> dict[str, str]:
    """Parse a `set -a`-style KEY=VALUE file (llm_key) without executing it."""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip().strip("'\"")
    return out


class EvolutionController:
    def __init__(
        self,
        repo_root: str | Path | None = None,
        snapshot_dir: str | Path = DEFAULT_SNAPSHOT,
        config_name: str = DEFAULT_CONFIG,
        python_exe: str | None = None,
        train_months: int = 6,
        dev_months: int = 3,
        holdout_months: int = 3,
        skip_months: int = 0,
    ):
        self.repo_root = (
            Path(repo_root).resolve()
            if repo_root
            else Path(__file__).resolve().parents[2]
        )
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_db = self.snapshot_dir / "ashare_pools_snapshot.db"
        self.snapshot_tasks = (
            self.snapshot_dir
            / "tasks"
            / "ashare_trader_open"
            / "standardized_data.jsonl"
        )
        self.config_name = config_name
        self.python_exe = python_exe or sys.executable
        self.runs_root = self.repo_root / ".evolution" / "runs"
        self._split_counts = (train_months, dev_months, holdout_months)
        self._skip_months = skip_months

    # -------------------------------------------------------------- tasks

    def tasks_and_splits(self) -> tuple[list[dict], MonthSplits]:
        tasks = load_tasks(self.snapshot_tasks)
        train, dev, holdout = self._split_counts
        return tasks, make_splits(
            tasks,
            train_months=train,
            dev_months=dev,
            holdout_months=holdout,
            skip_months=self._skip_months,
        )

    # --------------------------------------------------------------- arms

    def run_dir(self, run_id: str) -> Path:
        return self.runs_root / run_id

    def run_arm(
        self,
        run_id: str,
        arm_name: str,
        skill_dir: str | Path,
        level: str = "probe",
        months: tuple[str, ...] | None = None,
        cleanup_db: bool = False,
        verify_db_sha: bool = False,
    ) -> Path:
        """Run one isolated benchmark arm; returns its output dir."""
        tasks, splits = self.tasks_and_splits()
        months = tuple(months) if months else splits.level_months(level)

        run_dir = self.run_dir(run_id)
        arm_dir = run_dir / "arms" / arm_name
        out_dir = arm_dir / "out"
        if out_dir.exists():
            raise RuntimeError(f"arm output already exists: {out_dir}")
        arm_dir.mkdir(parents=True, exist_ok=True)

        data_dir = arm_dir / "data"
        write_task_subset(tasks, months, data_dir)

        db_copy = arm_dir / "ashare_pools_arm.db"
        subprocess.run(
            [
                "cp",
                "--reflink=auto",
                "--preserve=mode,timestamps",
                str(self.snapshot_db),
                str(db_copy),
            ],
            check=True,
        )

        store_dir = arm_dir / "memory_bank"
        store_dir.mkdir(parents=True, exist_ok=True)

        skill_dir = Path(skill_dir).resolve()
        skill_files = sorted(skill_dir.glob("*.md"))
        if len(skill_files) != 1:
            raise RuntimeError(
                f"arm skill dir must contain exactly one .md skill, "
                f"found {len(skill_files)} in {skill_dir}"
            )
        skill_sha = sha256_text(skill_files[0].read_text(encoding="utf-8"))

        env = dict(os.environ)
        for key, value in _load_key_file(self.repo_root.parent / "llm_key").items():
            env.setdefault(key, value)
        env.update(
            {
                "ASHARE_OPEN_DB": str(db_copy),
                "DATA_DIR": str(data_dir),
                "ASHARE_TRADER_RUN_ID": f"hermes_{run_id}_{arm_name}",
                "HERMES_SKILLS_DIR": str(skill_dir),
                "HERMES_STORE_DIR": str(store_dir),
                "TUSHARE_TOKEN_FILE": env.get(
                    "TUSHARE_TOKEN_FILE", str(self.repo_root.parent / "tushare_token")
                ),
                "CHINESE_CONTEXT": env.get("CHINESE_CONTEXT", "true"),
                "MEM0_TELEMETRY": "false",
                # Keyword-only skill matching: deterministic across arms and
                # avoids an embedding API call that could route differently.
                "MEMSKILL_EMBEDDING_ENABLED": "false",
            }
        )

        manifest = {
            "run_id": run_id,
            "arm": arm_name,
            "level": level,
            "months": list(months),
            "config_name": self.config_name,
            "skill_dir": str(skill_dir),
            "skill_file": skill_files[0].name,
            "skill_sha256": skill_sha,
            "snapshot_db": str(self.snapshot_db),
            "snapshot_db_sha256": (
                _sha256_file(self.snapshot_db) if verify_db_sha else "skipped"
            ),
            "started_at": _utcnow(),
        }
        (arm_dir / "arm_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        log_path = arm_dir / "benchmark.log"
        with open(log_path, "w", encoding="utf-8") as log_fh:
            proc = subprocess.run(
                [
                    self.python_exe,
                    "main.py",
                    "common-benchmark",
                    f"--config_file_name={self.config_name}",
                    f"output_dir={out_dir}",
                ],
                cwd=self.repo_root,
                env=env,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
            )

        manifest["finished_at"] = _utcnow()
        manifest["exit_code"] = proc.returncode
        manifest["skill_injection_verified"] = self._verify_injection(
            out_dir, skill_files[0]
        )
        (arm_dir / "arm_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        if cleanup_db:
            db_copy.unlink(missing_ok=True)

        if proc.returncode != 0:
            raise RuntimeError(
                f"arm {arm_name} failed (exit {proc.returncode}); see {log_path}"
            )
        attempts = list(out_dir.glob("task_*_attempt_1.json"))
        if len(attempts) != len(months):
            raise RuntimeError(
                f"arm {arm_name} produced {len(attempts)} attempts, "
                f"expected {len(months)}"
            )
        return out_dir

    @staticmethod
    def _verify_injection(out_dir: Path, skill_file: Path) -> bool:
        """Confirm the arm's prompts actually contained the candidate body."""
        try:
            body_head = (
                skill_file.read_text(encoding="utf-8").split("---", 2)[-1].strip()[:60]
            )
            for attempt in sorted(out_dir.glob("task_*_attempt_1.json")):
                data = json.loads(attempt.read_text(encoding="utf-8"))
                history = data.get("main_agent_message_history") or {}
                messages = (
                    history.get("message_history")
                    if isinstance(history, dict)
                    else history
                ) or []
                first_user = next(
                    (
                        m
                        for m in messages
                        if isinstance(m, dict) and m.get("role") == "user"
                    ),
                    None,
                )
                content = (first_user or {}).get("content", "")
                if isinstance(content, list):
                    text = "\n".join(
                        str(part.get("text", ""))
                        for part in content
                        if isinstance(part, dict)
                    )
                else:
                    text = str(content)
                if "Top Skill Preview" not in text or body_head[:30] not in text:
                    return False
            return True
        except Exception:
            return False

    # ---------------------------------------------------------- evaluation

    def evaluate_pair(
        self,
        run_id: str,
        level: str,
        baseline_out: str | Path,
        candidate_out: str | Path,
        months: tuple[str, ...] | None = None,
    ) -> dict:
        tasks, splits = self.tasks_and_splits()
        months = tuple(months) if months else splits.level_months(level)
        subset = filter_tasks(tasks, months)
        baseline_arm = fitness_mod.evaluate_arm(
            baseline_out, subset, self.snapshot_db
        )
        candidate_arm = fitness_mod.evaluate_arm(
            candidate_out, subset, self.snapshot_db
        )
        report = fitness_mod.fitness_report(level, baseline_arm, candidate_arm)
        report["run_id"] = run_id
        report["months"] = list(months)
        reports_dir = self.run_dir(run_id) / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        out_path = reports_dir / f"fitness_{level}.json"
        out_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (reports_dir / f"fitness_{level}.md").write_text(
            render_report_markdown(report), encoding="utf-8"
        )
        return report

    # -------------------------------------------------------------- misc

    def cleanup_arm_dbs(self, run_id: str) -> int:
        removed = 0
        for db in self.run_dir(run_id).glob("arms/*/ashare_pools_arm.db"):
            db.unlink()
            removed += 1
        return removed


def render_report_markdown(report: dict) -> str:
    base, cand, paired = report["baseline"], report["candidate"], report["paired"]
    gates = report["gates"]
    lines = [
        f"# Fitness report — level={report['level']} run={report.get('run_id', '')}",
        "",
        f"Months: {', '.join(report.get('months', []))}",
        "",
        "| arm | total | index | excess | maxDD | worst | win | fees |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, arm in (("baseline", base), ("candidate", cand)):
        lines.append(
            f"| {label} | {arm['total_return']*100:+.2f}% "
            f"| {arm['index_return']*100:+.2f}% "
            f"| {arm['excess_return']*100:+.2f}% "
            f"| {arm['max_drawdown']*100:.2f}% "
            f"| {arm['worst_month']*100:+.2f}% "
            f"| {arm['win_rate']*100:.0f}% "
            f"| {arm['fees']:.0f} |"
        )
    lines += [
        "",
        f"Paired diffs (pp): {paired['diffs_pp']}",
        f"mean={paired['mean_diff_pp']:+.2f}pp sd={paired['stdev_diff_pp']:.2f}pp "
        f"wins={paired['wins']} losses={paired['losses']} "
        f"sign_p={paired['sign_test_p']}",
        "",
        f"Hard gates: {'PASS' if gates['passed'] else 'FAIL'}",
    ]
    lines += [f"- {f}" for f in gates["failures"]]
    lines += ["", f"Ranking score: {report['score']:+.4f}", ""]
    return "\n".join(lines)


def cleanup_dir(path: str | Path) -> None:
    shutil.rmtree(path, ignore_errors=True)
