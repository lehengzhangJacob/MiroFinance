#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 MiromindAI
#
# R1-best vs pre-evolution baseline, WITH memory, WITHOUT Hermes evolution.
#
# Both arms run the full MemSkill trader runtime (trader episodes recorded at
# each month barrier, visible only after their exit date matures) over the
# same frozen 24-month snapshot. The ONLY difference between the arms is the
# injected skill text:
#
#   baseline  = pre-R1 open skill  0a931278001c
#   r1_best   = R1 promoted skill  3aebb813bd33
#
# Uses agent/own_glm3 exclusively so it does not share quota with the ongoing
# Hermes evolution chain (own_glm).
"""Memory-ON skill ablation: baseline vs R1-best over 24 months."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
HERMES_ROOT = AGENT_ROOT / "MiroMemSkill_hermes"
sys.path.insert(0, str(HERMES_ROOT))

ABLATION_ROOT = Path(__file__).resolve().parent
KEY_FILE = AGENT_ROOT / "own_glm3"
SNAPSHOT = AGENT_ROOT / "shared" / "ashare_open_stocks_glm52_24m_20260715"
BASELINE_SKILL = ABLATION_ROOT / "baseline_0a931278001c"
# Clean dir with exactly one .md (controller rejects fitness/README siblings).
R1_SKILL = ABLATION_ROOT / "skill_r1_3aebb813bd33"
RUNS_ROOT = ABLATION_ROOT / "runs"

CONFIG_NAME = "agent_ashare_trader_open_hermes_memfull_glm"

# Reporting segments (month prefix YYYY-MM). full_24m is added automatically.
SEGMENTS = {
    "formal_12m": ("2024-07", "2025-06"),   # original MemSkill-Full window
    "dev_6m": ("2025-07", "2025-12"),       # R1 selection window (skill-only)
    "holdout_6m": ("2026-01", "2026-06"),   # R1 sealed holdout window
}


def _load_own_glm3() -> None:
    """Force own_glm3 (first line of the file is the key; rest is notes)."""
    if not KEY_FILE.exists():
        raise SystemExit(f"missing API key file: {KEY_FILE}")
    lines = KEY_FILE.read_text(encoding="utf-8").splitlines()
    token = lines[0].strip() if lines else ""
    if not re.fullmatch(r"[A-Za-z0-9._-]{20,}", token):
        raise SystemExit(f"first line of {KEY_FILE} does not look like a key")
    os.environ["GLM_API_KEY"] = token
    os.environ["VISION_API_KEY"] = token
    for var in (
        "http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
        "all_proxy", "ALL_PROXY",
    ):
        os.environ.pop(var, None)
    print(f"GLM_API_KEY loaded from {KEY_FILE.name} (len={len(token)})")


def _month_range(months: tuple[str, ...], first: str, last: str) -> tuple[str, ...]:
    return tuple(m for m in months if first <= m[:7] <= last)


def _count_episode_logs(arm_dir: Path) -> int:
    """Sanity check that memory was ON: count month-barrier episode writes."""
    log = arm_dir / "benchmark.log"
    if not log.exists():
        return 0
    return sum(
        1
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines()
        if re.search(r"trader\[\d{4}-\d{2}\] (ADD|UPDATE) trader episode", line)
    )


def main() -> None:
    import fire

    def run(run_id: str = "", cleanup_db: bool = True) -> dict:
        _load_own_glm3()
        from src.evolution.controller import (
            EvolutionController,
            render_report_markdown,
        )
        from src.evolution.fitness import evaluate_arm, fitness_report
        from src.evolution.splits import filter_tasks, load_tasks

        run_id = run_id or "mem_ablation_24m"
        out_root = RUNS_ROOT / run_id

        ctrl = EvolutionController(
            repo_root=HERMES_ROOT,
            snapshot_dir=SNAPSHOT,
            config_name=CONFIG_NAME,
            train_months=24,
            dev_months=0,
            holdout_months=0,
        )
        ctrl.runs_root = RUNS_ROOT  # keep artifacts inside ablation/runs

        tasks = load_tasks(SNAPSHOT / "tasks" / "ashare_trader_open" / "standardized_data.jsonl")
        months = tuple(str(t["metadata"]["as_of"]) for t in tasks)
        if len(months) != 24:
            raise SystemExit(f"expected 24 months in 24m snapshot, got {len(months)}")
        print(f"=== memory-ON ablation {run_id} months={months[0]}..{months[-1]} ===")
        print(f"=== key=own_glm3 config={CONFIG_NAME} snapshot={SNAPSHOT.name} ===")

        arms = {
            "baseline": BASELINE_SKILL,
            "r1_best": R1_SKILL,
        }
        episode_counts: dict[str, int] = {}
        for name, skill_dir in arms.items():
            if not (skill_dir / "ashare_open_portfolio.md").exists():
                raise SystemExit(f"missing skill: {skill_dir}")
            print(f"=== arm {name} skill={skill_dir.name} ===", flush=True)
            ctrl.run_arm(
                run_id,
                name,
                skill_dir,
                level="train",
                months=months,
                cleanup_db=cleanup_db,
            )
            episode_counts[name] = _count_episode_logs(
                out_root / "arms" / name
            )
            print(f"=== arm {name} done, trader episodes logged: "
                  f"{episode_counts[name]} ===", flush=True)

        reports_dir = out_root / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        segments = {"full_24m": (months[0][:7], months[-1][:7]), **SEGMENTS}
        summary: dict[str, dict] = {}
        for seg_name, (first, last) in segments.items():
            seg_months = _month_range(months, first, last)
            if not seg_months:
                print(f"segment {seg_name}: no months, skipped")
                continue
            subset = filter_tasks(tasks, seg_months)
            baseline_arm = evaluate_arm(
                out_root / "arms" / "baseline" / "out", subset, ctrl.snapshot_db
            )
            r1_arm = evaluate_arm(
                out_root / "arms" / "r1_best" / "out", subset, ctrl.snapshot_db
            )
            report = fitness_report(seg_name, baseline_arm, r1_arm)
            report["run_id"] = run_id
            report["months"] = list(seg_months)
            report["api_key_file"] = KEY_FILE.name
            report["config_name"] = CONFIG_NAME
            report["baseline_skill"] = "0a931278001c"
            report["candidate_skill"] = "3aebb813bd33"
            report["trader_episodes_logged"] = episode_counts
            report["note"] = (
                "memory-ON ablation: full MemSkill trader runtime (episodic "
                "memory with exit-date embargo) in BOTH arms; only the "
                "injected skill text differs; no Hermes evolution involved"
            )
            (reports_dir / f"fitness_{seg_name}.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            (reports_dir / f"fitness_{seg_name}.md").write_text(
                render_report_markdown(report), encoding="utf-8"
            )
            summary[seg_name] = {
                "months": f"{seg_months[0]}..{seg_months[-1]}",
                "baseline_total": report["baseline"]["total_return"],
                "r1_total": report["candidate"]["total_return"],
                "baseline_sharpe": report["baseline"].get("annualized_sharpe"),
                "r1_sharpe": report["candidate"].get("annualized_sharpe"),
                "mean_paired_diff_pp": report["paired"]["mean_diff_pp"],
                "wins": report["paired"]["wins"],
                "losses": report["paired"]["losses"],
            }
            print(json.dumps({seg_name: summary[seg_name]},
                             ensure_ascii=False, indent=2))

        (reports_dir / "summary.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "config_name": CONFIG_NAME,
                    "snapshot": SNAPSHOT.name,
                    "api_key_file": KEY_FILE.name,
                    "trader_episodes_logged": episode_counts,
                    "segments": summary,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"=== done -> {reports_dir} ===")
        return summary

    fire.Fire(run)


if __name__ == "__main__":
    main()
