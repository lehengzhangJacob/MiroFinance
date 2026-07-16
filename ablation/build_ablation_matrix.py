#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 MiromindAI
#
# Cross-run ablation matrix on the frozen 24-month snapshot.
#
# Champion = R1 skill 3aebb813bd33 + memory (leave-one-out framing).
# Pulls finished arms from:
#   runs/plain_ablation_24m    (own_glm6): plain (no skill, no memory)
#   runs/memonly_ablation_24m  (own_glm4): mem_only
#   runs/skillonly_r1_24m      (own_glm5): R1 skill, no memory
#   runs/mem_ablation_24m      (own_glm3): R1 skill + memory (= Full)
# and replays them with the same deterministic evaluator, per segment.
# Arms that are missing or incomplete are skipped; re-run any time:
#
#   /home/msj_team/.conda/envs/Miro/bin/python ablation/build_ablation_matrix.py
"""Build leave-one-out matrix around final R1 (3aebb813bd33)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
HERMES_ROOT = AGENT_ROOT / "MiroMemSkill_hermes"
sys.path.insert(0, str(HERMES_ROOT))

ABLATION_ROOT = Path(__file__).resolve().parent
SNAPSHOT = AGENT_ROOT / "shared" / "ashare_open_stocks_glm52_24m_20260715"
RUNS_ROOT = ABLATION_ROOT / "runs"
REPORTS_DIR = ABLATION_ROOT / "reports"

EXPECTED_MONTHS = 24
# Pairing baseline for leave-one-out deltas: prefer Full R1, else plain.
PAIR_BASELINE = "full_r1"
PAIR_FALLBACK = "plain"

# label -> (arm dir, description)
# Primary 2x2 around final R1; optional baseline+mem kept if present.
MATRIX_ARMS: dict[str, tuple[Path, str]] = {
    "plain": (
        RUNS_ROOT / "plain_ablation_24m" / "arms" / "plain",
        "no memory, no skill",
    ),
    "skill_only_pre": (
        RUNS_ROOT / "_stitch_skillonly_baseline_24m",
        "pre-evolution skill 0a931278001c, memory OFF "
        "(stitched from formal24m_20260715 baseline rollouts)",
    ),
    "skill_only_r1": (
        RUNS_ROOT / "skillonly_r1_24m" / "arms" / "r1_best",
        "FINAL R1 skill 3aebb813bd33, memory OFF",
    ),
    "mem_only": (
        RUNS_ROOT / "memonly_ablation_24m" / "arms" / "mem_only",
        "trader-episode memory, no skill (w/o skill)",
    ),
    "wo_self_evolve": (
        RUNS_ROOT / "mem_ablation_24m" / "arms" / "baseline",
        "memory + pre-evolution skill 0a931278001c (w/o self-evolve)",
    ),
    "full_r1": (
        RUNS_ROOT / "mem_ablation_24m" / "arms" / "r1_best",
        "FINAL = memory + R1 skill 3aebb813bd33",
    ),
}

# Offline-stitched replay dirs: no arm_manifest.json, month count is the gate.
STITCHED_ARMS = {"skill_only_pre"}

ARM_METRIC_KEYS = (
    "total_return",
    "index_return",
    "excess_return",
    "max_drawdown",
    "annualized_sharpe",
    "worst_month",
    "win_rate",
    "fees",
)

SEGMENTS = {
    "full_24m": ("2024-07", "2026-06"),
    "formal_12m": ("2024-07", "2025-06"),
    "dev_6m": ("2025-07", "2025-12"),
    "holdout_6m": ("2026-01", "2026-06"),
}


def _arm_ready(arm_dir: Path, stitched: bool = False) -> tuple[bool, str]:
    out_dir = arm_dir / "out"
    if not out_dir.is_dir():
        return False, "no output dir (not started)"
    attempts = len(list(out_dir.glob("task_*_attempt_1.json")))
    if attempts != EXPECTED_MONTHS:
        return False, f"incomplete: {attempts}/{EXPECTED_MONTHS} months"
    if stitched:
        return True, "ok (stitched replay)"
    manifest_path = arm_dir / "arm_manifest.json"
    if not manifest_path.exists():
        return False, "no manifest"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("snapshot_db") != str(SNAPSHOT / "ashare_pools_snapshot.db"):
        return False, f"foreign snapshot: {manifest.get('snapshot_db')}"
    if manifest.get("exit_code") != 0:
        return False, f"exit_code={manifest.get('exit_code')}"
    return True, "ok"


def main() -> None:
    from src.evolution.fitness import evaluate_arm, fitness_report
    from src.evolution.splits import filter_tasks, load_tasks

    tasks = load_tasks(
        SNAPSHOT / "tasks" / "ashare_trader_open" / "standardized_data.jsonl"
    )
    months = tuple(str(t["metadata"]["as_of"]) for t in tasks)
    snapshot_db = SNAPSHOT / "ashare_pools_snapshot.db"

    ready: dict[str, Path] = {}
    status: dict[str, str] = {}
    for label, (arm_dir, _desc) in MATRIX_ARMS.items():
        ok, reason = _arm_ready(arm_dir, stitched=label in STITCHED_ARMS)
        status[label] = reason
        print(f"arm {label}: {reason}")
        if ok:
            ready[label] = arm_dir / "out"
    if not ready:
        raise SystemExit("no completed arms; nothing to report")

    pair_ref = (
        PAIR_BASELINE if PAIR_BASELINE in ready
        else PAIR_FALLBACK if PAIR_FALLBACK in ready
        else None
    )
    matrix: dict[str, dict] = {}
    lines = [
        "# Ablation matrix — final R1 = 3aebb813bd33 "
        f"({SNAPSHOT.name})",
        "",
        "Leave-one-out around FINAL = memory + R1 skill.",
        f"Paired column is vs `{pair_ref or 'n/a'}` "
        "(prefer full_r1; fall back to plain while Full is unfinished).",
        "",
        "| arm | description | status |",
        "|---|---|---|",
    ]
    for label, (_dir, desc) in MATRIX_ARMS.items():
        lines.append(f"| {label} | {desc} | {status[label]} |")

    for seg_name, (first, last) in SEGMENTS.items():
        seg_months = tuple(m for m in months if first <= m[:7] <= last)
        if not seg_months:
            continue
        subset = filter_tasks(tasks, seg_months)
        rows: dict[str, dict] = {}
        for label, out_dir in ready.items():
            rows[label] = evaluate_arm(out_dir, subset, snapshot_db)

        paired: dict[str, dict] = {}
        if pair_ref and pair_ref in rows:
            for label, arm in rows.items():
                if label == pair_ref:
                    continue
                # fitness_report(baseline, candidate) -> candidate - baseline.
                # Put pair_ref first so deltas show "ablated arm minus Full".
                report = fitness_report(seg_name, rows[pair_ref], arm)
                paired[label] = report["paired"]

        matrix[seg_name] = {
            "months": [seg_months[0], seg_months[-1]],
            "pair_ref": pair_ref,
            "arms": {
                label: {key: arm[key] for key in ARM_METRIC_KEYS}
                for label, arm in rows.items()
            },
            f"paired_vs_{pair_ref or 'none'}": paired,
        }

        lines += [
            "",
            f"## {seg_name} ({seg_months[0]} .. {seg_months[-1]})",
            "",
            "| arm | total | index | excess | maxDD | sharpe | worst | win | "
            f"vs {pair_ref or 'n/a'} (mean pp) | W-L | sign p |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for label in MATRIX_ARMS:
            if label not in rows:
                continue
            arm = rows[label]
            sharpe = arm.get("annualized_sharpe")
            sharpe_cell = f"{sharpe:.2f}" if sharpe is not None else "—"
            pair = paired.get(label)
            if label == pair_ref:
                pair_cells = "— | — | —"
            elif pair is None:
                pair_cells = "n/a | n/a | n/a"
            else:
                pair_cells = (
                    f"{pair['mean_diff_pp']:+.2f} | "
                    f"{pair['wins']}-{pair['losses']} | "
                    f"{pair['sign_test_p']}"
                )
            lines.append(
                f"| {label} | {arm['total_return']*100:+.2f}% "
                f"| {arm['index_return']*100:+.2f}% "
                f"| {arm['excess_return']*100:+.2f}% "
                f"| {arm['max_drawdown']*100:.2f}% "
                f"| {sharpe_cell} "
                f"| {arm['worst_month']*100:+.2f}% "
                f"| {arm['win_rate']*100:.0f}% "
                f"| {pair_cells} |"
            )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "matrix_24m.json").write_text(
        json.dumps(
            {"snapshot": SNAPSHOT.name, "arm_status": status, "segments": matrix},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (REPORTS_DIR / "matrix_24m.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"matrix -> {REPORTS_DIR / 'matrix_24m.md'}")


if __name__ == "__main__":
    main()
