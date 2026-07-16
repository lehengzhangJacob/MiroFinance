#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 MiromindAI
#
# Skill-only ablation of the R1 final skill (3aebb813bd33):
# keep the promoted skill, turn memory completely OFF.
#
# Single arm over the frozen 24-month snapshot under the Hermes skill-only
# config (memory constructed but never written or injected; skill injected
# FULL BODY). This is leave-one-out: Full = R1+memory, this arm = R1 only.
#
# Uses agent/own_glm5 exclusively.
"""Skill-only ablation: R1 skill 3aebb813bd33, memory OFF, 24 months."""

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
KEY_FILE = AGENT_ROOT / "own_glm5"
SNAPSHOT = AGENT_ROOT / "shared" / "ashare_open_stocks_glm52_24m_20260715"
# Clean dir with exactly one .md (controller rejects fitness/README siblings).
R1_SKILL = ABLATION_ROOT / "skill_r1_3aebb813bd33"
RUNS_ROOT = ABLATION_ROOT / "runs"

CONFIG_NAME = "agent_ashare_trader_open_hermes_glm"
ARM_NAME = "r1_best"
SKILL_SHORT = "3aebb813bd33"

SEGMENTS = {
    "formal_12m": ("2024-07", "2025-06"),
    "dev_6m": ("2025-07", "2025-12"),
    "holdout_6m": ("2026-01", "2026-06"),
}

EPISODE_RE = re.compile(
    r"trader\[\d{4}-\d{2}\] (ADD|UPDATE) trader episode"
)


def _load_own_glm5() -> None:
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


def _preflight_config() -> None:
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(
        config_dir=str(HERMES_ROOT / "config"), version_base=None
    ):
        cfg = compose(config_name=CONFIG_NAME)

    memory = cfg.get("memory")
    if not memory:
        raise SystemExit("skill-only config lost its memory section")
    checks = {
        "enabled": (bool(memory.get("enabled")), True),
        "inject_enabled": (bool(memory.get("inject_enabled")), False),
        "reflection_enabled": (bool(memory.get("reflection_enabled")), False),
        "outcome_logging_enabled": (
            bool(memory.get("outcome_logging_enabled")), False
        ),
        "skill_enabled": (bool(memory.get("skill_enabled")), True),
        "skill_preview_max_chars": (
            int(memory.get("skill_preview_max_chars", 1500)), 0
        ),
    }
    for field, (actual, expected) in checks.items():
        if actual != expected:
            raise SystemExit(
                f"skill-only config: memory.{field}={actual!r}, "
                f"expected {expected!r}"
            )
    print("preflight OK: skill-only = R1 full-body skill, zero memory R/W")


def _month_range(months: tuple[str, ...], first: str, last: str) -> tuple[str, ...]:
    return tuple(m for m in months if first <= m[:7] <= last)


def _count_episode_logs(arm_dir: Path) -> int:
    log = arm_dir / "benchmark.log"
    if not log.exists():
        return 0
    return sum(
        1
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines()
        if EPISODE_RE.search(line)
    )


def _first_user_text(attempt_file: Path) -> str:
    data = json.loads(attempt_file.read_text(encoding="utf-8"))
    history = data.get("main_agent_message_history") or {}
    messages = (
        history.get("message_history") if isinstance(history, dict) else history
    ) or []
    first_user = next(
        (m for m in messages if isinstance(m, dict) and m.get("role") == "user"),
        None,
    )
    content = (first_user or {}).get("content", "")
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text", "")) for part in content if isinstance(part, dict)
        )
    return str(content)


def _verify_arm(arm_dir: Path) -> dict:
    out_dir = arm_dir / "out"
    attempts = sorted(out_dir.glob("task_*_attempt_1.json"))
    episodes = _count_episode_logs(arm_dir)
    skill_hits = 0
    memory_hits = 0
    for attempt in attempts:
        text = _first_user_text(attempt)
        if "Top Skill Preview" in text:
            skill_hits += 1
        if "（严格 walk-forward）" in text:
            memory_hits += 1
    result = {
        "attempts": len(attempts),
        "episodes_logged": episodes,
        "prompts_with_skill": skill_hits,
        "prompts_with_memory": memory_hits,
    }
    if skill_hits != len(attempts):
        raise SystemExit(
            f"arm {ARM_NAME}: skill preview in {skill_hits}/{len(attempts)} "
            "prompts"
        )
    if episodes or memory_hits:
        raise SystemExit(
            f"arm {ARM_NAME}: memory OFF arm unexpectedly touched memory "
            f"(episodes={episodes}, injected={memory_hits})"
        )
    print(f"=== arm {ARM_NAME} verified: {result} ===", flush=True)
    return result


def main() -> None:
    import fire

    def run(run_id: str = "", cleanup_db: bool = True) -> dict:
        _load_own_glm5()
        _preflight_config()
        from src.evolution.controller import EvolutionController
        from src.evolution.fitness import evaluate_arm
        from src.evolution.splits import filter_tasks, load_tasks

        run_id = run_id or "skillonly_r1_24m"
        out_root = RUNS_ROOT / run_id

        ctrl = EvolutionController(
            repo_root=HERMES_ROOT,
            snapshot_dir=SNAPSHOT,
            config_name=CONFIG_NAME,
            train_months=24,
            dev_months=0,
            holdout_months=0,
        )
        ctrl.runs_root = RUNS_ROOT

        tasks = load_tasks(
            SNAPSHOT / "tasks" / "ashare_trader_open" / "standardized_data.jsonl"
        )
        months = tuple(str(t["metadata"]["as_of"]) for t in tasks)
        if len(months) != 24:
            raise SystemExit(f"expected 24 months in 24m snapshot, got {len(months)}")
        if not (R1_SKILL / "ashare_open_portfolio.md").exists():
            raise SystemExit(f"missing R1 skill: {R1_SKILL}")

        print(f"=== skill-only R1 ablation {run_id} "
              f"months={months[0]}..{months[-1]} ===")
        print(f"=== key=own_glm5 skill={SKILL_SHORT} config={CONFIG_NAME} "
              f"snapshot={SNAPSHOT.name} ===")
        print(f"=== arm {ARM_NAME} (FINAL skill, memory OFF) ===", flush=True)

        ctrl.run_arm(
            run_id,
            ARM_NAME,
            R1_SKILL,
            level="train",
            months=months,
            cleanup_db=cleanup_db,
        )
        arm_check = _verify_arm(out_root / "arms" / ARM_NAME)

        reports_dir = out_root / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        segments = {"full_24m": (months[0][:7], months[-1][:7]), **SEGMENTS}
        summary: dict[str, dict] = {}
        for seg_name, (first, last) in segments.items():
            seg_months = _month_range(months, first, last)
            if not seg_months:
                continue
            subset = filter_tasks(tasks, seg_months)
            arm = evaluate_arm(
                out_root / "arms" / ARM_NAME / "out", subset, ctrl.snapshot_db
            )
            report = {
                "level": seg_name,
                "run_id": run_id,
                "months": list(seg_months),
                "api_key_file": KEY_FILE.name,
                "config_name": CONFIG_NAME,
                "skill": SKILL_SHORT,
                "arm": ARM_NAME,
                "role": "skill_only_of_final_r1",
                "metrics": {
                    k: arm[k]
                    for k in (
                        "total_return",
                        "index_return",
                        "excess_return",
                        "max_drawdown",
                        "worst_month",
                        "win_rate",
                        "fees",
                    )
                },
                "arm_check": arm_check,
                "note": (
                    "leave-one-out skill-only arm of final R1 "
                    f"{SKILL_SHORT}: skill ON, memory OFF"
                ),
            }
            (reports_dir / f"fitness_{seg_name}.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            m = report["metrics"]
            md = (
                f"# Skill-only R1 — {seg_name} run={run_id}\n\n"
                f"Skill: `{SKILL_SHORT}` | memory: OFF\n\n"
                f"Months: {', '.join(seg_months)}\n\n"
                f"| total | index | excess | maxDD | worst | win | fees |\n"
                f"|---:|---:|---:|---:|---:|---:|---:|\n"
                f"| {m['total_return']*100:+.2f}% "
                f"| {m['index_return']*100:+.2f}% "
                f"| {m['excess_return']*100:+.2f}% "
                f"| {m['max_drawdown']*100:.2f}% "
                f"| {m['worst_month']*100:+.2f}% "
                f"| {m['win_rate']*100:.0f}% "
                f"| {m['fees']:.0f} |\n"
            )
            (reports_dir / f"fitness_{seg_name}.md").write_text(md, encoding="utf-8")
            summary[seg_name] = {
                "months": f"{seg_months[0]}..{seg_months[-1]}",
                **report["metrics"],
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
                    "skill": SKILL_SHORT,
                    "role": "skill_only_of_final_r1",
                    "arm_check": arm_check,
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
