#!/usr/bin/env python3
"""Summarize validation run metrics into validation_report.md."""

import json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"


def latest_metrics(glob_pattern: str) -> dict | None:
    dirs = sorted(OUTPUT.glob(glob_pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    for d in dirs:
        mf = d / "metrics.json"
        if mf.exists():
            return json.loads(mf.read_text())
    return None


def main():
    phases = [
        ("Phase 1 Skill", "commit_message_writer/*/metrics.json", latest_metrics("commit_message_writer/*")),
        ("Phase 2 Tools", "tools_dev/*/metrics.json", latest_metrics("tools_dev/*")),
        ("Phase 3 Prompt", "prompt_coding_agent_guidelines/*/metrics.json", latest_metrics("prompt_coding_agent_guidelines/*")),
        ("Phase 4 Code", "code_text_helpers/*/metrics.json", latest_metrics("code_text_helpers/*")),
    ]

    lines = [
        "# Hermes Self-Evolution Validation Report",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Summary",
        "",
        "| Phase | Baseline | Evolved | Improvement |",
        "|-------|----------|---------|-------------|",
    ]

    improved = 0
    for name, _, m in phases:
        if not m:
            lines.append(f"| {name} | — | — | — |")
            continue
        b = m.get("baseline_score", 0)
        e = m.get("evolved_score", 0)
        imp = m.get("improvement", e - b)
        if imp > 0:
            improved += 1
        lines.append(f"| {name} | {b:.3f} | {e:.3f} | {imp:+.3f} |")

    lines.extend([
        "",
        f"**Phases improved:** {improved}/4",
        "",
        "## GEPA traces",
        "",
        "Inspect `output/*/gepa_trace.jsonl` for proposed mutations and accept/reject decisions.",
        "",
        "## Mechanism",
        "",
        "Hermes self-evolution = **text mutation (GEPA)** + **task-specific fitness** + **constraint gates (size, structure, pytest)**.",
        "Independent dev-assistant scenario; no external project dependencies.",
    ])

    report = ROOT / "validation_report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {report}")


if __name__ == "__main__":
    main()
