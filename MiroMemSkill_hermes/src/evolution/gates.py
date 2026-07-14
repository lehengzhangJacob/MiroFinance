# SPDX-FileCopyrightText: 2026 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

"""L0 static gates: free checks every candidate must pass before any rollout.

The gates are differential where possible — a pattern already present in the
baseline is allowed, anything new of that kind fails closed. This keeps the
gate honest as the baseline evolves without hard-coding today's skill text.
"""

from __future__ import annotations

import re

from src.evolution.types import GateResult, SkillArtifact

# Securities identifiers and date literals must not be introduced by a
# candidate: skills are procedural and point-in-time generic by contract.
TICKER_RE = re.compile(r"\b\d{6}\.(?:SH|SZ|BJ)\b")
DATE_RE = re.compile(r"\b(?:20\d{2}[-/]\d{1,2}[-/]\d{1,2}|20\d{6})\b")

MAX_BODY_CHARS = 12_000
MAX_GROWTH_RATIO = 1.6
MIN_GROWTH_ABS = 2_000  # short baselines may still grow by this many chars


def run_static_gates(
    baseline: SkillArtifact,
    candidate: SkillArtifact,
    max_body_chars: int = MAX_BODY_CHARS,
    max_growth_ratio: float = MAX_GROWTH_RATIO,
    min_growth_abs: int = MIN_GROWTH_ABS,
) -> GateResult:
    failures: list[str] = []
    details: dict = {}

    if candidate.name != baseline.name:
        failures.append(
            f"name changed: {baseline.name!r} -> {candidate.name!r}"
        )

    if candidate.frontmatter.strip() != baseline.frontmatter.strip():
        failures.append("frontmatter modified (must stay byte-identical)")

    body = candidate.body.strip()
    if not body:
        failures.append("empty body")

    details["baseline_chars"] = len(baseline.body)
    details["candidate_chars"] = len(candidate.body)
    if len(candidate.body) > max_body_chars:
        failures.append(
            f"body too long: {len(candidate.body)} > {max_body_chars} chars"
        )
    growth_cap = max(
        int(len(baseline.body) * max_growth_ratio),
        len(baseline.body) + min_growth_abs,
    )
    if len(candidate.body) > growth_cap:
        failures.append(
            f"body grew too much: {len(candidate.body)} > cap {growth_cap} chars"
        )

    new_tickers = set(TICKER_RE.findall(candidate.body)) - set(
        TICKER_RE.findall(baseline.body)
    )
    if new_tickers:
        failures.append(f"introduces ticker codes: {sorted(new_tickers)}")

    new_dates = set(DATE_RE.findall(candidate.body)) - set(
        DATE_RE.findall(baseline.body)
    )
    if new_dates:
        failures.append(f"introduces date literals: {sorted(new_dates)}")

    if candidate.digest == baseline.digest:
        failures.append("candidate is byte-identical to baseline")

    return GateResult(passed=not failures, failures=failures, details=details)
