# SPDX-FileCopyrightText: 2026 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

"""Deterministic financial fitness for skill candidates.

Wraps the frozen open-universe evaluator (``scripts/ashare/eval_open_trader.py``:
lot rounding, real fees, sequential compounding) and reduces a paired
baseline/candidate comparison to:

- per-month paired diffs with an exact two-sided sign test,
- hard vetoes (invalid outputs, drawdown blow-ups) that no score can override,
- a scalar ranking score used ONLY to order surviving candidates.

No LLM is involved anywhere in this module.
"""

from __future__ import annotations

import importlib.util
import math
import sqlite3
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EVALUATOR_PATH = REPO_ROOT / "scripts" / "ashare" / "eval_open_trader.py"

# Hard-veto thresholds.
MAX_INVALID_MONTHS = 0
MAX_DRAWDOWN_DEGRADATION_PP = 5.0  # candidate mdd may not be >5pp worse

# Ranking-score shape.
DRAWDOWN_PENALTY_WEIGHT = 0.25


def _load_evaluator():
    spec = importlib.util.spec_from_file_location(
        "hermes_eval_open_trader", EVALUATOR_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_EVAL = None


def evaluator():
    global _EVAL
    if _EVAL is None:
        _EVAL = _load_evaluator()
    return _EVAL


def evaluate_arm(
    run_dir: str | Path, tasks: list[dict], db_path: str | Path
) -> dict:
    """Replay one arm's boxed allocations over ``tasks`` with real frictions."""
    ev = evaluator()
    allocations = ev.extract_run_allocations(Path(run_dir))
    conn = sqlite3.connect(str(db_path))
    try:
        market = ev.Market(conn)
        total, months = ev.replay(market, tasks, allocations)
    finally:
        conn.close()
    metrics = ev.replay_metrics(total, months)
    index_total = 1.0
    for month in months:
        index_total *= 1.0 + float(month.get("index", 0.0))
    invalid_months = [m["as_of"] for m in months if "note" in m]
    return {
        "run_dir": str(run_dir),
        "months": months,
        "total_return": total,
        "index_return": index_total - 1.0,
        "excess_return": total - (index_total - 1.0),
        "max_drawdown": metrics["max_drawdown"],
        "worst_month": metrics["worst_month"],
        "win_rate": metrics["win_rate"],
        "fees": metrics["fees"],
        "invalid_months": invalid_months,
    }


def _sign_test_p(wins: int, losses: int) -> float:
    """Exact two-sided sign test over non-tied paired months."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2**n
    return min(1.0, 2.0 * tail)


def paired_stats(baseline_months: list[dict], candidate_months: list[dict]) -> dict:
    base = {m["as_of"]: float(m["net"]) for m in baseline_months}
    cand = {m["as_of"]: float(m["net"]) for m in candidate_months}
    common = sorted(set(base) & set(cand))
    if set(base) != set(cand):
        raise ValueError(
            f"paired months mismatch: baseline={sorted(base)} candidate={sorted(cand)}"
        )
    diffs = [cand[a] - base[a] for a in common]
    wins = sum(d > 0 for d in diffs)
    losses = sum(d < 0 for d in diffs)
    return {
        "months": common,
        "diffs_pp": [round(d * 100, 4) for d in diffs],
        "mean_diff_pp": round(statistics.mean(diffs) * 100, 4) if diffs else 0.0,
        "stdev_diff_pp": (
            round(statistics.stdev(diffs) * 100, 4) if len(diffs) > 1 else 0.0
        ),
        "wins": wins,
        "losses": losses,
        "ties": len(diffs) - wins - losses,
        "sign_test_p": round(_sign_test_p(wins, losses), 6),
    }


def hard_gates(baseline: dict, candidate: dict) -> dict:
    """Vetoes that reject a candidate regardless of its mean return edge."""
    failures: list[str] = []
    if len(candidate["invalid_months"]) > MAX_INVALID_MONTHS:
        failures.append(
            f"invalid/unparseable allocations in months {candidate['invalid_months']}"
        )
    degradation_pp = (baseline["max_drawdown"] - candidate["max_drawdown"]) * 100
    if degradation_pp > MAX_DRAWDOWN_DEGRADATION_PP:
        failures.append(
            "max drawdown degraded by "
            f"{degradation_pp:.2f}pp (> {MAX_DRAWDOWN_DEGRADATION_PP}pp cap)"
        )
    return {"passed": not failures, "failures": failures}


def ranking_score(paired: dict, baseline: dict, candidate: dict) -> float:
    """Order surviving candidates: mean paired edge minus a drawdown penalty."""
    degradation_pp = max(
        0.0, (baseline["max_drawdown"] - candidate["max_drawdown"]) * 100
    )
    return round(
        paired["mean_diff_pp"] - DRAWDOWN_PENALTY_WEIGHT * degradation_pp, 6
    )


def fitness_report(
    level: str,
    baseline_arm: dict,
    candidate_arm: dict,
) -> dict:
    paired = paired_stats(baseline_arm["months"], candidate_arm["months"])
    gates = hard_gates(baseline_arm, candidate_arm)

    def _summary(arm: dict) -> dict:
        return {
            "run_dir": arm["run_dir"],
            "total_return": round(arm["total_return"], 6),
            "index_return": round(arm["index_return"], 6),
            "excess_return": round(arm["excess_return"], 6),
            "max_drawdown": round(arm["max_drawdown"], 6),
            "worst_month": round(arm["worst_month"], 6),
            "win_rate": round(arm["win_rate"], 4),
            "fees": round(arm["fees"], 2),
            "invalid_months": arm["invalid_months"],
        }

    return {
        "level": level,
        "baseline": _summary(baseline_arm),
        "candidate": _summary(candidate_arm),
        "paired": paired,
        "gates": gates,
        "score": ranking_score(paired, baseline_arm, candidate_arm),
    }
