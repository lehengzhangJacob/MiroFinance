# SPDX-FileCopyrightText: 2026 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

import json
import math
import statistics

import pytest

from src.evolution.fitness import (
    _sign_test_p,
    evaluator,
    fitness_report,
    hard_gates,
    paired_stats,
)
from src.evolution.splits import make_splits, write_task_subset
from tests.evolution.conftest import make_tasks

MONTHS_12 = [
    "2024-07-01",
    "2024-08-01",
    "2024-09-02",
    "2024-10-09",
    "2024-11-06",
    "2024-12-04",
    "2025-01-02",
    "2025-02-07",
    "2025-03-07",
    "2025-04-07",
    "2025-05-08",
    "2025-06-06",
]


def test_splits_are_chronological_and_disjoint():
    tasks = make_tasks(MONTHS_12)
    splits = make_splits(tasks)
    assert splits.train == tuple(MONTHS_12[:6])
    assert splits.dev == tuple(MONTHS_12[6:9])
    assert splits.holdout == tuple(MONTHS_12[9:])
    assert splits.level_months("probe") == tuple(MONTHS_12[:2])
    assert not (set(splits.train) & set(splits.dev) & set(splits.holdout))


def test_splits_reject_short_task_lists():
    with pytest.raises(ValueError):
        make_splits(make_tasks(MONTHS_12[:5]))


MONTHS_24 = MONTHS_12 + [
    "2025-07-04",
    "2025-08-01",
    "2025-09-01",
    "2025-10-09",
    "2025-11-04",
    "2025-12-02",
    "2026-01-05",
    "2026-02-03",
    "2026-03-04",
    "2026-04-01",
    "2026-05-06",
    "2026-06-03",
]


def test_splits_24_months_12_6_6():
    tasks = make_tasks(MONTHS_24)
    splits = make_splits(tasks, train_months=12, dev_months=6, holdout_months=6)
    assert splits.train == tuple(MONTHS_24[:12])
    assert splits.dev == tuple(MONTHS_24[12:18])
    assert splits.holdout == tuple(MONTHS_24[18:])
    assert splits.level_months("probe") == tuple(MONTHS_24[:2])
    assert len(set(splits.train) | set(splits.dev) | set(splits.holdout)) == 24


def test_write_task_subset_layout(tmp_path):
    tasks = make_tasks(MONTHS_12)
    out = write_task_subset(tasks, tuple(MONTHS_12[:2]), tmp_path / "data")
    assert out == tmp_path / "data" / "ashare_trader_open" / "standardized_data.jsonl"
    lines = [json.loads(l) for l in out.read_text().splitlines()]
    assert [t["metadata"]["as_of"] for t in lines] == MONTHS_12[:2]
    with pytest.raises(ValueError, match="missing"):
        write_task_subset(tasks, ("1999-01-01",), tmp_path / "data2")


def test_sign_test_exact_values():
    assert _sign_test_p(0, 0) == 1.0
    assert _sign_test_p(2, 0) == pytest.approx(0.5)
    assert _sign_test_p(10, 0) == pytest.approx(2 / 1024)
    assert _sign_test_p(6, 6) == pytest.approx(1.0)


def _arm(
    months_net: dict[str, float],
    mdd: float = -0.10,
    invalid=(),
    sharpe: float | None = 1.0,
) -> dict:
    months = [
        {"as_of": k, "net": v, "index": 0.0, "capital": 1.0}
        for k, v in months_net.items()
    ]
    for a in invalid:
        months.append({"as_of": a, "net": 0.0, "index": 0.0, "note": "invalid->cash"})
    return {
        "run_dir": "x",
        "months": months,
        "total_return": sum(months_net.values()),
        "index_return": 0.0,
        "excess_return": sum(months_net.values()),
        "max_drawdown": mdd,
        "annualized_sharpe": sharpe,
        "worst_month": min(months_net.values(), default=0.0),
        "win_rate": 0.5,
        "fees": 100.0,
        "invalid_months": list(invalid),
    }


def test_paired_stats_and_mismatch_detection():
    base = _arm({"a": 0.01, "b": -0.02})
    cand = _arm({"a": 0.03, "b": -0.01})
    stats = paired_stats(base["months"], cand["months"])
    assert stats["wins"] == 2 and stats["losses"] == 0
    assert stats["mean_diff_pp"] == pytest.approx(1.5)
    with pytest.raises(ValueError, match="mismatch"):
        paired_stats(base["months"], _arm({"a": 0.03})["months"])


def test_hard_gates_veto_invalid_and_drawdown():
    base = _arm({"a": 0.01}, mdd=-0.10)
    ok = _arm({"a": 0.02}, mdd=-0.12)
    assert hard_gates(base, ok)["passed"]

    invalid = _arm({"a": 0.02}, invalid=("b",))
    assert not hard_gates(base, invalid)["passed"]

    blowup = _arm({"a": 0.02}, mdd=-0.16)
    result = hard_gates(base, blowup)
    assert not result["passed"]
    assert any("drawdown" in f for f in result["failures"])


def test_fitness_report_score_penalizes_drawdown():
    base = _arm({"a": 0.01, "b": 0.01}, mdd=-0.10)
    cand = _arm({"a": 0.02, "b": 0.02}, mdd=-0.14)
    report = fitness_report("probe", base, cand)
    assert report["paired"]["mean_diff_pp"] == pytest.approx(1.0)
    # score = 1.0 - 0.25 * 4pp degradation = 0.0
    assert report["score"] == pytest.approx(0.0)
    assert report["gates"]["passed"]


def test_annualized_sharpe_formula_and_edge_cases():
    ev = evaluator()
    rets = [0.02, -0.01, 0.03, 0.00, -0.02, 0.04]
    expected = math.sqrt(12.0) * statistics.mean(rets) / statistics.stdev(rets)
    assert ev.annualized_sharpe(rets) == pytest.approx(expected)
    # A flat monthly risk-free rate equal to the mean zeroes the ratio.
    assert ev.annualized_sharpe(
        rets, risk_free_monthly=statistics.mean(rets)
    ) == pytest.approx(0.0)
    # Fewer than two observations or zero volatility -> None (rendered as "—").
    assert ev.annualized_sharpe([]) is None
    assert ev.annualized_sharpe([0.05]) is None
    assert ev.annualized_sharpe([0.01, 0.01, 0.01]) is None


def test_replay_metrics_include_sharpe_with_cash_fallback_months():
    ev = evaluator()
    months = [
        {"as_of": "a", "net": 0.02, "capital": 1.02, "fees": 1.0},
        {"as_of": "b", "net": -0.01, "capital": 1.01, "fees": 1.0},
        # Invalid month falls back to cash and counts as a 0-return month.
        {"as_of": "c", "net": 0.0, "capital": 1.01, "note": "invalid->cash"},
    ]
    metrics = ev.replay_metrics(0.01, months)
    expected = ev.annualized_sharpe([0.02, -0.01, 0.0])
    assert metrics["annualized_sharpe"] == pytest.approx(expected)
    assert metrics["valid_months"] == 2.0


def test_fitness_report_carries_and_rounds_sharpe():
    base = _arm({"a": 0.01, "b": 0.02}, sharpe=1.23456789)
    cand = _arm({"a": 0.02, "b": 0.03}, sharpe=None)
    report = fitness_report("probe", base, cand)
    assert report["baseline"]["annualized_sharpe"] == pytest.approx(1.2346)
    assert report["candidate"]["annualized_sharpe"] is None
    # Report must stay JSON-serializable with a missing-sharpe arm.
    json.dumps(report)
