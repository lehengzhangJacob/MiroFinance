# SPDX-FileCopyrightText: 2026 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from src.evolution.fitness import (
    _sign_test_p,
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


def test_splits_walk_forward_skip_months():
    tasks = make_tasks(MONTHS_24)
    rounds = {
        step: make_splits(
            tasks,
            train_months=12,
            dev_months=6,
            holdout_months=2,
            skip_months=step,
        )
        for step in (0, 2, 4)
    }
    assert rounds[0].holdout == tuple(MONTHS_24[18:20])
    assert rounds[2].holdout == tuple(MONTHS_24[20:22])
    assert rounds[4].holdout == tuple(MONTHS_24[22:24])
    # windows roll together and holdouts never overlap across rounds
    assert rounds[2].train == tuple(MONTHS_24[2:14])
    assert rounds[2].dev == tuple(MONTHS_24[14:20])
    holdouts = [set(r.holdout) for r in rounds.values()]
    assert not (holdouts[0] & holdouts[1] or holdouts[1] & holdouts[2])
    with pytest.raises(ValueError, match="need >="):
        make_splits(
            tasks,
            train_months=12,
            dev_months=6,
            holdout_months=2,
            skip_months=5,
        )
    with pytest.raises(ValueError, match="skip_months"):
        make_splits(tasks, skip_months=-1)


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


def _arm(months_net: dict[str, float], mdd: float = -0.10, invalid=()) -> dict:
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
