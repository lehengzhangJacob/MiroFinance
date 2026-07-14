# SPDX-FileCopyrightText: 2026 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

"""Chronological train / dev / holdout splits over monthly trader tasks.

Months are ordered by ``as_of``; there is no shuffling anywhere. The holdout
tail is sealed: the controller only touches it through the registry's
one-shot lease, and the generator never sees its outcomes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MonthSplits:
    train: tuple[str, ...]
    dev: tuple[str, ...]
    holdout: tuple[str, ...]

    def level_months(self, level: str) -> tuple[str, ...]:
        if level == "train":
            return self.train
        if level == "dev":
            return self.dev
        if level == "holdout":
            return self.holdout
        if level == "probe":
            # Cheap fidelity: first two train months.
            return self.train[:2]
        raise ValueError(f"unknown level {level!r}")


def load_tasks(tasks_path: str | Path) -> list[dict]:
    tasks = [
        json.loads(line)
        for line in Path(tasks_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    tasks.sort(key=lambda t: str(t["metadata"]["as_of"]))
    return tasks


def make_splits(
    tasks: list[dict],
    train_months: int = 6,
    dev_months: int = 3,
    holdout_months: int = 3,
) -> MonthSplits:
    as_ofs = [str(t["metadata"]["as_of"]) for t in tasks]
    if len(set(as_ofs)) != len(as_ofs):
        raise ValueError("duplicate as_of dates in task file")
    if len(as_ofs) < train_months + dev_months + holdout_months:
        raise ValueError(
            f"need >= {train_months + dev_months + holdout_months} months, "
            f"got {len(as_ofs)}"
        )
    return MonthSplits(
        train=tuple(as_ofs[:train_months]),
        dev=tuple(as_ofs[train_months : train_months + dev_months]),
        holdout=tuple(
            as_ofs[
                train_months + dev_months : train_months
                + dev_months
                + holdout_months
            ]
        ),
    )


def filter_tasks(tasks: list[dict], months: tuple[str, ...]) -> list[dict]:
    wanted = set(months)
    return [t for t in tasks if str(t["metadata"]["as_of"]) in wanted]


def write_task_subset(
    tasks: list[dict], months: tuple[str, ...], dest_dir: str | Path
) -> Path:
    """Materialize a month-subset task file in benchmark layout.

    ``dest_dir`` becomes a DATA_DIR: the benchmark config appends
    ``ashare_trader_open/standardized_data.jsonl``.
    """
    subset = filter_tasks(tasks, months)
    if len(subset) != len(months):
        found = {str(t["metadata"]["as_of"]) for t in subset}
        raise ValueError(f"months missing from task file: {set(months) - found}")
    out_dir = Path(dest_dir) / "ashare_trader_open"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "standardized_data.jsonl"
    out_path.write_text(
        "".join(json.dumps(t, ensure_ascii=False) + "\n" for t in subset),
        encoding="utf-8",
    )
    return out_path
