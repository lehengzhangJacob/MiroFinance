# SPDX-FileCopyrightText: 2026 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

"""Freeze the 24-month open-universe snapshot.

Splices the task file so the frozen 12 months stay byte-identical to the
original benchmark (same stock_pool despite stock_basic_all having been
refreshed to 2026-07 names), while the extended months come from the
synthesized generator run against the extended mirror:

    months  2024-07 .. 2025-06  -> rows copied verbatim from the frozen file
    months  2025-07 .. 2026-06  -> synthesized (chained windows, parity-checked)

Also writes manifest.json with sha256 + stats, mirroring the original
snapshot's manifest layout.

Usage:
    python scripts/ashare/freeze_open_snapshot_24m.py \
        --snapshot-dir /path/to/new_snapshot \
        --frozen-tasks /path/to/old_snapshot/tasks/.../standardized_data.jsonl \
        --generated-tasks /tmp/gen_24m/standardized_data.jsonl
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sqlite3
from pathlib import Path


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-dir", required=True)
    parser.add_argument("--frozen-tasks", required=True)
    parser.add_argument("--generated-tasks", required=True)
    parser.add_argument("--splice-from", default="2025-07")
    args = parser.parse_args()

    snapshot_dir = Path(args.snapshot_dir)
    db_path = snapshot_dir / "ashare_pools_snapshot.db"
    tasks_out = snapshot_dir / "tasks" / "ashare_trader_open" / "standardized_data.jsonl"
    tasks_out.parent.mkdir(parents=True, exist_ok=True)

    frozen_rows = [
        line
        for line in Path(args.frozen_tasks).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    generated_rows = [
        line
        for line in Path(args.generated_tasks).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    def as_of(row: str) -> str:
        return json.loads(row)["metadata"]["as_of"]

    frozen_months = {as_of(r)[:7] for r in frozen_rows}
    spliced = list(frozen_rows)
    added = []
    for row in generated_rows:
        month = as_of(row)[:7]
        if month >= args.splice_from:
            if month in frozen_months:
                raise ValueError(f"splice month {month} already frozen")
            spliced.append(row)
            added.append(as_of(row))
    spliced.sort(key=as_of)
    if len(spliced) != len(frozen_rows) + len(added):
        raise ValueError("splice produced duplicate months")

    tasks_out.write_text("".join(r + "\n" for r in spliced), encoding="utf-8")
    print(f"tasks: {len(frozen_rows)} frozen + {len(added)} synthesized -> {tasks_out}")

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
    stats = {
        "integrity_check": integrity,
        "market_daily_rows": conn.execute("SELECT COUNT(*) FROM market_daily").fetchone()[0],
        "market_daily_basic_rows": conn.execute(
            "SELECT COUNT(*) FROM market_daily_basic"
        ).fetchone()[0],
        "etf_daily_rows": conn.execute("SELECT COUNT(*) FROM etf_daily").fetchone()[0],
        "index_daily_rows": conn.execute("SELECT COUNT(*) FROM index_daily").fetchone()[0],
        "trade_cal_rows": conn.execute("SELECT COUNT(*) FROM trade_cal").fetchone()[0],
        "stock_basic_all_rows": conn.execute(
            "SELECT COUNT(*) FROM stock_basic_all"
        ).fetchone()[0],
        "market_daily_codes": conn.execute(
            "SELECT COUNT(DISTINCT ts_code) FROM market_daily"
        ).fetchone()[0],
        "market_daily_dates": list(
            conn.execute("SELECT MIN(trade_date), MAX(trade_date) FROM market_daily").fetchone()
        ),
    }
    conn.close()

    metas = [json.loads(r)["metadata"] for r in spliced]
    manifest = {
        "version": 1,
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "scope": "24-month open-universe skill-evolution benchmark",
        "derived_from": "shared/ashare_open_stocks_glm52_20260714 (first 12 months byte-identical)",
        "notes": [
            "months 2024-07..2025-06 copied verbatim from the frozen 12-month benchmark",
            "months 2025-07..2026-06 synthesized from the extended mirror (chained 20-session windows)",
            "stock_basic_all refreshed 2026-07: ST flags are current-name based for ALL months "
            "(same documented PIT caveat as the original benchmark)",
        ],
        "artifacts": {
            "database": {
                "path": "ashare_pools_snapshot.db",
                "size": db_path.stat().st_size,
                "sha256": sha256_file(db_path),
            },
            "tasks": {
                "path": "tasks/ashare_trader_open/standardized_data.jsonl",
                "size": tasks_out.stat().st_size,
                "sha256": sha256_file(tasks_out),
            },
        },
        "database_stats": stats,
        "task_stats": {
            "task_count": len(metas),
            "task_ids": [f"ashare_open_trader_{m['as_of']}" for m in metas],
            "entry_dates": [m["entry_date"] for m in metas],
            "exit_dates": [m["exit_date"] for m in metas],
            "index_returns": [m["index_return"] for m in metas],
            "pool_sizes": [m["pool_size"] for m in metas],
        },
    }
    manifest_path = snapshot_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"manifest -> {manifest_path}")
    print(f"db sha256: {manifest['artifacts']['database']['sha256']}")
    print(f"tasks sha256: {manifest['artifacts']['tasks']['sha256']}")


if __name__ == "__main__":
    main()
