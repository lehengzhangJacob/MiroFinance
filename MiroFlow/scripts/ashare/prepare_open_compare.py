#!/usr/bin/env python3
"""Freeze immutable inputs for paired MiroFlow/MiroMemSkill comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_AGENT_ROOT = Path("/home/msj_team/Jacob/agent")
DEFAULT_MEMSKILL = DEFAULT_AGENT_ROOT / "MiroMemSkill"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sqlite_stats(path: Path) -> dict[str, Any]:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise RuntimeError(f"snapshot integrity check failed: {integrity}")
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        required = {
            "market_daily",
            "market_daily_basic",
            "stock_basic_all",
            "trade_cal",
            "index_daily",
        }
        missing = sorted(required - tables)
        if missing:
            raise RuntimeError(f"snapshot missing tables: {missing}")
        result: dict[str, Any] = {"integrity_check": integrity}
        for table in sorted(
            required | ({"etf_daily", "fina_cache"} & tables)
        ):
            result[f"{table}_rows"] = conn.execute(
                f"SELECT COUNT(*) FROM {table}"
            ).fetchone()[0]
        result["market_daily_codes"] = conn.execute(
            "SELECT COUNT(DISTINCT ts_code) FROM market_daily"
        ).fetchone()[0]
        result["market_daily_dates"] = list(
            conn.execute(
                "SELECT MIN(trade_date),MAX(trade_date) FROM market_daily"
            ).fetchone()
        )
        return result
    finally:
        conn.close()


def task_stats(path: Path) -> dict[str, Any]:
    tasks = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(tasks) != 12:
        raise RuntimeError(f"expected 12 open-market tasks, found {len(tasks)}")
    task_ids = [task["task_id"] for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise RuntimeError("open-market task IDs are not unique")
    entries = [task["metadata"]["entry_date"] for task in tasks]
    exits = [task["metadata"]["exit_date"] for task in tasks]
    if entries != sorted(entries):
        raise RuntimeError("open-market tasks are not in monthly order")
    if any(previous > current for previous, current in zip(exits, entries[1:])):
        raise RuntimeError("open-market task windows overlap")
    pools = [task["metadata"]["stock_pool"] for task in tasks]
    if any(len(pool) != len(set(pool)) for pool in pools):
        raise RuntimeError("an open-market stock pool contains duplicates")
    return {
        "task_count": len(tasks),
        "task_ids": task_ids,
        "entry_dates": entries,
        "exit_dates": exits,
        "pool_sizes": [len(pool) for pool in pools],
        "pool_size_min": min(map(len, pools)),
        "pool_size_max": max(map(len, pools)),
    }


def freeze_database(source: Path, destination: Path) -> None:
    temporary = destination.with_suffix(".tmp")
    temporary.unlink(missing_ok=True)
    source_conn = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    target_conn = sqlite3.connect(temporary)
    try:
        source_conn.execute("PRAGMA busy_timeout=30000")
        source_conn.backup(target_conn, pages=4096, sleep=0.05)
        target_conn.execute("PRAGMA journal_mode=DELETE")
        target_conn.commit()
    finally:
        target_conn.close()
        source_conn.close()
    temporary.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-db",
        type=Path,
        default=DEFAULT_MEMSKILL / "data" / "ashare_pools.db",
    )
    parser.add_argument(
        "--source-tasks",
        type=Path,
        default=(
            DEFAULT_MEMSKILL
            / "data"
            / "ashare_trader_open"
            / "standardized_data.jsonl"
        ),
    )
    parser.add_argument(
        "--source-server",
        type=Path,
        default=(
            DEFAULT_MEMSKILL
            / "src"
            / "tool"
            / "mcp_servers"
            / "ashare_open_mcp_server.py"
        ),
    )
    parser.add_argument(
        "--source-evaluator",
        type=Path,
        default=(
            DEFAULT_MEMSKILL
            / "scripts"
            / "ashare"
            / "eval_open_trader.py"
        ),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    sources = {
        "database": args.source_db.resolve(),
        "tasks": args.source_tasks.resolve(),
        "server": args.source_server.resolve(),
        "evaluator": args.source_evaluator.resolve(),
    }
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing comparison sources: " + ", ".join(missing))

    out = args.out.resolve()
    manifest_path = out / "manifest.json"
    if manifest_path.exists() and not args.force:
        raise FileExistsError(
            f"comparison snapshot already exists: {out}; use --force to replace"
        )
    out.mkdir(parents=True, exist_ok=True)
    task_path = out / "tasks" / "ashare_trader_open" / "standardized_data.jsonl"
    code_dir = out / "code"
    task_path.parent.mkdir(parents=True, exist_ok=True)
    code_dir.mkdir(parents=True, exist_ok=True)
    database_path = out / "ashare_pools_snapshot.db"
    server_path = code_dir / "ashare_open_mcp_server.py"
    evaluator_path = code_dir / "eval_open_trader.py"

    freeze_database(sources["database"], database_path)
    shutil.copy2(sources["tasks"], task_path)
    shutil.copy2(sources["server"], server_path)
    shutil.copy2(sources["evaluator"], evaluator_path)

    files = {
        "database": database_path,
        "tasks": task_path,
        "server": server_path,
        "evaluator": evaluator_path,
    }
    manifest = {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "stocks-only open-universe paired project comparison",
        "model": {
            "name": "glm-5.2",
            "thinking_mode": "enabled",
            "temperature": 1.0,
            "top_p": 1.0,
            "max_tokens": 32000,
            "keep_tool_result": 6,
            "max_turns": 24,
            "max_tool_calls_per_turn": 16,
        },
        "sources": {name: str(path) for name, path in sources.items()},
        "artifacts": {
            name: {
                "path": str(path.relative_to(out)),
                "size": path.stat().st_size,
                "sha256": sha256(path),
            }
            for name, path in files.items()
        },
        "database_stats": sqlite_stats(database_path),
        "task_stats": task_stats(task_path),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"frozen comparison snapshot -> {out}")
    for name, item in manifest["artifacts"].items():
        print(f"{name}: {item['sha256']} ({item['size']} bytes)")


if __name__ == "__main__":
    main()
