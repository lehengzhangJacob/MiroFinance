#!/usr/bin/env python3
"""Two-process write/read smoke test for official Mem0 + shared Qdrant."""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

NAMESPACE = "__qdrant_concurrency_smoke__"


def _worker(worker_id: int, barrier: mp.synchronize.Barrier, queue: mp.Queue) -> None:
    try:
        from src.memory.official_mem0_store import OfficialMem0Store

        store = OfficialMem0Store(ROOT / "memory_bank", namespace=NAMESPACE)
        barrier.wait(timeout=30)
        for item_id in range(2):
            store.add(
                (
                    f"并发验证 worker={worker_id} item={item_id}："
                    "该记录仅用于确认多进程共享 Qdrant 写入。"
                ),
                metadata={
                    "source": "concurrency_smoke",
                    "entry_month": "2024-07",
                    "available_after": "20240731",
                    "functional_stance": "neutral",
                    "worker_id": worker_id,
                    "item_id": item_id,
                },
            )
        queue.put(("ok", worker_id))
    except BaseException as exc:  # noqa: BLE001 - child error must reach parent
        queue.put(("error", worker_id, repr(exc)))


def main() -> int:
    if not os.getenv("GLM_API_KEY"):
        raise RuntimeError("GLM_API_KEY is required")

    from src.memory.official_mem0_store import OfficialMem0Store

    coordinator = OfficialMem0Store(ROOT / "memory_bank", namespace=NAMESPACE)
    coordinator.reset_namespace()

    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(2)
    queue = ctx.Queue()
    processes = [
        ctx.Process(target=_worker, args=(worker_id, barrier, queue))
        for worker_id in range(2)
    ]
    try:
        for process in processes:
            process.start()
        results = [queue.get(timeout=120) for _ in processes]
        for process in processes:
            process.join(timeout=30)
        failures = [result for result in results if result[0] != "ok"] + [
            ("exit", process.pid, process.exitcode)
            for process in processes
            if process.exitcode != 0
        ]
        if failures:
            raise RuntimeError(f"concurrent workers failed: {failures}")

        records = coordinator.all_records()
        identities = {
            (
                int(record.metadata["worker_id"]),
                int(record.metadata["item_id"]),
            )
            for record in records
        }
        expected = {(0, 0), (0, 1), (1, 0), (1, 1)}
        if identities != expected:
            raise AssertionError(
                f"expected four unique records, got {sorted(identities)}"
            )
        print("Qdrant concurrency smoke passed: 2 processes, 4 unique records.")
        return 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        coordinator.reset_namespace()


if __name__ == "__main__":
    raise SystemExit(main())
