#!/usr/bin/env python3
"""Offline smoke tests for the A-share ranking benchmark."""

from __future__ import annotations

import asyncio
import io
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.ashare.gen_rank_tasks import main as generate_tasks  # noqa: E402
from scripts.ashare.eval_rank import (  # noqa: E402
    _sign_flip_pvalue,
    paired_comparisons,
)
from common_benchmark import BenchmarkEvaluator  # noqa: E402
from src.memory.memory import Mem0Memory  # noqa: E402
from src.memory.rank_reflection import (  # noqa: E402
    build_rank_factor_block,
    factor_reliability,
)
from src.memory.vector_store import VectorStore  # noqa: E402
from src.tool.mcp_servers.ashare_mcp_server import (  # noqa: E402
    ashare_cross_section_snapshot,
)
from src.utils.ashare_rank import evaluate_ranking, parse_ranked_codes  # noqa: E402


def main() -> None:
    generate_tasks()
    task_path = ROOT / "data" / "ashare_rank" / "standardized_data.jsonl"
    tasks = [
        json.loads(line)
        for line in task_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(tasks) == 12
    for task in tasks:
        metadata = task["metadata"]
        pool = metadata["stock_pool"]
        truth = metadata["ground_truth_rank"]
        assert len(pool) == len(set(pool)) == 16
        assert set(truth) == set(pool)
        assert truth == sorted(
            pool,
            key=lambda code: (-metadata["excess_returns"][code], code),
        )

    first = tasks[0]
    pool = first["metadata"]["stock_pool"]
    truth = first["metadata"]["ground_truth_rank"]
    raw = "\\boxed{" + ",".join(truth) + "}"
    parsed = parse_ranked_codes(raw, pool)
    assert parsed.ok and parsed.codes == truth
    assert not parse_ranked_codes(",".join(truth[:-1]), pool).ok
    assert not parse_ranked_codes(",".join(truth[:-1] + [truth[0]]), pool).ok

    perfect = evaluate_ranking(
        truth,
        truth,
        first["metadata"]["excess_returns"],
    )
    reversed_metrics = evaluate_ranking(
        list(reversed(truth)),
        truth,
        first["metadata"]["excess_returns"],
    )
    assert abs(perfect["rank_ic"] - 1.0) < 1e-12
    assert abs(reversed_metrics["rank_ic"] + 1.0) < 1e-12
    assert perfect["spread"] > 0
    assert abs(_sign_flip_pvalue([1.0, 1.0, 1.0, 1.0]) - 0.125) < 1e-12
    paired = paired_comparisons(
        [
            (
                {"run": "baseline"},
                [
                    {
                        "task_id": "m1",
                        "parse_ok": True,
                        "rank_ic": 0.0,
                        "top_excess": 0.01,
                        "spread": 0.00,
                    }
                ],
            ),
            (
                {"run": "memory"},
                [
                    {
                        "task_id": "m1",
                        "parse_ok": True,
                        "rank_ic": 0.2,
                        "top_excess": 0.03,
                        "spread": 0.01,
                    }
                ],
            ),
        ]
    )
    assert paired[0]["delta_mean_rank_ic"] == 0.2
    assert abs(paired[0]["delta_mean_top4_excess"] - 0.02) < 1e-12

    snapshot = ashare_cross_section_snapshot.fn(first["metadata"]["entry_date"])
    frame = pd.read_csv(io.StringIO("\n".join(snapshot.splitlines()[2:])))
    assert len(frame) == 16
    assert set(frame["ts_code"]) == set(pool)
    assert {
        "rel20",
        "pe_pct",
        "ml_score",
        "ml_rank",
        "financial_ann_date",
        "netprofit_yoy",
    }.issubset(frame.columns)

    synthetic = []
    for month in range(1, 9):
        for rank in range(1, 17):
            synthetic.append(
                {
                    "task_id": f"rank_{month}_{rank}",
                    "entry_month": f"2024-{month:02d}",
                    "entry_date": f"2024{month:02d}01",
                    "exit_date": f"2024{month:02d}28",
                    "ts_code": f"{rank:06d}.SZ",
                    "rel20": float(rank),
                    "ml_rank": 17 - rank,
                    "excess_return": float(rank) / 100,
                }
            )
    stats = factor_reliability(synthetic, "20240901")
    rel20 = next(row for row in stats if row["feature"] == "rel20")
    assert (
        rel20["n_months"] == 8
        and rel20["mean_ic"] > 0.99
        and rel20["q_value"] <= 0.10
    )
    with tempfile.TemporaryDirectory() as tmp:
        memory = Mem0Memory(
            VectorStore(store_dir=tmp, namespace="rank"),
            api_key="smoke-test-key",
        )
        memory.log_samples(synthetic)
        assert build_rank_factor_block(memory, "20240501", min_months=3) == ""
        status_block = build_rank_factor_block(
            memory,
            "20240501",
            min_months=3,
            show_status_when_empty=True,
        )
        assert "历史因子记忆状态" in status_block and "没有因子通过" in status_block
        block = build_rank_factor_block(memory, "20240901", min_months=3)
        assert (
            "历史因子可靠性" in block
            and "近20日相对动量" in block
            and "FDR q=" in block
        )

    class DispatchProbe:
        cfg = OmegaConf.create(
            {"memory": {"enabled": True, "reflection_enabled": True, "reflection_mode": "rank_factor"}}
        )
        called = False

        async def _log_rank_factor_samples(self, month, month_tasks, month_results):
            self.called = True

    probe = DispatchProbe()
    asyncio.run(
        BenchmarkEvaluator._refresh_rolling_reflection(
            probe, "2024-07", []  # type: ignore[arg-type]
        )
    )
    asyncio.run(
        BenchmarkEvaluator._log_rolling_samples(
            probe, "2024-07", [], []  # type: ignore[arg-type]
        )
    )
    assert probe.called
    print(
        "[OK] 12 ranking tasks; parser/RankIC metrics; "
        "16-stock point-in-time batch snapshot; rank-factor memory"
    )


if __name__ == "__main__":
    main()
