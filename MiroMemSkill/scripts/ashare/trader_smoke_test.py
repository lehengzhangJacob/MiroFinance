#!/usr/bin/env python3
"""Offline smoke tests for the unified A-share trader benchmark."""

from __future__ import annotations

import asyncio
import io
import json
import os
import sys
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common_benchmark import BenchmarkEvaluator  # noqa: E402
from scripts.ashare.eval_trader import evaluate_allocations  # noqa: E402
from scripts.ashare.gen_trader_tasks import main as generate_tasks  # noqa: E402
from src.memory.context import build_memory_context_block, compact_task_query  # noqa: E402
from src.memory.memory import Mem0Memory  # noqa: E402
from src.memory.rank_reflection import factor_reliability  # noqa: E402
from src.memory.skills import SkillLibrary  # noqa: E402
from src.memory.store_factory import create_memory_store  # noqa: E402
from src.memory.vector_store import MemoryRecord  # noqa: E402
from src.tool.mcp_servers.ashare_mcp_server import (  # noqa: E402
    ashare_momentum_baseline,
    ashare_trader_universe_context,
)
from src.utils.ashare_trader import (  # noqa: E402
    PortfolioParseResult,
    evaluate_portfolio_month,
    parse_portfolio_weights,
)
from src.utils.ashare_trader_features import (  # noqa: E402
    compute_trader_feature_rows,
)
from utils.eval_utils import verify_answer_for_datasets  # noqa: E402


class InMemoryStore:
    """Minimal deterministic store for episode tests (no API or embedding)."""

    def __init__(self, store_dir: str | Path, namespace: str):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.namespace = namespace
        self.records: list[MemoryRecord] = []

    @contextmanager
    def _locked(self):
        yield

    def all_records(self, filters: dict[str, Any] | None = None):
        del filters
        return list(self.records)

    def add(self, content: str, metadata=None, embedding=None, source_task=""):
        del embedding, source_task
        record = MemoryRecord(
            id=str(uuid.uuid4()),
            content=content,
            metadata=dict(metadata or {}),
            embedding=[1.0],
        )
        self.records.append(record)
        return record

    def update(self, record_id: str, new_content: str, metadata_patch=None, source_task=""):
        del source_task
        record = next(item for item in self.records if item.id == record_id)
        record.content = new_content
        record.metadata.update(metadata_patch or {})
        return record

    def delete(self, record_id: str, source_task="", reason=""):
        del source_task, reason
        before = len(self.records)
        self.records = [record for record in self.records if record.id != record_id]
        return len(self.records) < before


def _load_tasks() -> list[dict[str, Any]]:
    path = ROOT / "data" / "ashare_trader" / "standardized_data.jsonl"
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_momentum_anchor_skill(tasks: list[dict[str, Any]]) -> None:
    previous = os.environ.get("MEMSKILL_EMBEDDING_ENABLED")
    os.environ["MEMSKILL_EMBEDDING_ENABLED"] = "false"
    try:
        skill_lib = SkillLibrary(ROOT / "memory_bank" / "skills_ashare")
        matches = skill_lib.match(
            compact_task_query(tasks[0]["task_question"]),
            top_k=1,
        )
        assert matches and matches[0][0].name == "ashare_portfolio_allocation"
        momentum_skill = skill_lib.get_skill("ashare_momentum_relative_strength")
        assert momentum_skill is not None
        assert Path(momentum_skill.path).name == "SKILL.md"
        assert "ashare_momentum_baseline" in momentum_skill.body

        with tempfile.TemporaryDirectory() as tmp:
            block = build_memory_context_block(
                task_description=tasks[0]["task_question"],
                store=Mem0Memory(InMemoryStore(tmp, "momentum-anchor")),
                skill_lib=skill_lib,
                memory_enabled=False,
                skill_enabled=True,
                skill_top_k=1,
                skill_preview_min_score=0.0,
                list_other_skills=False,
            )
        assert "### Top Skill Preview" in block
        assert "20 日相对动量 top4 软锚" in block
        assert "ashare_momentum_baseline" in block
        assert "不得使用未来数据" in block
    finally:
        if previous is None:
            os.environ.pop("MEMSKILL_EMBEDDING_ENABLED", None)
        else:
            os.environ["MEMSKILL_EMBEDDING_ENABLED"] = previous


def test_tasks_and_panel(tasks: list[dict[str, Any]]) -> None:
    assert len(tasks) == 12
    assert [task["metadata"]["entry_date"] for task in tasks] == sorted(
        task["metadata"]["entry_date"] for task in tasks
    )
    assert all(
        previous["metadata"]["exit_date"] <= current["metadata"]["entry_date"]
        for previous, current in zip(tasks, tasks[1:])
    ), "one capital account cannot fund overlapping trader windows"
    assert any(
        task["metadata"]["entry_shift_sessions"] > 0 for task in tasks
    ), "smoke range should exercise the holiday-overlap adjustment"
    for task in tasks:
        metadata = task["metadata"]
        pool = metadata["stock_pool"]
        assert metadata["task_type"] == "portfolio_allocation"
        assert len(pool) == len(set(pool)) == 16
        assert set(metadata["stock_returns"]) == set(pool)
        assert set(metadata["excess_returns"]) == set(pool)
        assert metadata["exit_date"] > metadata["entry_date"]
        assert metadata["entry_date"][:6] == metadata["scheduled_month"]
        assert all(code in task["task_question"] for code in pool)
        assert "stock_returns" not in task["task_question"]
        assert "excess_returns" not in task["task_question"]

    first = tasks[0]
    panel = ashare_trader_universe_context.fn(
        first["metadata"]["entry_date"], 250
    )
    frame = pd.read_csv(io.StringIO("\n".join(panel.splitlines()[3:])))
    assert len(frame) == 16
    assert set(frame["ts_code"]) == set(first["metadata"]["stock_pool"])
    assert {
        "ret250",
        "rel250",
        "vol60_ann",
        "max_dd250",
        "pe_pct250",
        "ml_rank",
        "relative_monthly_path",
    }.issubset(frame.columns)
    assert "ground_truth" not in panel
    assert "excess_return" not in panel

    momentum = ashare_momentum_baseline.fn(
        first["metadata"]["entry_date"], 20, 4
    )
    expected_top4 = (
        frame.sort_values(["rel20", "ts_code"], ascending=[False, True])
        .head(4)["ts_code"]
        .tolist()
    )
    assert "严格点时" in momentum
    assert "参考组合:" in momentum
    assert all(f"{code}:0.25" in momentum for code in expected_top4)

    code = first["metadata"]["stock_pool"][0]
    rows = compute_trader_feature_rows(
        first["metadata"]["entry_date"],
        {code: first["metadata"]["stock_info"][code]},
        data_dir=ROOT / "data" / "ashare",
        lookback_days=250,
    )
    daily = pd.read_csv(
        ROOT / "data" / "ashare" / f"daily_{code}.csv",
        dtype={"trade_date": str},
    ).sort_values("trade_date")
    cut = daily[
        daily["trade_date"] <= first["metadata"]["entry_date"]
    ]["close_qfq"].astype(float)
    expected_ret20 = (cut.iloc[-1] / cut.iloc[-21] - 1.0) * 100
    assert abs(float(rows[0]["ret20"]) - round(expected_ret20, 1)) < 1e-12
    assert str(rows[0]["financial_ann_date"]) <= first["metadata"]["entry_date"]


def test_parser_and_finance(tasks: list[dict[str, Any]]) -> None:
    first = tasks[0]
    pool = first["metadata"]["stock_pool"]
    valid = (
        f"\\boxed{{{pool[0]}:0.25,{pool[1]}:0.20,"
        f"{pool[2]}:0.10,CASH:0.45}}"
    )
    parsed = parse_portfolio_weights(valid, pool)
    assert parsed.ok and parsed.cash == 0.45
    assert not parse_portfolio_weights(
        f"\\boxed{{{pool[0]}:0.30,CASH:0.70}}", pool
    ).ok
    assert not parse_portfolio_weights(
        f"\\boxed{{{pool[0]}:0.25,CASH:0.70}}", pool
    ).ok
    assert not parse_portfolio_weights(
        f"\\boxed{{{pool[0]}:25%,CASH:75%}}", pool
    ).ok
    assert not parse_portfolio_weights(
        f"\\boxed{{{pool[0]}:0.25}}", pool
    ).ok

    weights = {code: 0.0 for code in pool}
    weights[pool[0]] = 0.25
    result = evaluate_portfolio_month(
        weights,
        0.75,
        {code: 0.10 for code in pool},
        0.02,
        starting_capital=1_000_000.0,
        excess_returns={code: 0.08 for code in pool},
    )
    expected = 750_000.0 + (250_000.0 - 125.0) * 1.10
    expected -= max(expected - 750_000.0, 0.0) * 0.0015
    assert abs(result.ending_capital - expected) < 1e-6
    assert result.total_cost > 0

    task_map = {first["task_id"]: first}
    invalid = PortfolioParseResult(
        {code: 0.0 for code in pool}, 1.0, False, "synthetic failure"
    )
    summary, monthly = evaluate_allocations(
        "invalid",
        task_map,
        {first["task_id"]: invalid},
        initial_capital=1_000_000.0,
        open_cost=0.0005,
        close_cost=0.0015,
        min_cost=5.0,
    )
    assert summary["parsed"] == 0
    assert monthly[0]["cash"] == 1.0
    assert monthly[0]["ending_capital"] == 1_000_000.0

    judge = asyncio.run(
        verify_answer_for_datasets(
            None,  # type: ignore[arg-type]
            "ashare-trader",
            first["task_question"],
            first["ground_truth"],
            valid,
            first["metadata"],
        )
    )
    assert judge == "CORRECT"


def test_episode_memory_and_factors(tasks: list[dict[str, Any]]) -> None:
    first = tasks[0]
    pool = first["metadata"]["stock_pool"]
    with tempfile.TemporaryDirectory() as tmp:
        store = InMemoryStore(tmp, "trader-a")
        memory = Mem0Memory(store=store)
        weights = {code: 0.0 for code in pool}
        weights[pool[0]] = 0.25
        operation = memory.add_trader_episode(
            task_id=first["task_id"],
            month="2024-07",
            available_after=first["metadata"]["exit_date"],
            weights=weights,
            cash=0.75,
            gross_return=0.01,
            net_return=0.009,
            index_return=-0.02,
            active_return=0.029,
            total_cost=500.0,
            contributions={pool[0]: 0.009},
            reasoning="只使用决策日可见信息。",
            parse_ok=True,
        )
        assert operation.startswith("ADD")
        assert memory.add_trader_episode(
            task_id=first["task_id"],
            month="2024-07",
            available_after=first["metadata"]["exit_date"],
            weights=weights,
            cash=0.75,
            gross_return=0.01,
            net_return=0.009,
            index_return=-0.02,
            active_return=0.029,
            total_cost=500.0,
            contributions={pool[0]: 0.009},
            reasoning="只使用决策日可见信息。",
            parse_ok=True,
        ).startswith("UNCHANGED")
        assert len(store.records) == 1
        assert memory.trader_episode_block("20240728") == ""
        visible = memory.trader_episode_block("20240801")
        assert "2024-07" in visible and "主动收益 +2.90%" in visible

        block = build_memory_context_block(
            task_description=(
                "你是同一名交易员。当前日期为 2024-08-01（收盘后）。"
                "统一管理16只股票。数据使用规则（务必遵守）：..."
            ),
            store=memory,
            skill_lib=None,
            memory_enabled=False,
            skill_enabled=False,
            trader_episode_enabled=True,
        )
        assert "已到期的个人交易日志" in block

        isolated = Mem0Memory(InMemoryStore(tmp, "trader-b"))
        assert isolated.trader_episode_block("20240801") == ""

    synthetic = []
    for month in range(1, 9):
        for rank in range(1, 17):
            synthetic.append(
                {
                    "task_id": f"trader_{month}_{rank}",
                    "entry_month": f"2024-{month:02d}",
                    "entry_date": f"2024{month:02d}01",
                    "exit_date": f"2024{month:02d}28",
                    "ts_code": f"{rank:06d}.SZ",
                    "rel250": float(rank),
                    "excess_return": float(rank) / 100.0,
                }
            )
    stats = factor_reliability(synthetic, "20240901")
    rel250 = next(row for row in stats if row["feature"] == "rel250")
    assert rel250["n_months"] == 8
    assert rel250["mean_ic"] > 0.99
    assert rel250["q_value"] <= 0.10


def test_dispatch_and_configs() -> None:
    try:
        create_memory_store(
            store_dir="memory_bank",
            namespace="../../unsafe",
        )
        raise AssertionError("unsafe run-scoped namespace was accepted")
    except ValueError:
        pass

    class DispatchProbe:
        cfg = OmegaConf.create(
            {
                "memory": {
                    "enabled": True,
                    "reflection_enabled": True,
                    "reflection_mode": "trader",
                }
            }
        )
        called = False

        async def _log_trader_month(self, month, month_tasks, month_results):
            self.called = True

    probe = DispatchProbe()
    asyncio.run(
        BenchmarkEvaluator._log_rolling_samples(
            probe, "2024-07", [], []  # type: ignore[arg-type]
        )
    )
    assert probe.called

    config_dir = str(ROOT / "config")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        clean = compose(config_name="agent_ashare_trader_kimi")
    assert clean.benchmark.execution.max_concurrent == 1
    assert clean.benchmark.execution.task_order == "monthly"
    assert clean.benchmark.name == "ashare-trader"
    assert clean.main_agent.max_turns == -1
    assert clean.main_agent.max_tool_calls_per_turn == 16
    assert clean.main_agent.keep_tool_result == 6
    assert clean.main_agent.llm.thinking_mode == "enabled"
    assert clean.main_agent.llm.temperature == 1.0
    assert clean.main_agent.llm.max_tokens == 16000

    previous = os.environ.get("ASHARE_TRADER_RUN_ID")
    os.environ["ASHARE_TRADER_RUN_ID"] = "smoke-isolated"
    try:
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            memory = compose(config_name="agent_ashare_trader_mem0_kimi")
        OmegaConf.resolve(memory)
        assert memory.memory.namespace == "ashare_trader_smoke-isolated"
        assert memory.memory.reflection_mode == "trader"
        assert memory.memory.skill_top_k == 1
        assert memory.memory.skill_preview_min_score == 0.0
        assert memory.main_agent.max_turns == -1
        assert memory.main_agent.llm.thinking_mode == "enabled"
    finally:
        if previous is None:
            os.environ.pop("ASHARE_TRADER_RUN_ID", None)
        else:
            os.environ["ASHARE_TRADER_RUN_ID"] = previous


def main() -> None:
    generate_tasks()
    tasks = _load_tasks()
    test_momentum_anchor_skill(tasks)
    test_tasks_and_panel(tasks)
    test_parser_and_finance(tasks)
    test_episode_memory_and_factors(tasks)
    test_dispatch_and_configs()
    print(
        "[OK] 12 unified trader tasks; 250-session 16-stock PIT panel; "
        "momentum-anchor skill injection; strict allocations/costs; "
        "matured isolated memory; serial configs"
    )


if __name__ == "__main__":
    main()
