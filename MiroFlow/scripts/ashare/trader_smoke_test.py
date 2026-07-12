#!/usr/bin/env python3
"""Offline smoke tests for the unified A-share trader benchmark (MiroFlow)."""

from __future__ import annotations

import asyncio
import io
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.ashare.eval_trader import evaluate_allocations  # noqa: E402
from scripts.ashare.gen_trader_tasks import main as generate_tasks  # noqa: E402
from src.tool.mcp_servers.ashare_mcp_server import (  # noqa: E402
    ashare_trader_universe_context,
)
from src.utils.ashare_trader import (  # noqa: E402
    PortfolioParseResult,
    evaluate_portfolio_month,
    parse_portfolio_weights,
)
from src.utils.ashare_trader_features import compute_trader_feature_rows  # noqa: E402
from utils.eval_utils import verify_answer_for_datasets  # noqa: E402


def _load_tasks() -> list[dict[str, Any]]:
    path = ROOT / "data" / "ashare_trader" / "standardized_data.jsonl"
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_tasks_and_panel(tasks: list[dict[str, Any]]) -> None:
    assert len(tasks) == 12
    assert [task["metadata"]["entry_date"] for task in tasks] == sorted(
        task["metadata"]["entry_date"] for task in tasks
    )
    assert all(
        previous["metadata"]["exit_date"] <= current["metadata"]["entry_date"]
        for previous, current in zip(tasks, tasks[1:])
    ), "one capital account cannot fund overlapping trader windows"
    for task in tasks:
        metadata = task["metadata"]
        pool = metadata["stock_pool"]
        assert metadata["task_type"] == "portfolio_allocation"
        assert len(pool) == len(set(pool)) == 16
        assert set(metadata["stock_returns"]) == set(pool)
        assert metadata["exit_date"] > metadata["entry_date"]

    first = tasks[0]
    panel = ashare_trader_universe_context.fn(first["metadata"]["entry_date"], 250)
    frame = pd.read_csv(io.StringIO("\n".join(panel.splitlines()[3:])))
    assert len(frame) == 16
    assert set(frame["ts_code"]) == set(first["metadata"]["stock_pool"])
    assert "ground_truth" not in panel
    assert "excess_return" not in panel

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
    cut = daily[daily["trade_date"] <= first["metadata"]["entry_date"]]["close_qfq"].astype(
        float
    )
    expected_ret20 = (cut.iloc[-1] / cut.iloc[-21] - 1.0) * 100
    assert abs(float(rows[0]["ret20"]) - round(expected_ret20, 1)) < 1e-12


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

    task_map = {first["task_id"]: first}
    invalid = PortfolioParseResult(
        {code: 0.0 for code in pool}, 1.0, False, "synthetic failure"
    )
    summary, _monthly = evaluate_allocations(
        "invalid",
        task_map,
        {first["task_id"]: invalid},
        initial_capital=1_000_000.0,
        open_cost=0.0005,
        close_cost=0.0015,
        min_cost=5.0,
    )
    assert summary["parsed"] == 0

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


def test_configs() -> None:
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
    OmegaConf.resolve(clean)


def main() -> None:
    generate_tasks()
    tasks = _load_tasks()
    test_tasks_and_panel(tasks)
    test_parser_and_finance(tasks)
    test_configs()
    print(
        "[OK] MiroFlow ashare-trader: 12 monthly tasks, 250-session panel, "
        "strict allocations/costs, serial monthly config"
    )


if __name__ == "__main__":
    main()
