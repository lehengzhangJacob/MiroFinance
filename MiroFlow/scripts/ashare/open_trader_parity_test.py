#!/usr/bin/env python3
"""Offline parity checks for the paired stocks-only open-market benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


FLOW_ROOT = Path(__file__).resolve().parents[2]
AGENT_ROOT = FLOW_ROOT.parent
MEM_ROOT = AGENT_ROOT / "MiroMemSkill"
DEFAULT_SNAPSHOT = (
    AGENT_ROOT / "shared" / "ashare_open_stocks_glm52_20260714"
)
sys.path.insert(0, str(FLOW_ROOT))

from scripts.ashare.eval_open_trader import (  # noqa: E402
    artifact_path,
    load_manifest,
    load_reference_evaluator,
    sha256,
)
from src.utils.ashare_trader import parse_portfolio_weights  # noqa: E402


MODEL_FIELDS = (
    "model_name",
    "thinking_mode",
    "temperature",
    "top_p",
    "min_p",
    "top_k",
    "max_tokens",
    "keep_tool_result",
    "oai_tool_thinking",
)


def load_tasks(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def compose_agent(root: Path, config_name: str) -> Any:
    with initialize_config_dir(version_base=None, config_dir=str(root / "config")):
        config = compose(config_name=config_name)
    OmegaConf.resolve(config)
    return config


def memskill_validation(answer: str, metadata: dict[str, Any]) -> bool:
    program = """
import json,sys
from src.utils.ashare_trader import validate_portfolio_answer
payload=json.load(sys.stdin)
print("1" if validate_portfolio_answer(payload["answer"],payload["metadata"]).ok else "0")
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(MEM_ROOT)
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=MEM_ROOT,
        env=env,
        input=json.dumps({"answer": answer, "metadata": metadata}),
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout.strip() == "1"


def assert_config_parity() -> None:
    os.environ.setdefault("ASHARE_TRADER_RUN_ID", "parity-open-trader")
    flow = compose_agent(FLOW_ROOT, "agent_ashare_trader_open_glm")
    mem = compose_agent(MEM_ROOT, "agent_ashare_trader_open_memskill_glm")
    assert flow.benchmark.name == mem.benchmark.name == "ashare-trader-open"
    assert flow.benchmark.execution.max_concurrent == 1
    assert mem.benchmark.execution.max_concurrent == 1
    assert flow.benchmark.execution.pass_at_k == mem.benchmark.execution.pass_at_k == 1
    assert flow.benchmark.execution.task_order == mem.benchmark.execution.task_order
    assert flow.main_agent.max_turns == mem.main_agent.max_turns == 24
    assert (
        flow.main_agent.max_tool_calls_per_turn
        == mem.main_agent.max_tool_calls_per_turn
        == 16
    )
    for field in MODEL_FIELDS:
        assert flow.main_agent.llm.get(field) == mem.main_agent.llm.get(field), field
    assert list(flow.main_agent.tool_config) == list(mem.main_agent.tool_config) == [
        "tool-ashare-open"
    ]
    assert not flow.get("memory")
    assert mem.memory.enabled is True
    assert mem.memory.skill_enabled is True
    assert mem.memory.inject_enabled is False
    assert mem.memory.inject_top_k == 0
    assert mem.memory.reflection_mode == "trader"
    assert mem.memory.namespace == "ashare_trader_open_parity-open-trader"
    assert (
        flow.main_agent.generic_task_guidance_enabled
        == mem.main_agent.generic_task_guidance_enabled
        is False
    )
    assert (
        flow.main_agent.output_process.reuse_terminal_response
        == mem.main_agent.output_process.reuse_terminal_response
        is True
    )


def assert_task_and_validator_parity(tasks: list[dict[str, Any]]) -> None:
    assert len(tasks) == 12
    pool_sizes = [len(task["metadata"]["stock_pool"]) for task in tasks]
    assert min(pool_sizes) >= 4_800 and max(pool_sizes) < 5_000
    assert [task["metadata"]["entry_date"] for task in tasks] == sorted(
        task["metadata"]["entry_date"] for task in tasks
    )
    first = tasks[0]
    pool = first["metadata"]["stock_pool"]
    answer = "\\boxed{" + ",".join(f"{code}:0.25" for code in pool[:4])
    answer += ",CASH:0.00}"
    assert parse_portfolio_weights(answer, pool).ok
    assert memskill_validation(answer, first["metadata"])
    outside = "\\boxed{999999.SH:0.25,CASH:0.75}"
    assert not parse_portfolio_weights(outside, pool).ok
    assert not memskill_validation(outside, first["metadata"])


def assert_database(path: Path, expected_sha: str) -> None:
    assert sha256(path) == expected_sha
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert conn.execute("SELECT COUNT(*) FROM market_daily").fetchone()[0] > 2_600_000
        assert conn.execute(
            "SELECT COUNT(DISTINCT ts_code) FROM market_daily"
        ).fetchone()[0] > 5_000
    finally:
        conn.close()


def assert_evaluator_determinism(
    evaluator_path: Path,
    evaluator_sha: str,
    tasks_path: Path,
    database_path: Path,
) -> None:
    reference = load_reference_evaluator(evaluator_path, evaluator_sha)
    reference.TASKS = tasks_path
    reference.DB_PATH = database_path
    tasks = reference.load_tasks()
    conn = sqlite3.connect(f"file:{database_path}?mode=ro", uri=True)
    try:
        market = reference.Market(conn)
        allocations = {
            task["metadata"]["as_of"]: {
                **{
                    code: 0.25
                    for code in task["metadata"]["stock_pool"][:4]
                },
                "CASH": 0.0,
            }
            for task in tasks
        }
        first = reference.replay(market, tasks, allocations)
        second = reference.replay(market, tasks, allocations)
        assert first == second
    finally:
        conn.close()


def assert_open_episode_return_parity(
    reference: Any,
    tasks: list[dict[str, Any]],
    database_path: Path,
) -> None:
    first = tasks[0]["metadata"]
    codes = first["stock_pool"][:4]
    program = """
import json,sys
from common_benchmark import _open_market_holding_returns
payload=json.load(sys.stdin)
print(json.dumps(_open_market_holding_returns(
    payload["database"],
    payload["codes"],
    payload["entry"],
    payload["exit"],
), sort_keys=True))
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(MEM_ROOT)
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=MEM_ROOT,
        env=env,
        input=json.dumps(
            {
                "database": str(database_path),
                "codes": codes,
                "entry": first["entry_date"],
                "exit": first["exit_date"],
            }
        ),
        text=True,
        capture_output=True,
        check=True,
    )
    actual = json.loads(result.stdout)
    conn = sqlite3.connect(f"file:{database_path}?mode=ro", uri=True)
    try:
        market = reference.Market(conn)
        expected = {
            code: market.asset_window_return(
                code,
                first["entry_date"],
                first["exit_date"],
            )
            for code in codes
        }
    finally:
        conn.close()
    assert actual == expected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    args = parser.parse_args()
    snapshot = args.snapshot.resolve()
    manifest = load_manifest(snapshot)
    tasks_path = artifact_path(snapshot, manifest, "tasks")
    database_path = artifact_path(snapshot, manifest, "database")
    server_path = artifact_path(snapshot, manifest, "server")
    evaluator_path = artifact_path(snapshot, manifest, "evaluator")

    current_server = (
        MEM_ROOT / "src" / "tool" / "mcp_servers" / "ashare_open_mcp_server.py"
    )
    assert sha256(current_server) == manifest["artifacts"]["server"]["sha256"]
    assert sha256(server_path) == manifest["artifacts"]["server"]["sha256"]
    assert_config_parity()
    tasks = load_tasks(tasks_path)
    assert_task_and_validator_parity(tasks)
    assert_database(
        database_path,
        manifest["artifacts"]["database"]["sha256"],
    )
    assert_evaluator_determinism(
        evaluator_path,
        manifest["artifacts"]["evaluator"]["sha256"],
        tasks_path,
        database_path,
    )
    reference = load_reference_evaluator(
        evaluator_path,
        manifest["artifacts"]["evaluator"]["sha256"],
    )
    assert_open_episode_return_parity(reference, tasks, database_path)

    manifest_digest = hashlib.sha256(
        (snapshot / "manifest.json").read_bytes()
    ).hexdigest()
    print(
        "[OK] open-market parity: 12 shared stocks-only tasks, matching "
        "GLM-5.2/tool settings, MemSkill memory profile, validators, "
        "episode returns, database and deterministic evaluator"
    )
    print(f"snapshot_manifest_sha256={manifest_digest}")


if __name__ == "__main__":
    main()
