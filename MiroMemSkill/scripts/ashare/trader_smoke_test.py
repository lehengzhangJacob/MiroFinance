#!/usr/bin/env python3
"""Offline smoke tests for the unified A-share trader benchmark."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from omegaconf.errors import InterpolationResolutionError

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common_benchmark import (  # noqa: E402
    BenchmarkEvaluator,
    BenchmarkTask,
    TaskStatus,
    _render_satellite_candidate_block,
    _SATELLITE_DECISION_FACTOR_FIELDS,
    _build_core_satellite_attribution,
)
from scripts.ashare.eval_trader import evaluate_allocations  # noqa: E402
from scripts.ashare.gen_trader_tasks import main as generate_tasks  # noqa: E402
from src.core.orchestrator import (  # noqa: E402
    MAX_TRADER_PORTFOLIO_REPAIRS,
    REQUIRED_TRADER_ANCHOR_TOOLS,
    _ashare_trader_portfolio_error,
    _ashare_trader_terminal_error,
    _canonicalize_core_satellite_response,
    _is_rate_limit_error,
    _validate_ashare_trader_tool_call,
)
from src.memory.context import build_memory_context_block, compact_task_query  # noqa: E402
from src.memory.memory import Mem0Memory  # noqa: E402
from src.memory.rank_reflection import factor_reliability  # noqa: E402
from src.memory.skills import SkillLibrary  # noqa: E402
from src.memory.store_factory import create_memory_store  # noqa: E402
from src.memory.vector_store import MemoryRecord  # noqa: E402
from src.tool.mcp_servers.ashare_mcp_server import (  # noqa: E402
    ashare_excess_satellite_candidates,
    ashare_market_breadth,
    ashare_momentum_baseline,
    ashare_trader_universe_context,
)
from src.utils.ashare_anchor import (  # noqa: E402
    CORE_SATELLITE_SLEEVES,
    AnchorPolicy,
    assemble_core_satellite_allocation,
    build_anchor_snapshot,
    classify_anchor_risk_signals,
    validate_anchor_allocation,
    validate_core_satellite_allocation,
)
from src.utils.ashare_market_breadth import (  # noqa: E402
    clear_market_breadth_cache,
    compute_market_breadth_regime,
)
from src.utils.ashare_satellite import load_excess_signal_candidates  # noqa: E402
from src.utils.ashare_trader import (  # noqa: E402
    PortfolioParseResult,
    canonicalize_core_satellite_answer,
    evaluate_portfolio_month,
    parse_portfolio_weights,
    validate_portfolio_answer,
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


def _core_satellite_fixture(
    regime: str,
) -> tuple[AnchorPolicy, dict[str, Any], list[str], list[str], str]:
    top4 = [f"60000{number}.SH" for number in range(1, 5)]
    eligible = [f"6000{number:02d}.SH" for number in range(5, 11)]
    ineligible = "600011.SH"
    pool = [*top4, *eligible, ineligible]
    candidate_rows = [
        {
            "signal_date": "20250102",
            "ts_code": code,
            "score": round(1.0 - rank / 10.0, 4),
            "rank": rank,
            "n_stocks": len(pool),
            "train_end": "20241129",
            "target": "excess_vs_000300.SH",
        }
        for rank, code in enumerate(eligible, start=1)
    ]
    policy = AnchorPolicy(enabled=True, mode="core_satellite")
    snapshot = {
        "version": 2,
        "as_of": "20250110",
        "top4": top4,
        "anchor_weights": {code: 0.25 for code in top4},
        "anchor_cash": 0.0,
        "market_breadth": {"regime": regime},
        "ranked_prediction_candidates": candidate_rows,
    }
    metadata = {
        "task_type": "portfolio_allocation",
        "entry_date": "20250110",
        "stock_pool": pool,
        "max_stock_weight": 0.25,
        "anchor_policy": policy.to_dict(),
        "anchor_snapshot": snapshot,
        "satellite_candidates": candidate_rows,
        "satellite_signal_date": "20250102",
        "satellite_target": "excess_vs_000300.SH",
    }
    return policy, metadata, top4, eligible, ineligible


def _assert_core_satellite_rejected(
    weights: dict[str, float],
    cash: float,
    *,
    metadata: dict[str, Any],
    expected_error: str,
) -> None:
    result = validate_core_satellite_allocation(
        weights,
        cash,
        snapshot=metadata["anchor_snapshot"],
        policy=AnchorPolicy.from_mapping(metadata["anchor_policy"]),
    )
    assert not result.ok
    assert expected_error in result.error, result.error


def test_core_satellite_policy_matrix() -> None:
    expected = {
        "risk_on": {
            "core_stock_weight": 0.20,
            "core_total_weight": 0.80,
            "satellite_stock_weight": 0.10,
            "satellite_count": 2,
            "satellite_total_weight": 0.20,
            "cash_weight": 0.0,
        },
        "neutral": {
            "core_stock_weight": 0.225,
            "core_total_weight": 0.90,
            "satellite_stock_weight": 0.10,
            "satellite_count": 1,
            "satellite_total_weight": 0.10,
            "cash_weight": 0.0,
        },
        "defensive": {
            "core_stock_weight": 0.225,
            "core_total_weight": 0.90,
            "satellite_stock_weight": 0.05,
            "satellite_count": 1,
            "satellite_total_weight": 0.05,
            "cash_weight": 0.05,
        },
    }
    assert set(CORE_SATELLITE_SLEEVES) == set(expected)
    for regime, sleeve_expected in expected.items():
        sleeve = CORE_SATELLITE_SLEEVES[regime]
        assert {
            key: sleeve.to_dict()[key] for key in sleeve_expected
        } == sleeve_expected
        policy, metadata, top4, eligible, _ = _core_satellite_fixture(regime)
        allocation = assemble_core_satellite_allocation(
            metadata["anchor_snapshot"]
        )
        assert allocation["top4"] == top4
        assert allocation["satellite_codes"] == eligible[
            : sleeve_expected["satellite_count"]
        ]
        assert set(allocation["top4"]).isdisjoint(allocation["satellite_codes"])
        assert set(allocation["core_weights"].values()) == {
            sleeve_expected["core_stock_weight"]
        }
        assert set(allocation["satellite_weights"].values()) == {
            sleeve_expected["satellite_stock_weight"]
        }
        assert allocation["cash"] == sleeve_expected["cash_weight"]
        assert abs(
            sum(allocation["weights"].values()) + allocation["cash"] - 1.0
        ) < 1e-12
        valid = validate_core_satellite_allocation(
            allocation["weights"],
            allocation["cash"],
            snapshot=metadata["anchor_snapshot"],
            policy=policy,
        )
        assert valid.ok, valid.error
        assert valid.metrics["top4_holding_count"] == 4
        assert valid.metrics["satellite_count"] == sleeve_expected["satellite_count"]
        assert valid.metrics["expected_core_stock_weight"] == sleeve_expected[
            "core_stock_weight"
        ]
        assert valid.metrics["expected_satellite_stock_weight"] == sleeve_expected[
            "satellite_stock_weight"
        ]

    _, metadata, top4, eligible, _ = _core_satellite_fixture("risk_on")
    canonical = assemble_core_satellite_allocation(metadata["anchor_snapshot"])
    weights = dict(canonical["weights"])

    missing_core = dict(weights)
    missing_core.pop(top4[0])
    _assert_core_satellite_rejected(
        missing_core,
        canonical["cash"],
        metadata=metadata,
        expected_error="all four momentum leaders are mandatory",
    )

    wrong_core = dict(weights)
    wrong_core[top4[0]] -= 0.01
    wrong_core[top4[1]] += 0.01
    _assert_core_satellite_rejected(
        wrong_core,
        canonical["cash"],
        metadata=metadata,
        expected_error="core weight",
    )

    try:
        assemble_core_satellite_allocation(
            metadata["anchor_snapshot"],
            selected_satellites=[top4[0], eligible[0]],
        )
        raise AssertionError("top4 stock was accepted as a satellite")
    except ValueError as error:
        assert "satellites must be non-top4 stocks" in str(error)

    too_many = dict(weights)
    too_many[eligible[2]] = 0.10
    _assert_core_satellite_rejected(
        too_many,
        canonical["cash"],
        metadata=metadata,
        expected_error="requires exactly 2 satellites; held 3",
    )

    too_few = dict(weights)
    too_few.pop(eligible[1])
    _assert_core_satellite_rejected(
        too_few,
        canonical["cash"] + 0.10,
        metadata=metadata,
        expected_error="requires exactly 2 satellites; held 1",
    )

    wrong_satellite = dict(weights)
    wrong_satellite[eligible[0]] -= 0.01
    wrong_satellite[eligible[1]] += 0.01
    _assert_core_satellite_rejected(
        wrong_satellite,
        canonical["cash"],
        metadata=metadata,
        expected_error="satellite weight",
    )

    _assert_core_satellite_rejected(
        weights,
        0.01,
        metadata=metadata,
        expected_error="cash weight",
    )
    _assert_core_satellite_rejected(
        weights,
        0.02,
        metadata=metadata,
        expected_error="must sum to 100%",
    )

    duplicate = dict(weights)
    duplicate[top4[0].lower()] = 0.0
    _assert_core_satellite_rejected(
        duplicate,
        canonical["cash"],
        metadata=metadata,
        expected_error="duplicate normalized code",
    )

    outside = dict(weights)
    outside.pop(eligible[1])
    outside["999999.SH"] = 0.10
    _assert_core_satellite_rejected(
        outside,
        canonical["cash"],
        metadata=metadata,
        expected_error="not eligible prediction candidates",
    )
    outside_answer = "\\boxed{" + ",".join(
        [
            *(f"{code}:{weight}" for code, weight in outside.items()),
            "CASH:0.00",
        ]
    ) + "}"
    parsed_outside = validate_portfolio_answer(outside_answer, metadata)
    assert not parsed_outside.ok
    assert "codes outside pool" in parsed_outside.error


def test_core_satellite_intent_canonicalization() -> None:
    _, metadata, top4, eligible, ineligible = _core_satellite_fixture("risk_on")
    selected = eligible[-2:]
    malformed = (
        "Use the eligible alternatives despite malformed weights. "
        f"\\boxed{{{selected[1]}:999,{selected[0]}:not-a-number,CASH:-7}}"
    )
    accepted = canonicalize_core_satellite_answer(malformed, metadata)
    assert accepted is not None
    assert accepted.selection_source == "agent"
    assert not accepted.selection_fallback
    assert accepted.selected_codes == tuple(selected)
    assert accepted.deterministic_codes == tuple(eligible[:2])
    assert accepted.diagnostic == "accepted 2 eligible satellite code(s) from answer"
    assert set(accepted.weights) == set(top4) | set(selected)
    assert set(accepted.weights[code] for code in top4) == {0.20}
    assert set(accepted.weights[code] for code in selected) == {0.10}
    assert accepted.cash == 0.0

    canonical_text, audit = _canonicalize_core_satellite_response(
        malformed,
        metadata,
    )
    assert canonical_text.endswith(audit.canonical_boxed_answer)
    assert metadata["satellite_selection"] == {
        "selected_codes": selected,
        "deterministic_codes": eligible[:2],
        "selection_fallback": False,
        "source": "agent",
        "diagnostic": "accepted 2 eligible satellite code(s) from answer",
        "candidate_signal_date": "20250102",
        "target": "excess_vs_000300.SH",
    }
    judge = asyncio.run(
        verify_answer_for_datasets(
            None,  # type: ignore[arg-type]
            "ashare-trader",
            "synthetic core-satellite task",
            "",
            audit.canonical_boxed_answer,
            metadata,
        )
    )
    assert judge == "CORRECT"

    fallback_cases = {
        "missing": "",
        "too_many": ",".join(eligible[:3]),
        "ineligible": ",".join([*selected, ineligible]),
        "outside_pool": ",".join([*selected, "999999.SH"]),
    }
    for case, answer in fallback_cases.items():
        fallback = canonicalize_core_satellite_answer(answer, metadata)
        assert fallback is not None, case
        assert fallback.selection_fallback, case
        assert fallback.selection_source == "deterministic_fallback", case
        assert fallback.selected_codes == tuple(eligible[:2]), case
        assert fallback.deterministic_codes == tuple(eligible[:2]), case
        assert fallback.cash < 1.0
        assert abs(sum(fallback.weights.values()) + fallback.cash - 1.0) < 1e-12
        assert "deterministic fallback was used" in fallback.diagnostic

    no_snapshot = dict(metadata)
    no_snapshot.pop("anchor_snapshot")
    try:
        canonicalize_core_satellite_answer("", no_snapshot)
        raise AssertionError("missing anchor snapshot did not fail closed")
    except ValueError as error:
        assert "non-empty anchor_snapshot" in str(error)

    no_snapshot_candidates = dict(metadata)
    no_snapshot_candidates["anchor_snapshot"] = {
        key: value
        for key, value in metadata["anchor_snapshot"].items()
        if key != "ranked_prediction_candidates"
    }
    try:
        canonicalize_core_satellite_answer("", no_snapshot_candidates)
        raise AssertionError("missing snapshot candidates did not fail closed")
    except ValueError as error:
        assert "snapshot/candidates cannot form an allocation" in str(error)

    no_structured_candidates = dict(metadata)
    no_structured_candidates.pop("satellite_candidates")
    try:
        _canonicalize_core_satellite_response("", no_structured_candidates)
        raise AssertionError("missing structured candidates did not fail closed")
    except ValueError as error:
        assert "structured satellite_candidates" in str(error)

    deterministic = canonicalize_core_satellite_answer("", metadata)
    assert deterministic is not None
    assert deterministic.selection_fallback
    assert deterministic.cash != 1.0
    assert sum(deterministic.weights.values()) > 0.0
    assert _ashare_trader_portfolio_error("\\boxed{CASH:1.00}", metadata)

    assert "mandatory hard-anchor tools" in _ashare_trader_terminal_error(
        deterministic.canonical_boxed_answer,
        metadata,
        set(),
    )
    assert "ashare_market_breadth" in _ashare_trader_terminal_error(
        deterministic.canonical_boxed_answer,
        metadata,
        {"ashare_momentum_baseline"},
    )
    assert (
        _ashare_trader_terminal_error(
            deterministic.canonical_boxed_answer,
            metadata,
            set(REQUIRED_TRADER_ANCHOR_TOOLS),
        )
        == ""
    )


def test_satellite_signal_pit_and_manifest(tasks: list[dict[str, Any]]) -> None:
    with tempfile.TemporaryDirectory() as directory:
        data_dir = Path(directory)
        signal = data_dir / "qlib_excess_signal.csv"
        pd.DataFrame(
            [
                {
                    "signal_date": "20241202",
                    "ts_code": "600009.SH",
                    "score": 0.1,
                    "rank": 1,
                    "n_stocks": 1,
                    "train_end": "20241031",
                    "target": "excess_vs_000300.SH",
                },
                {
                    "signal_date": "20250105",
                    "ts_code": "600003.SH",
                    "score": 0.99,
                    "rank": 1,
                    "n_stocks": 4,
                    "train_end": "20241129",
                    "target": "excess_vs_000300.SH",
                },
                {
                    "signal_date": "20250105",
                    "ts_code": "600002.SH",
                    "score": 0.8,
                    "rank": 2,
                    "n_stocks": 4,
                    "train_end": "20241129",
                    "target": "excess_vs_000300.SH",
                },
                {
                    "signal_date": "20250105",
                    "ts_code": "600001.SH",
                    "score": 0.8,
                    "rank": 3,
                    "n_stocks": 4,
                    "train_end": "20241129",
                    "target": "excess_vs_000300.SH",
                },
                {
                    "signal_date": "20250105",
                    "ts_code": "600004.SH",
                    "score": 0.7,
                    "rank": 4,
                    "n_stocks": 4,
                    "train_end": "20241129",
                    "target": "excess_vs_000300.SH",
                },
                {
                    "signal_date": "20250120",
                    "ts_code": "600008.SH",
                    "score": 999.0,
                    "rank": 1,
                    "n_stocks": 1,
                    "train_end": "20241213",
                    "target": "excess_vs_000300.SH",
                },
            ]
        ).to_csv(signal, index=False)
        digest = hashlib.sha256(signal.read_bytes()).hexdigest()
        (data_dir / "qlib_excess_signal_manifest.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "file": signal.name,
                    "sha256": digest,
                }
            ),
            encoding="utf-8",
        )
        rows = load_excess_signal_candidates(
            "2025-01-10",
            ["600003.SH"],
            data_dir=data_dir,
        )
        assert [row["ts_code"] for row in rows] == [
            "600001.SH",
            "600002.SH",
            "600004.SH",
        ]
        assert {row["signal_date"] for row in rows} == {"20250105"}
        assert all(row["signal_date"] <= "20250110" for row in rows)
        assert "600003.SH" not in {row["ts_code"] for row in rows}
        assert "600008.SH" not in {row["ts_code"] for row in rows}
        assert (
            load_excess_signal_candidates(
                "20250110",
                [],
                data_dir=data_dir,
                exact_date=True,
            )
            == []
        )

        with signal.open("a", encoding="utf-8") as handle:
            handle.write("\n")
        try:
            load_excess_signal_candidates(
                "20250110",
                [],
                data_dir=data_dir,
            )
            raise AssertionError("tampered satellite signal passed manifest validation")
        except ValueError as error:
            assert "checksum mismatch" in str(error)

    first = tasks[0]
    rendered = ashare_excess_satellite_candidates.fn(
        first["metadata"]["entry_date"]
    )
    assert "不包含未来或已实现标签" in rendered
    candidate_frame = pd.read_csv(
        io.StringIO("\n".join(rendered.splitlines()[3:]))
    )
    assert list(candidate_frame.columns) == [
        "signal_date",
        "ts_code",
        "score",
        "rank",
        "n_stocks",
        "train_end",
        "target",
    ]
    assert all(
        str(value) <= first["metadata"]["entry_date"]
        for value in candidate_frame["signal_date"]
    )
    assert not {
        "label",
        "LABEL0",
        "ground_truth",
        "realized_return",
        "stock_return",
        "excess_return",
    }.intersection(candidate_frame.columns)


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
        assert "20 日相对动量 top4 硬锚" in block
        assert "ashare_momentum_baseline" in block
        assert "ashare_market_breadth" in block
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
        assert "硬锚实验必须先调用 ashare_momentum_baseline" in task["task_question"]
        assert "Qlib 不能单独否决" in task["task_question"]
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


def test_hard_anchor_and_breadth(tasks: list[dict[str, Any]]) -> None:
    first = tasks[0]
    metadata = first["metadata"]
    pool = metadata["stock_pool"]
    entry = metadata["entry_date"]

    breadth = ashare_market_breadth.fn(entry)
    assert "全A市场广度" in breadth
    assert "risk_on" in breadth or "neutral" in breadth or "defensive" in breadth
    regime = compute_market_breadth_regime(entry, data_dir=ROOT / "data" / "ashare")
    assert regime["regime"] in {"risk_on", "neutral", "defensive"}

    policy60 = AnchorPolicy(enabled=True, min_top4_weight=0.60)
    policy75 = AnchorPolicy(enabled=True, min_top4_weight=0.75)
    snapshot = build_anchor_snapshot(
        entry,
        metadata["stock_info"],
        data_dir=ROOT / "data" / "ashare",
        policy=policy60,
    )
    top4 = snapshot["top4"]
    assert len(top4) == 4
    non_top4 = [code for code in pool if code not in top4]
    dual_snapshot = {
        **snapshot,
        "risk_signals": {
            code: {
                "active": ["trend_break", "valuation_overheat"],
                "count": 2,
                "non_qlib_count": 2,
            }
            for code in top4
        },
    }
    qlib_only_snapshot = {
        **snapshot,
        "risk_signals": {
            code: {
                "active": ["qlib_conflict"],
                "count": 1,
                "non_qlib_count": 0,
            }
            for code in top4
        },
    }
    no_signal_snapshot = {
        **snapshot,
        "risk_signals": {
            code: {"active": [], "count": 0, "non_qlib_count": 0}
            for code in top4
        },
    }

    compliant = {code: 0.25 for code in top4}
    compliant["CASH"] = 0.0
    boxed = "\\boxed{" + ",".join(
        f"{code}:{weight:.2f}" for code, weight in compliant.items()
    ) + "}"
    parsed = validate_portfolio_answer(
        boxed,
        {
            **metadata,
            "anchor_policy": policy60.to_dict(),
            "anchor_snapshot": snapshot,
        },
    )
    assert parsed.ok
    mild_underweight = validate_anchor_allocation(
        {
            top4[0]: 0.24,
            top4[1]: 0.25,
            top4[2]: 0.25,
            top4[3]: 0.25,
        },
        0.01,
        snapshot=no_signal_snapshot,
        policy=policy60,
    )
    assert not mild_underweight.ok

    low_exposure = {code: 0.0 for code in pool}
    low_exposure[top4[0]] = 0.25
    low_exposure[top4[1]] = 0.25
    low_exposure["CASH"] = 0.50
    low_boxed = "\\boxed{" + ",".join(
        f"{code}:{weight:.2f}"
        for code, weight in low_exposure.items()
        if code == "CASH" or weight > 0
    ) + "}"
    assert not validate_portfolio_answer(
        low_boxed,
        {
            **metadata,
            "anchor_policy": policy60.to_dict(),
            "anchor_snapshot": snapshot,
        },
    ).ok

    qlib_only = classify_anchor_risk_signals(
        {"ml_rank": 16, "rel5": 0.05, "rel60": 0.10, "ma20_gap": 0.05},
        vol20_q75=30.0,
        pool_size=16,
    )
    assert qlib_only["count"] == 1
    assert qlib_only["non_qlib_count"] == 0
    dual_classified = classify_anchor_risk_signals(
        {"rel5": -2.0, "ma20_gap": -1.0, "pe_pct250": 95.0},
        vol20_q75=30.0,
        pool_size=16,
    )
    assert set(dual_classified["active"]) == {
        "trend_break",
        "valuation_overheat",
    }

    exact60 = validate_anchor_allocation(
        {top4[0]: 0.20, top4[1]: 0.20, top4[2]: 0.20},
        0.40,
        snapshot=dual_snapshot,
        policy=policy60,
    )
    assert exact60.ok
    below60 = validate_anchor_allocation(
        {top4[0]: 0.20, top4[1]: 0.20, top4[2]: 0.19},
        0.41,
        snapshot=dual_snapshot,
        policy=policy60,
    )
    assert not below60.ok
    for market_regime in ("risk_on", "neutral", "defensive"):
        regime_snapshot = {
            **dual_snapshot,
            "market_breadth": {"regime": market_regime},
        }
        assert not validate_anchor_allocation(
            {top4[0]: 0.20, top4[1]: 0.20, top4[2]: 0.19},
            0.41,
            snapshot=regime_snapshot,
            policy=policy60,
        ).ok

    exact75 = validate_anchor_allocation(
        {top4[0]: 0.25, top4[1]: 0.25, top4[2]: 0.25},
        0.25,
        snapshot=dual_snapshot,
        policy=policy75,
    )
    assert exact75.ok
    below75 = validate_anchor_allocation(
        {top4[0]: 0.25, top4[1]: 0.25, top4[2]: 0.24},
        0.26,
        snapshot=dual_snapshot,
        policy=policy75,
    )
    assert not below75.ok

    qlib_veto = validate_anchor_allocation(
        {top4[0]: 0.20, top4[1]: 0.20, top4[2]: 0.20},
        0.40,
        snapshot=qlib_only_snapshot,
        policy=policy60,
    )
    assert not qlib_veto.ok
    assert "Qlib cannot stand alone" in qlib_veto.error

    dual_signal_veto = validate_anchor_allocation(
        {top4[0]: 0.20, top4[1]: 0.20, top4[2]: 0.20},
        0.40,
        snapshot=dual_snapshot,
        policy=policy60,
    )
    assert dual_signal_veto.ok

    one_replacement = validate_anchor_allocation(
        {
            top4[0]: 0.20,
            top4[1]: 0.20,
            top4[2]: 0.20,
            non_top4[0]: 0.20,
        },
        0.20,
        snapshot=dual_snapshot,
        policy=policy60,
    )
    assert one_replacement.ok
    two_replacements = validate_anchor_allocation(
        {
            top4[0]: 0.20,
            top4[1]: 0.20,
            top4[2]: 0.20,
            non_top4[0]: 0.10,
            non_top4[1]: 0.10,
        },
        0.20,
        snapshot=dual_snapshot,
        policy=policy60,
    )
    assert not two_replacements.ok
    assert "at most 1 replacement" in two_replacements.error

    clear_market_breadth_cache()
    before = compute_market_breadth_regime(entry, data_dir=ROOT / "data" / "ashare")
    after = compute_market_breadth_regime(entry, data_dir=ROOT / "data" / "ashare")
    assert before == after

    with tempfile.TemporaryDirectory() as directory:
        data_dir = Path(directory)
        breadth_frame = pd.read_csv(
            ROOT / "data" / "ashare" / "market_breadth_daily.csv",
            dtype={"trade_date": str},
        )
        index_frame = pd.read_csv(
            ROOT / "data" / "ashare" / "index_000300.SH.csv",
            dtype={"trade_date": str},
        )
        breadth_frame.to_csv(data_dir / "market_breadth_daily.csv", index=False)
        index_frame.to_csv(data_dir / "index_000300.SH.csv", index=False)
        clear_market_breadth_cache()
        historical = compute_market_breadth_regime(entry, data_dir=data_dir)

        future = breadth_frame["trade_date"] > entry
        breadth_frame.loc[
            future,
            [
                "adv_ratio_1d",
                "adv_ratio_5d",
                "above_ma20_ratio",
                "above_ma60_ratio",
                "positive_ret20_ratio",
            ],
        ] = 1.0
        index_frame.loc[index_frame["trade_date"] > entry, "close"] = 1e12
        breadth_frame.to_csv(data_dir / "market_breadth_daily.csv", index=False)
        index_frame.to_csv(data_dir / "index_000300.SH.csv", index=False)
        clear_market_breadth_cache()
        assert compute_market_breadth_regime(entry, data_dir=data_dir) == historical

    flow_root = ROOT.parent / "MiroFlow"
    local_breadth = ROOT / "data" / "ashare" / "market_breadth_daily.csv"
    flow_breadth = flow_root / "data" / "ashare" / "market_breadth_daily.csv"
    assert local_breadth.read_bytes() == flow_breadth.read_bytes()
    local_manifest = json.loads(
        (ROOT / "data" / "ashare" / "market_breadth_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    flow_manifest = json.loads(
        (
            flow_root / "data" / "ashare" / "market_breadth_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert local_manifest["sha256"] == flow_manifest["sha256"]
    assert hashlib.sha256(local_breadth.read_bytes()).hexdigest() == (
        local_manifest["sha256"]
    )
    assert not (
        ROOT / "data" / "ashare" / ".market_breadth_checkpoint.json.gz"
    ).exists()

    parity_script = """
import json
from pathlib import Path
from src.utils.ashare_anchor import AnchorPolicy, build_anchor_snapshot

root = Path.cwd()
tasks = [
    json.loads(line)
    for line in (root / "data/ashare_trader/standardized_data.jsonl")
        .read_text(encoding="utf-8").splitlines()
    if line.strip()
]
policy = AnchorPolicy(enabled=True, min_top4_weight=0.60)
results = []
for task in tasks:
    metadata = task["metadata"]
    results.append(build_anchor_snapshot(
        metadata["entry_date"],
        metadata["stock_info"],
        data_dir=root / "data/ashare",
        policy=policy,
    ))
print(json.dumps(results, sort_keys=True))
"""
    remote = subprocess.run(
        [sys.executable, "-c", parity_script],
        cwd=flow_root,
        check=True,
        capture_output=True,
        text=True,
    )
    flow_snapshots = json.loads(remote.stdout.strip().splitlines()[-1])
    local_snapshots = [
        build_anchor_snapshot(
            task["metadata"]["entry_date"],
            task["metadata"]["stock_info"],
            data_dir=ROOT / "data" / "ashare",
            policy=policy60,
        )
        for task in tasks
    ]
    assert flow_snapshots == local_snapshots


def test_trader_guardrails(tasks: list[dict[str, Any]]) -> None:
    assert MAX_TRADER_PORTFOLIO_REPAIRS == 2
    assert REQUIRED_TRADER_ANCHOR_TOOLS == {
        "ashare_momentum_baseline",
        "ashare_market_breadth",
    }
    metadata = tasks[0]["metadata"]
    pool = metadata["stock_pool"]
    cutoff = metadata["entry_date"]
    _validate_ashare_trader_tool_call(
        "ashare_price_history",
        {"ts_code": pool[0], "as_of": cutoff},
        metadata,
    )

    try:
        _validate_ashare_trader_tool_call(
            "ashare_price_history",
            {"ts_code": "600519.SH", "as_of": cutoff},
            metadata,
        )
        raise AssertionError("stock outside the trader pool was accepted")
    except ValueError as error:
        assert "outside the task pool" in str(error)

    try:
        _validate_ashare_trader_tool_call(
            "ashare_index_history",
            {"as_of": "2099-01-01"},
            metadata,
        )
        raise AssertionError("lookahead date was accepted")
    except ValueError as error:
        assert "lookahead" in str(error)

    valid = f"{pool[0]}:0.25,{pool[1]}:0.25,CASH:0.50"
    assert _ashare_trader_portfolio_error(valid, metadata) == ""
    policy = AnchorPolicy(enabled=True, min_top4_weight=0.60)
    anchored_metadata = {
        **metadata,
        "anchor_policy": policy.to_dict(),
        "anchor_snapshot": build_anchor_snapshot(
            cutoff,
            metadata["stock_info"],
            data_dir=ROOT / "data" / "ashare",
            policy=policy,
        ),
    }
    assert _ashare_trader_portfolio_error(valid, anchored_metadata)
    _validate_ashare_trader_tool_call(
        "ashare_momentum_baseline",
        {"as_of": cutoff, "window": 20, "top_k": 4},
        anchored_metadata,
    )
    _validate_ashare_trader_tool_call(
        "ashare_market_breadth",
        {"as_of": cutoff},
        anchored_metadata,
    )
    try:
        _validate_ashare_trader_tool_call(
            "ashare_momentum_baseline",
            {"as_of": cutoff, "window": 60, "top_k": 4},
            anchored_metadata,
        )
        raise AssertionError("wrong hard-anchor momentum window was accepted")
    except ValueError as error:
        assert "window=20 and top_k=4" in str(error)

    top4 = anchored_metadata["anchor_snapshot"]["top4"]
    anchor_answer = ",".join(f"{code}:0.25" for code in top4) + ",CASH:0.00"
    assert "mandatory hard-anchor tools" in _ashare_trader_terminal_error(
        anchor_answer,
        anchored_metadata,
        set(),
    )
    assert (
        _ashare_trader_terminal_error(
            anchor_answer,
            anchored_metadata,
            set(REQUIRED_TRADER_ANCHOR_TOOLS),
        )
        == ""
    )
    assert _ashare_trader_portfolio_error(
        "600519.SH 贵州茅台",
        metadata,
    )
    rate_limit = RuntimeError("Error code: 429 - rate limit reached")
    wrapped = RuntimeError("wrapped request failure")
    wrapped.__cause__ = rate_limit
    assert _is_rate_limit_error(rate_limit)
    assert _is_rate_limit_error(wrapped)
    assert not _is_rate_limit_error(RuntimeError("connection reset"))

    task_data = tasks[0]
    task = BenchmarkTask(
        task_id=task_data["task_id"],
        task_question=task_data["task_question"],
        ground_truth=task_data["ground_truth"],
        file_path=None,
        metadata=anchored_metadata,
    )
    with tempfile.TemporaryDirectory() as directory:
        log_path = Path(directory) / f"task_{task.task_id}_attempt_1.json"
        log_path.write_text(
            json.dumps(
                {
                    "final_boxed_answer": valid,
                    "judge_result": "INCORRECT",
                }
            ),
            encoding="utf-8",
        )
        evaluator = SimpleNamespace(
            output_dir=Path(directory),
            benchmark_name="ashare-trader",
        )
        cached = BenchmarkEvaluator.scan_latest_attempt(
            evaluator,  # type: ignore[arg-type]
            task,
            1,
        )
        assert cached["status"] == TaskStatus.RUN_FAILED
        assert "cached trader allocation rejected" in str(cached["error_message"])


def test_core_satellite_runtime_attachment(
    tasks: list[dict[str, Any]],
) -> None:
    previous = os.environ.get("ASHARE_TRADER_RUN_ID")
    os.environ["ASHARE_TRADER_RUN_ID"] = "smoke-core-runtime"
    try:
        config_dir = str(ROOT / "config")
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            core = compose(
                config_name="agent_ashare_trader_core_satellite_attribution_kimi"
            )
        assert core.benchmark.anchor_policy.enabled is True
        assert core.benchmark.anchor_policy.mode == "core_satellite"
        assert core.benchmark.anchor_policy.candidate_limit == 6
        assert core.benchmark.execution.max_concurrent == 1
        assert core.benchmark.execution.task_order == "monthly"
        assert core.main_agent.llm.thinking_mode == "enabled"
        assert core.main_agent.llm.max_tokens == 16000
        assert core.memory.namespace == (
            "ashare_trader_core_satellite_attribution_smoke-core-runtime"
        )
        assert core.memory.skill_enabled is False
        assert core.memory.skill_top_k == 0

        task_data = tasks[0]
        runtime_task = BenchmarkTask(
            task_id=task_data["task_id"],
            task_question=task_data["task_question"],
            ground_truth=task_data["ground_truth"],
            file_path=None,
            metadata=dict(task_data["metadata"]),
        )
        BenchmarkEvaluator._attach_trader_anchor_policy(
            SimpleNamespace(cfg=core),  # type: ignore[arg-type]
            runtime_task,
        )
    finally:
        if previous is None:
            os.environ.pop("ASHARE_TRADER_RUN_ID", None)
        else:
            os.environ["ASHARE_TRADER_RUN_ID"] = previous

    snapshot = runtime_task.metadata["anchor_snapshot"]
    rows = runtime_task.metadata["satellite_candidates"]
    signal_rows = [
        dict(row)
        for row in load_excess_signal_candidates(
            task_data["metadata"]["entry_date"],
            snapshot["top4"],
            data_dir=ROOT / "data" / "ashare",
        )
        if row["ts_code"] in set(task_data["metadata"]["stock_pool"])
    ][:6]
    signal_fields = {
        "signal_date",
        "ts_code",
        "score",
        "rank",
        "n_stocks",
        "train_end",
        "target",
    }
    assert len(rows) == len(signal_rows) == 6
    assert [row["ts_code"] for row in rows] == [
        row["ts_code"] for row in signal_rows
    ]
    assert [
        {field: row[field] for field in signal_fields} for row in rows
    ] == signal_rows
    feature_rows = compute_trader_feature_rows(
        task_data["metadata"]["entry_date"],
        task_data["metadata"]["stock_info"],
        data_dir=ROOT / "data" / "ashare",
        lookback_days=250,
    )
    factors_by_code = {row["ts_code"]: row for row in feature_rows}
    expected_fields = signal_fields | {
        "decision_as_of",
        *_SATELLITE_DECISION_FACTOR_FIELDS,
    }
    for row in rows:
        assert set(row) == expected_fields
        assert row["decision_as_of"] == task_data["metadata"]["entry_date"]
        assert str(row["signal_date"]) <= row["decision_as_of"]
        assert str(row["train_end"]) <= row["decision_as_of"]
        expected_factors = factors_by_code[row["ts_code"]]
        for factor in _SATELLITE_DECISION_FACTOR_FIELDS:
            expected = expected_factors[factor]
            if expected is None or expected == "":
                expected = "NA"
            assert row[factor] == expected
        financial_date = row["financial_ann_date"]
        assert financial_date == "NA" or str(financial_date) <= row["decision_as_of"]
    forbidden_fields = (
        "ground_truth",
        "ground_truth_rank",
        "realized_return",
        "realized_holding_return",
        "stock_return",
        "excess_return",
        "excess_returns",
        "LABEL0",
    )
    serialized_rows = json.dumps(rows, ensure_ascii=False)
    assert all(field not in serialized_rows for field in forbidden_fields)
    assert snapshot["ranked_prediction_candidates"] == [
        row["ts_code"] for row in rows
    ]
    assert set(snapshot["top4"]).isdisjoint(
        snapshot["eligible_prediction_candidates"]
    )
    assert runtime_task.metadata["satellite_candidate_limit"] == 6
    assert runtime_task.metadata["satellite_signal_date"] == rows[0]["signal_date"]
    assert runtime_task.metadata["satellite_target"] == rows[0]["target"]
    required_count = int(
        snapshot["core_satellite_sleeve"]["satellite_count"]
    )
    assert runtime_task.metadata["satellite_required_count"] == required_count

    marker = "## 超额收益卫星候选（系统点时注入）"
    assert runtime_task.task_question.count(marker) == 1
    candidate_prompt = runtime_task.task_question.split(marker, 1)[1]
    assert "已排除 rel20 top4" in candidate_prompt
    assert f"必须只选 {required_count} 只卫星代码（不多不少）" in candidate_prompt
    assert "固定权重与无效输出回退由系统处理" in candidate_prompt
    assert "不得机械照抄排名" in candidate_prompt
    assert "若无可信反向证据，仍可选择模型排名最前者" in candidate_prompt
    assert "已到期的结构化记忆（若有）" in candidate_prompt
    for position, row in enumerate(rows, start=1):
        assert f"{position}. {row['ts_code']}" in candidate_prompt
        assert f"score={float(row['score']):.8g}" in candidate_prompt
        assert f"signal_rank={int(row['rank'])}" in candidate_prompt
        assert f"signal_date={row['signal_date']}" in candidate_prompt
        assert f"train_end={row['train_end']}" in candidate_prompt
        assert f"target={row['target']}" in candidate_prompt
        assert f"decision_as_of={row['decision_as_of']}" in candidate_prompt
    for factor in _SATELLITE_DECISION_FACTOR_FIELDS:
        assert f"{factor}=" in candidate_prompt
    for forbidden in forbidden_fields:
        assert forbidden not in candidate_prompt
    missing_row = dict(rows[0])
    missing_row["financial_ann_date"] = ""
    missing_row["ml_rank"] = None
    missing_prompt = _render_satellite_candidate_block(
        [missing_row],
        required_count=1,
    )
    assert "financial_ann_date=NA" in missing_prompt
    assert "ml_rank=NA" in missing_prompt

    allocation = assemble_core_satellite_allocation(snapshot)
    boxed = "\\boxed{" + ",".join(
        [
            *(
                f"{code}:{weight}"
                for code, weight in allocation["weights"].items()
            ),
            f"CASH:{allocation['cash']}",
        ]
    ) + "}"
    assert validate_portfolio_answer(boxed, runtime_task.metadata).ok
    fallback = canonicalize_core_satellite_answer("", runtime_task.metadata)
    assert fallback is not None
    assert fallback.canonical_boxed_answer != "\\boxed{CASH:1.00}"
    assert sum(fallback.weights.values()) > 0.0


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
    assert summary["anchor_floor"] == 0.60
    assert summary["anchor_compliance_rate"] == 0.0
    assert "cumulative_deviation_vs_anchor" in summary
    assert set(summary["breadth_regime_performance"]) == {
        "risk_on",
        "neutral",
        "defensive",
    }
    assert monthly[0]["cash"] == 1.0
    assert monthly[0]["ending_capital"] == 1_000_000.0
    assert "top4_overlap_count" in monthly[0]
    assert "deviation_pnl" in monthly[0]

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
        attribution = {
            "anchor_weights": {pool[0]: 0.25},
            "anchor_cash": 0.75,
            "anchor_net_return": 0.008,
            "anchor_active_return": 0.028,
            "deviation_active_return": 0.001,
            "cost_delta": 0.0,
            "cash_delta": 0.0,
            "overlap_count": 1,
            "dropped_from_anchor": [],
            "added_vs_anchor": [],
            "weight_delta_contributions": {},
            "market_regime": "neutral",
        }
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
            anchor_attribution=attribution,
            reasoning="只使用决策日可见信息。",
            parse_ok=True,
        )
        assert operation.startswith("ADD")
        assert "当时的点时理由" not in store.records[0].content
        assert "偏离" in store.records[0].content
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
            anchor_attribution=attribution,
            reasoning="只使用决策日可见信息。",
            parse_ok=True,
        ).startswith("UNCHANGED")
        assert len(store.records) == 1
        assert memory.trader_episode_block("20240728") == ""
        visible = memory.trader_episode_block("20240801")
        assert "2024-07" in visible and "主动收益 +2.90%" in visible
        assert "实际组合相对同期动量锚点累计差 +0.10%" in visible
        assert "不得继承过去的长篇理由" in visible
        assert "只使用决策日可见信息" not in visible

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
        assert "已到期的动量锚点偏离审计" in block

        isolated = Mem0Memory(InMemoryStore(tmp, "trader-b"))
        assert isolated.trader_episode_block("20240801") == ""

        legacy_store = InMemoryStore(tmp, "trader-legacy")
        legacy_store.add(
            "LEGACY_MODEL_REASONING_MUST_NOT_BE_INJECTED",
            metadata={
                "source": "trader_episode",
                "task_id": "legacy-2024-06",
                "entry_month": "2024-06",
                "available_after": "20240630",
                "net_return": 0.01,
                "index_return": 0.0,
                "active_return": 0.01,
            },
        )
        legacy_block = Mem0Memory(legacy_store).trader_episode_block("20240801")
        assert "旧记录没有锚点归因" in legacy_block
        assert "LEGACY_MODEL_REASONING_MUST_NOT_BE_INJECTED" not in legacy_block

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


def test_structured_satellite_attribution_memory() -> None:
    _, metadata, top4, eligible, _ = _core_satellite_fixture("neutral")
    selected = eligible[2]
    deterministic = eligible[0]
    metadata["satellite_selection"] = {
        "selected_codes": [selected],
        "deterministic_codes": [deterministic],
        "selection_fallback": False,
        "source": "agent",
        "diagnostic": "accepted 1 eligible satellite code(s) from answer",
        "candidate_signal_date": "20250102",
        "target": "excess_vs_000300.SH",
    }
    allocation = assemble_core_satellite_allocation(
        metadata["anchor_snapshot"],
        selected_satellites=[selected],
    )
    stock_returns = {
        code: (position - 4) / 100.0
        for position, code in enumerate(metadata["stock_pool"], start=1)
    }
    index_return = 0.005
    excess_returns = {
        code: value - index_return for code, value in stock_returns.items()
    }
    actual_result = evaluate_portfolio_month(
        allocation["weights"],
        allocation["cash"],
        stock_returns,
        index_return,
        starting_capital=1_000_000.0,
        excess_returns=excess_returns,
    )
    pure_result = evaluate_portfolio_month(
        {code: 0.25 for code in top4},
        0.0,
        stock_returns,
        index_return,
        starting_capital=1_000_000.0,
        excess_returns=excess_returns,
    )
    attribution = _build_core_satellite_attribution(
        metadata=metadata,
        snapshot=metadata["anchor_snapshot"],
        actual_weights=allocation["weights"],
        actual_cash=allocation["cash"],
        actual_result=actual_result,
        stock_returns=stock_returns,
        excess_returns=excess_returns,
        index_return=index_return,
        starting_capital=1_000_000.0,
        open_cost=0.0005,
        close_cost=0.0015,
        min_cost=5.0,
        anchor_attribution={"anchor_net_return": pure_result.net_return},
    )
    assert attribution is not None
    assert attribution["selected_codes"] == [selected]
    assert attribution["deterministic_codes"] == [deterministic]
    assert attribution["selection_source"] == "agent"
    assert attribution["selection_fallback"] is False
    assert attribution["selection_differs"] is True
    assert attribution["satellite_weight"] == 0.10
    assert attribution["core_weight"] == 0.90
    assert attribution["cash_weight"] == 0.0
    assert {
        row["code"] for row in attribution["candidate_facts"]
    } == {selected, deterministic}

    secret = "MODEL_CHAIN_OF_THOUGHT_MUST_NEVER_PERSIST"
    satellite_payload = {
        **attribution,
        "model_reasoning": secret,
        "candidate_facts": [
            {**row, "rationale": secret}
            for row in attribution["candidate_facts"]
        ],
    }
    with tempfile.TemporaryDirectory() as tmp:
        store = InMemoryStore(tmp, "core-satellite-attribution")
        memory = Mem0Memory(store=store)
        operation = memory.add_trader_episode(
            task_id="ashare_trader_2025-01-02",
            month="2025-01",
            available_after="20250131",
            weights=allocation["weights"],
            cash=allocation["cash"],
            gross_return=actual_result.gross_return,
            net_return=actual_result.net_return,
            index_return=index_return,
            active_return=actual_result.active_return,
            total_cost=actual_result.total_cost,
            contributions=actual_result.contributions,
            satellite_attribution=satellite_payload,
            reasoning=secret,
            parse_ok=True,
        )
        assert operation.startswith("ADD")
        assert len(store.records) == 1
        record = store.records[0]
        assert record.metadata["satellite_attribution_version"] == 1
        stored = record.metadata["satellite_attribution"]
        assert stored["selected_codes"] == [selected]
        assert stored["deterministic_codes"] == [deterministic]
        assert stored["selection_source"] == "agent"
        assert stored["satellite_net_contribution"] == round(
            attribution["satellite_net_contribution"],
            10,
        )
        assert all(
            "rationale" not in row and "analysis" not in row
            for row in stored["candidate_facts"]
        )
        serialized = json.dumps(
            {"content": record.content, "metadata": record.metadata},
            ensure_ascii=False,
            sort_keys=True,
        )
        assert secret not in serialized
        assert "model_reasoning" not in serialized
        assert memory.trader_episode_block("20250130") == ""

        visible = memory.trader_episode_block("20250201")
        assert "有成熟卫星归因 1 期" in visible
        assert "卫星净贡献逐期合计" in visible
        assert "实际与确定性top卫星不同 1 期（source=agent 1 期）" in visible
        assert "模型/agent选择相对确定性反事实复合增量" in visible
        assert f"实际卫星={selected}" in visible
        assert f"确定性top={deterministic}" in visible
        assert (
            f"{attribution['satellite_net_contribution']:+.2%}" in visible
        )
        assert secret not in visible


def test_dispatch_and_configs() -> None:
    try:
        create_memory_store(
            store_dir="memory_bank",
            namespace="../../unsafe",
        )
        raise AssertionError("unsafe run-scoped namespace was accepted")
    except ValueError:
        pass

    config_dir = str(ROOT / "config")
    previous_run_id = os.environ.pop("ASHARE_TRADER_RUN_ID", None)
    previous_api_key = os.environ.get("KIMI_API_KEY")
    os.environ["KIMI_API_KEY"] = "offline-smoke-placeholder"
    try:
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            unresolved = compose(
                config_name="agent_ashare_trader_core_satellite_attribution_kimi"
            )
        try:
            OmegaConf.resolve(unresolved)
            raise AssertionError(
                "core-satellite attribution config accepted no explicit run id"
            )
        except InterpolationResolutionError as error:
            assert "ASHARE_TRADER_RUN_ID" in str(error)
    finally:
        if previous_run_id is not None:
            os.environ["ASHARE_TRADER_RUN_ID"] = previous_run_id
        if previous_api_key is None:
            os.environ.pop("KIMI_API_KEY", None)
        else:
            os.environ["KIMI_API_KEY"] = previous_api_key

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

    with initialize_config_dir(version_base=None, config_dir=config_dir):
        anchor60 = compose(config_name="agent_ashare_trader_anchor60_kimi")
    assert anchor60.benchmark.anchor_policy.enabled is True
    assert anchor60.benchmark.anchor_policy.min_top4_weight == 0.60
    task_data = _load_tasks()[0]
    runtime_task = BenchmarkTask(
        task_id=task_data["task_id"],
        task_question=task_data["task_question"],
        ground_truth=task_data["ground_truth"],
        file_path=None,
        metadata=dict(task_data["metadata"]),
    )
    attach_probe = SimpleNamespace(cfg=anchor60)
    BenchmarkEvaluator._attach_trader_anchor_policy(
        attach_probe,  # type: ignore[arg-type]
        runtime_task,
    )
    assert runtime_task.metadata["anchor_policy"]["min_top4_weight"] == 0.60
    assert (
        runtime_task.metadata["anchor_snapshot"]["as_of"]
        == task_data["metadata"]["entry_date"]
    )
    assert "## 动量硬锚（系统将确定性复核）" in runtime_task.task_question
    assert "ground_truth_rank" not in runtime_task.task_question
    assert "satellite_candidates" not in runtime_task.metadata
    assert "satellite_required_count" not in runtime_task.metadata

    with initialize_config_dir(version_base=None, config_dir=config_dir):
        anchor75 = compose(config_name="agent_ashare_trader_anchor75_kimi")
    assert anchor75.benchmark.anchor_policy.min_top4_weight == 0.75

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

        with initialize_config_dir(version_base=None, config_dir=config_dir):
            attribution = compose(
                config_name="agent_ashare_trader_attribution_kimi"
            )
        OmegaConf.resolve(attribution)
        assert attribution.benchmark.anchor_policy.min_top4_weight == 0.60
        assert attribution.memory.namespace == (
            "ashare_trader_attribution_smoke-isolated"
        )
        assert attribution.memory.skill_enabled is False
        assert attribution.memory.skill_top_k == 0

        with initialize_config_dir(version_base=None, config_dir=config_dir):
            core = compose(
                config_name="agent_ashare_trader_core_satellite_attribution_kimi"
            )
        OmegaConf.resolve(core)
        assert core.benchmark.anchor_policy.enabled is True
        assert core.benchmark.anchor_policy.mode == "core_satellite"
        assert core.benchmark.anchor_policy.candidate_limit == 6
        assert core.memory.namespace == (
            "ashare_trader_core_satellite_attribution_smoke-isolated"
        )
        assert core.memory.skill_enabled is False
        assert core.memory.skill_top_k == 0
    finally:
        if previous is None:
            os.environ.pop("ASHARE_TRADER_RUN_ID", None)
        else:
            os.environ["ASHARE_TRADER_RUN_ID"] = previous


def main() -> None:
    generate_tasks()
    tasks = _load_tasks()
    test_core_satellite_policy_matrix()
    test_core_satellite_intent_canonicalization()
    test_momentum_anchor_skill(tasks)
    test_tasks_and_panel(tasks)
    test_satellite_signal_pit_and_manifest(tasks)
    test_hard_anchor_and_breadth(tasks)
    test_trader_guardrails(tasks)
    test_core_satellite_runtime_attachment(tasks)
    test_parser_and_finance(tasks)
    test_episode_memory_and_factors(tasks)
    test_structured_satellite_attribution_memory()
    test_dispatch_and_configs()
    print(
        "[OK] MemSkill ashare-trader: legacy coverage plus exact core-satellite "
        "sleeves, PIT candidates, canonical intent, and matured factual attribution"
    )


if __name__ == "__main__":
    main()
