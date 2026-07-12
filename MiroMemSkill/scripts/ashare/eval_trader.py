#!/usr/bin/env python3
"""Backtest unified A-share trader allocations with costs and cash."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils.ashare_anchor import (  # noqa: E402
    AnchorPolicy,
    assemble_core_satellite_allocation,
    build_anchor_snapshot,
    validate_anchor_allocation,
    validate_core_satellite_allocation,
)
from src.utils.ashare_satellite import (  # noqa: E402
    load_excess_signal_candidates,
)
from src.utils.ashare_trader import (  # noqa: E402
    DEFAULT_CLOSE_COST,
    DEFAULT_MIN_COST,
    DEFAULT_OPEN_COST,
    PortfolioParseResult,
    PortfolioMonthResult,
    cash_allocation,
    evaluate_anchor_deviation,
    evaluate_portfolio_month,
    parse_portfolio_weights,
)
from src.utils.ashare_trader_features import (  # noqa: E402
    compute_trader_feature_rows,
)

DEFAULT_TASKS = ROOT / "data" / "ashare_trader" / "standardized_data.jsonl"
DATA_DIR = ROOT / "data" / "ashare"
CORE_SATELLITE_BASELINE = "core_satellite_deterministic(PIT excess model)"
CORE_SATELLITE_CANDIDATE_LIMIT = 6
DYNAMIC_CORE_CASH_DILUTION_BASELINE = "dynamic_core_cash_dilution"
# Pre-registered reproducible arms; each seed is deterministically derived per task.
RANDOM_SATELLITE_SEEDS = (7, 17, 29, 43, 71)
ORACLE_SATELLITE_BASELINE = "oracle_satellite_upper_bound(PIT top6,不可交易)"


def load_tasks(path: str | Path) -> dict[str, dict[str, Any]]:
    tasks = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {str(task["task_id"]): task for task in tasks}


def load_run(path: str | Path) -> dict[str, str]:
    answers: dict[str, str] = {}
    for filename in glob.glob(str(Path(path) / "task_*_attempt_1.json")):
        data = json.loads(Path(filename).read_text(encoding="utf-8"))
        if data.get("status") != "completed":
            continue
        task_id = str(data.get("task_id") or data.get("task_name") or "")
        if task_id:
            answers[task_id] = str(
                data.get("final_boxed_answer") or data.get("model_boxed_answer") or ""
            )
    return answers


def _normalized_codes(value: Any) -> list[str]:
    if isinstance(value, str):
        raw: Sequence[Any] = [value]
    elif isinstance(value, Mapping):
        raw = list(value)
    elif isinstance(value, Sequence):
        raw = value
    else:
        return []
    return list(
        dict.fromkeys(code for item in raw if (code := str(item or "").strip().upper()))
    )


def _optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
    return None


def load_run_audits(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load canonical core-satellite selection facts without changing load_run."""
    audits: dict[str, dict[str, Any]] = {}
    step_names = {
        "trader_core_satellite_canonicalized",
        "trader_core_satellite_finalized",
    }
    for filename in glob.glob(str(Path(path) / "task_*_attempt_1.json")):
        try:
            data = json.loads(Path(filename).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("status") != "completed":
            continue
        task_id = str(data.get("task_id") or data.get("task_name") or "")
        if not task_id:
            continue
        selected: dict[str, Any] | None = None
        raw_steps = data.get("step_logs")
        if not isinstance(raw_steps, list):
            raw_steps = data.get("steps")
        for step in raw_steps if isinstance(raw_steps, list) else []:
            if not isinstance(step, Mapping) or step.get("step_name") not in step_names:
                continue
            metadata = step.get("metadata")
            if not isinstance(metadata, Mapping):
                continue
            source = str(metadata.get("source") or "").strip().lower()
            if source not in {"agent", "deterministic_fallback"}:
                source = "unknown"
            selected_codes = _normalized_codes(metadata.get("selected_codes"))
            deterministic_codes = _normalized_codes(metadata.get("deterministic_codes"))
            fallback = _optional_bool(metadata.get("selection_fallback"))
            if (
                not selected_codes
                and not deterministic_codes
                and fallback is None
                and source == "unknown"
            ):
                continue
            selected = {
                "audit_present": True,
                "audit_step": str(step.get("step_name")),
                "selected_codes": selected_codes,
                "deterministic_codes": deterministic_codes,
                "selection_fallback": fallback,
                "source": source,
                "candidate_signal_date": str(
                    metadata.get("candidate_signal_date") or ""
                ).strip(),
                "target": str(metadata.get("target") or "").strip(),
            }
        if selected is not None:
            audits[task_id] = selected
    return audits


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def _sample_std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else float("nan")


def _annualized_ratio(values: list[float]) -> float:
    std = _sample_std(values)
    return (
        _mean(values) / std * math.sqrt(12)
        if math.isfinite(std) and std > 0
        else float("nan")
    )


def _max_drawdown(capitals: list[float]) -> float:
    if not capitals:
        return float("nan")
    peak = capitals[0]
    worst = 0.0
    for capital in capitals:
        peak = max(peak, capital)
        if peak > 0:
            worst = min(worst, capital / peak - 1.0)
    return worst


def _finite_mean(values: list[float]) -> float:
    clean = [value for value in values if math.isfinite(value)]
    return _mean(clean)


def parse_run_allocations(
    tasks: Mapping[str, dict[str, Any]],
    answers: Mapping[str, str],
) -> dict[str, PortfolioParseResult]:
    parsed: dict[str, PortfolioParseResult] = {}
    for task_id, task in tasks.items():
        metadata = task["metadata"]
        if task_id not in answers:
            pool = metadata["stock_pool"]
            parsed[task_id] = PortfolioParseResult(
                {code: 0.0 for code in pool},
                1.0,
                False,
                "missing completed answer",
            )
            continue
        parsed[task_id] = parse_portfolio_weights(
            answers[task_id],
            metadata["stock_pool"],
            max_stock_weight=float(metadata.get("max_stock_weight", 0.25)),
        )
    return parsed


def _allocation_from_weights(
    pool: Sequence[str],
    weights: Mapping[str, Any],
    cash: Any,
    *,
    ok: bool = True,
    error: str = "",
) -> PortfolioParseResult:
    normalized = {str(code): float(weights.get(str(code), 0.0)) for code in pool}
    return PortfolioParseResult(normalized, float(cash), ok, error)


def build_core_satellite_contexts(
    tasks: Mapping[str, dict[str, Any]],
    *,
    policy: AnchorPolicy | None = None,
    candidate_limit: int = CORE_SATELLITE_CANDIDATE_LIMIT,
) -> dict[str, dict[str, Any]]:
    """Rebuild the runtime policy state strictly from point-in-time inputs."""
    core_policy = policy or AnchorPolicy(enabled=True, mode="core_satellite")
    if not core_policy.enabled or core_policy.mode != "core_satellite":
        raise ValueError("core-satellite contexts require an enabled core policy")
    if candidate_limit <= 0:
        raise ValueError("candidate_limit must be positive")

    contexts: dict[str, dict[str, Any]] = {}
    for task_id, task in tasks.items():
        metadata = task["metadata"]
        pool = [str(code).upper() for code in metadata["stock_pool"]]
        pool_set = set(pool)
        context: dict[str, Any] = {
            "policy_mode": "core_satellite",
            "policy": core_policy,
            "candidate_limit": candidate_limit,
            "candidate_rows": [],
            "candidate_codes": [],
            "candidate_signal_date": "",
            "target": "",
            "top4": [],
            "snapshot": None,
            "deterministic_allocation": None,
            "deterministic_satellites": [],
            "pure_momentum_allocation": None,
            "dynamic_core_cash_dilution_allocation": None,
            "assembled_deterministic": None,
            "error": "",
        }
        try:
            as_of = str(metadata["entry_date"])
            snapshot = build_anchor_snapshot(
                as_of,
                metadata["stock_info"],
                data_dir=DATA_DIR,
                policy=core_policy,
            )
            context["snapshot"] = snapshot
            context["top4"] = [str(code) for code in snapshot.get("top4", [])]
            context["pure_momentum_allocation"] = _allocation_from_weights(
                pool,
                snapshot.get("anchor_weights", {}),
                snapshot.get("anchor_cash", 0.0),
            )

            loaded = load_excess_signal_candidates(
                as_of,
                context["top4"],
                data_dir=DATA_DIR,
            )
            candidates = [
                dict(row)
                for row in loaded
                if str(row.get("ts_code", "")).strip().upper() in pool_set
            ][:candidate_limit]
            context["candidate_rows"] = candidates
            context["candidate_codes"] = [
                str(row.get("ts_code", "")).strip().upper() for row in candidates
            ]
            signal_dates = {
                str(row.get("signal_date", "")).strip() for row in candidates
            }
            targets = {str(row.get("target", "")).strip() for row in candidates}
            if len(signal_dates) != 1 or "" in signal_dates:
                raise ValueError("PIT candidates require one non-empty signal_date")
            if len(targets) != 1 or "" in targets:
                raise ValueError("PIT candidates require one non-empty target")
            context["candidate_signal_date"] = next(iter(signal_dates))
            context["target"] = next(iter(targets))

            snapshot = build_anchor_snapshot(
                as_of,
                metadata["stock_info"],
                data_dir=DATA_DIR,
                policy=core_policy,
                prediction_candidates=candidates,
            )
            context["snapshot"] = snapshot
            assembled = assemble_core_satellite_allocation(snapshot)
            deterministic = _allocation_from_weights(
                pool,
                assembled["weights"],
                assembled["cash"],
            )
            validation = validate_core_satellite_allocation(
                deterministic.weights,
                deterministic.cash,
                snapshot=snapshot,
                policy=core_policy,
            )
            if not validation.ok:
                raise ValueError(
                    "invalid deterministic core-satellite allocation: "
                    + validation.error
                )
            context["assembled_deterministic"] = assembled
            context["deterministic_allocation"] = deterministic
            context["deterministic_satellites"] = [
                str(code) for code in assembled["satellite_codes"]
            ]
            context["dynamic_core_cash_dilution_allocation"] = _allocation_from_weights(
                pool,
                assembled["core_weights"],
                float(assembled["cash"]) + float(assembled["satellite_total_weight"]),
            )
        except (KeyError, OSError, TypeError, ValueError) as exc:
            context["error"] = str(exc)
        contexts[task_id] = context
    return contexts


def _invalid_allocation(
    pool: Sequence[str],
    error: str,
) -> PortfolioParseResult:
    return PortfolioParseResult(
        {str(code): 0.0 for code in pool},
        1.0,
        False,
        error,
    )


def _core_satellite_allocation_for_selection(
    context: Mapping[str, Any],
    pool: Sequence[str],
    selected_codes: Sequence[str],
) -> PortfolioParseResult:
    snapshot = context.get("snapshot")
    policy = context.get("policy")
    if not isinstance(snapshot, Mapping) or not isinstance(policy, AnchorPolicy):
        raise ValueError(str(context.get("error") or "missing core-satellite context"))
    assembled = assemble_core_satellite_allocation(
        snapshot,
        selected_satellites=selected_codes,
    )
    allocation = _allocation_from_weights(
        pool,
        assembled["weights"],
        assembled["cash"],
    )
    validation = validate_core_satellite_allocation(
        allocation.weights,
        allocation.cash,
        snapshot=snapshot,
        policy=policy,
    )
    if not validation.ok:
        raise ValueError(validation.error)
    return allocation


def _random_satellite_codes(
    task_id: str,
    candidates: Sequence[str],
    satellite_count: int,
    seed: int,
) -> list[str]:
    """Select PIT candidates with a stable per-task derivation of a fixed seed."""
    if satellite_count <= 0 or len(candidates) < satellite_count:
        raise ValueError("insufficient PIT candidates for random satellites")
    digest = hashlib.sha256(f"ashare-core-satellite:{seed}:{task_id}".encode()).digest()
    task_seed = int.from_bytes(digest[:8], "big")
    return random.Random(task_seed).sample(list(candidates), satellite_count)


def _random_satellite_baseline_name(seed: int) -> str:
    return f"core_satellite_random(seed={seed})"


def _validate_dynamic_core_cash_dilution(
    allocation: PortfolioParseResult,
    context: Mapping[str, Any],
) -> tuple[bool, str]:
    """Validate fixed regime core with the full satellite sleeve moved to cash."""
    assembled = context.get("assembled_deterministic")
    if not isinstance(assembled, Mapping):
        return False, str(context.get("error") or "missing deterministic sleeve")
    core_weights = assembled.get("core_weights")
    if not isinstance(core_weights, Mapping):
        return False, "missing fixed core weights"
    expected_cash = float(assembled["cash"]) + float(
        assembled["satellite_total_weight"]
    )
    errors: list[str] = []
    for code, expected in core_weights.items():
        actual = float(allocation.weights.get(str(code), 0.0))
        if abs(actual - float(expected)) > 1e-8:
            errors.append(
                f"{code} core weight {actual:.3%} must equal " f"{float(expected):.3%}"
            )
    core_codes = {str(code) for code in core_weights}
    non_core = [
        str(code)
        for code, weight in allocation.weights.items()
        if str(code) not in core_codes and float(weight) > 1e-8
    ]
    if non_core:
        errors.append("dilution comparator cannot hold satellites")
    if abs(float(allocation.cash) - expected_cash) > 1e-8:
        errors.append(
            f"dilution cash {float(allocation.cash):.2%} must equal "
            f"{expected_cash:.2%}"
        )
    return not errors, "; ".join(errors)


def _anchor_policy_for_task(
    metadata: Mapping[str, Any],
    *,
    override: Mapping[str, Any] | None = None,
) -> AnchorPolicy:
    raw = dict(metadata.get("anchor_policy") or {})
    if override:
        raw.update(dict(override))
    if not raw:
        raw = {"enabled": True, "min_top4_weight": 0.60}
    return AnchorPolicy.from_mapping(raw)


def _anchor_metrics_for_month(
    metadata: Mapping[str, Any],
    allocation: PortfolioParseResult,
    *,
    starting_capital: float,
    actual_result: PortfolioMonthResult,
    open_cost: float,
    close_cost: float,
    min_cost: float,
    anchor_policy: AnchorPolicy | None = None,
) -> dict[str, Any]:
    policy = anchor_policy or _anchor_policy_for_task(metadata)
    if not policy.enabled:
        return {}
    snapshot = build_anchor_snapshot(
        str(metadata["entry_date"]),
        metadata["stock_info"],
        data_dir=DATA_DIR,
        policy=policy,
    )
    validation = validate_anchor_allocation(
        allocation.weights,
        allocation.cash,
        snapshot=snapshot,
        policy=policy,
    )
    deviation = evaluate_anchor_deviation(
        allocation.weights,
        allocation.cash,
        snapshot["anchor_weights"],
        float(snapshot.get("anchor_cash", 0.0)),
        metadata["stock_returns"],
        float(metadata["index_return"]),
        starting_capital=starting_capital,
        excess_returns=metadata.get("excess_returns"),
        open_cost=open_cost,
        close_cost=close_cost,
        min_cost=min_cost,
        actual_result=actual_result,
    )
    return {
        "anchor_floor": float(policy.min_top4_weight),
        "anchor_compliant": validation.ok,
        "anchor_errors": validation.error,
        "top4_exposure": validation.metrics.get("top4_exposure", 0.0),
        "top4_holding_count": validation.metrics.get("top4_holding_count", 0),
        "top4_overlap_count": deviation.get("overlap_count", 0),
        "replacement_count": validation.metrics.get("replacement_count", 0),
        "market_regime": str(snapshot.get("market_breadth", {}).get("regime", "")),
        "actual_net_return": deviation.get("actual_net_return", 0.0),
        "anchor_net_return": deviation.get("anchor_net_return", 0.0),
        "deviation_net_return": deviation.get("deviation_net_return", 0.0),
        "deviation_active_return": deviation.get("deviation_active_return", 0.0),
        "anchor_active_return": deviation.get("anchor_active_return", 0.0),
        "deviation_pnl": (
            starting_capital * float(deviation.get("deviation_net_return", 0.0))
        ),
    }


def _annualized_compound_return(total_factor: float, months: int) -> float:
    if months <= 0 or not math.isfinite(total_factor) or total_factor <= 0:
        return float("nan")
    return total_factor ** (12.0 / months) - 1.0


def _annualized_excess_return(
    strategy_factor: float,
    benchmark_factor: float,
    months: int,
) -> float:
    if benchmark_factor <= 0:
        return float("nan")
    return _annualized_compound_return(
        strategy_factor / benchmark_factor,
        months,
    )


def _evaluate_counterfactual(
    allocation: Any,
    metadata: Mapping[str, Any],
    *,
    starting_capital: float,
    open_cost: float,
    close_cost: float,
    min_cost: float,
) -> PortfolioMonthResult | None:
    if not isinstance(allocation, PortfolioParseResult) or not allocation.ok:
        return None
    try:
        return evaluate_portfolio_month(
            allocation.weights,
            allocation.cash,
            metadata["stock_returns"],
            float(metadata["index_return"]),
            starting_capital=starting_capital,
            excess_returns=metadata.get("excess_returns"),
            open_cost=open_cost,
            close_cost=close_cost,
            min_cost=min_cost,
        )
    except (KeyError, TypeError, ValueError):
        return None


def _ranked_satellites(
    allocation: PortfolioParseResult,
    top4: Sequence[str],
    candidates: Sequence[str],
) -> list[str]:
    top4_set = set(top4)
    candidate_rank = {code: rank for rank, code in enumerate(candidates)}
    return sorted(
        (
            str(code)
            for code, weight in allocation.weights.items()
            if str(code) not in top4_set and float(weight) > 1e-8
        ),
        key=lambda code: (candidate_rank.get(code, len(candidate_rank)), code),
    )


def _satellite_performance(
    codes: Sequence[str],
    allocation: PortfolioParseResult,
    result: PortfolioMonthResult | None,
    metadata: Mapping[str, Any],
    *,
    starting_capital: float,
) -> dict[str, float]:
    if result is None:
        return {
            "weight": float("nan"),
            "gross_return": float("nan"),
            "excess_return": float("nan"),
            "net_return": float("nan"),
            "weighted_return_contribution": float("nan"),
            "weighted_excess_contribution": float("nan"),
            "net_contribution": float("nan"),
            "gross_pnl": float("nan"),
            "net_pnl": float("nan"),
        }
    stock_returns = metadata["stock_returns"]
    index_return = float(metadata["index_return"])
    excess_returns = metadata.get("excess_returns") or {}
    weight = sum(float(allocation.weights.get(code, 0.0)) for code in codes)
    gross_contribution = sum(
        float(allocation.weights.get(code, 0.0)) * float(stock_returns[code])
        for code in codes
    )
    excess_contribution = sum(
        float(allocation.weights.get(code, 0.0))
        * float(
            excess_returns.get(
                code,
                float(stock_returns[code]) - index_return,
            )
        )
        for code in codes
    )
    net_contribution = sum(float(result.contributions.get(code, 0.0)) for code in codes)
    return {
        "weight": weight,
        "gross_return": (gross_contribution / weight if weight > 0 else float("nan")),
        "excess_return": (excess_contribution / weight if weight > 0 else float("nan")),
        "net_return": (net_contribution / weight if weight > 0 else float("nan")),
        "weighted_return_contribution": gross_contribution,
        "weighted_excess_contribution": excess_contribution,
        "net_contribution": net_contribution,
        "gross_pnl": starting_capital * gross_contribution,
        "net_pnl": starting_capital * net_contribution,
    }


def _core_satellite_metrics_for_month(
    *,
    name: str,
    run_kind: str,
    allocation: PortfolioParseResult,
    actual_result: PortfolioMonthResult,
    metadata: Mapping[str, Any],
    context: Mapping[str, Any],
    audit: Mapping[str, Any] | None,
    policy: AnchorPolicy,
    deterministic_result: PortfolioMonthResult | None,
    deterministic_at_actual_start: PortfolioMonthResult | None,
    dilution_result: PortfolioMonthResult | None,
    dilution_at_actual_start: PortfolioMonthResult | None,
    pure_momentum_result: PortfolioMonthResult | None,
    pure_momentum_at_actual_start: PortfolioMonthResult | None,
    initial_capital: float,
) -> dict[str, Any]:
    snapshot = context.get("snapshot")
    top4 = [str(code) for code in context.get("top4", [])]
    candidates = [str(code) for code in context.get("candidate_codes", [])]
    actual_satellites = _ranked_satellites(allocation, top4, candidates)
    pit_deterministic_codes = [
        str(code) for code in context.get("deterministic_satellites", [])
    ]
    deterministic_allocation = context.get("deterministic_allocation")
    pure_momentum_allocation = context.get("pure_momentum_allocation")
    dilution_allocation = context.get("dynamic_core_cash_dilution_allocation")
    pit_candidate_signal_date = str(context.get("candidate_signal_date") or "")
    pit_candidate_target = str(context.get("target") or "")

    audit_mapping = audit if isinstance(audit, Mapping) else {}
    audit_present = bool(audit_mapping.get("audit_present"))
    if audit_present:
        selected_codes = _normalized_codes(audit_mapping.get("selected_codes"))
        selection_origin = "run_log"
        selection_fallback = _optional_bool(audit_mapping.get("selection_fallback"))
        selection_source = str(audit_mapping.get("source") or "unknown").strip().lower()
    else:
        selected_codes = list(actual_satellites)
        selection_origin = "final_allocation_inferred"
        if run_kind == "baseline" and name == CORE_SATELLITE_BASELINE:
            selection_fallback = False
            selection_source = "deterministic_baseline"
        elif run_kind == "baseline":
            selection_fallback = None
            selection_source = "baseline"
        else:
            # Equality with the deterministic allocation is not evidence that
            # runtime fallback occurred.
            selection_fallback = None
            selection_source = "unknown"
    if selection_source not in {
        "agent",
        "deterministic_fallback",
        "deterministic_baseline",
        "baseline",
    }:
        selection_source = "unknown"
    audit_deterministic_codes = _normalized_codes(
        audit_mapping.get("deterministic_codes")
    )
    audit_candidate_signal_date = str(audit_mapping.get("candidate_signal_date") or "")
    audit_target = str(audit_mapping.get("target") or "")
    candidate_signal_date = (
        audit_candidate_signal_date
        if audit_present and audit_candidate_signal_date
        else pit_candidate_signal_date
    )
    candidate_target = (
        audit_target if audit_present and audit_target else pit_candidate_target
    )
    reported_deterministic_codes = (
        audit_deterministic_codes
        if audit_present and audit_deterministic_codes
        else pit_deterministic_codes
    )

    validation_ok = False
    validation_error = str(context.get("error") or "")
    validation_metrics: Mapping[str, Any] = {}
    if isinstance(snapshot, Mapping):
        try:
            validation = validate_core_satellite_allocation(
                allocation.weights,
                allocation.cash,
                snapshot=snapshot,
                policy=policy,
            )
            validation_ok = validation.ok
            validation_error = validation.error
            validation_metrics = validation.metrics
        except (KeyError, TypeError, ValueError) as exc:
            validation_error = str(exc)
    dilution_compliant, dilution_error = _validate_dynamic_core_cash_dilution(
        allocation,
        context,
    )
    paired_dilution_valid = validation_ok or dilution_compliant

    assembled = context.get("assembled_deterministic")
    sleeve = assembled.get("sleeve", {}) if isinstance(assembled, Mapping) else {}
    expected_core_stock_weight = float(sleeve.get("core_stock_weight", float("nan")))
    mandatory_top4_compliant = len(top4) == 4 and all(
        float(allocation.weights.get(code, 0.0)) > 1e-8 for code in top4
    )
    core_weight_compliant = (
        len(top4) == 4
        and math.isfinite(expected_core_stock_weight)
        and all(
            abs(float(allocation.weights.get(code, 0.0)) - expected_core_stock_weight)
            <= 1e-8
            for code in top4
        )
    )
    core_weight = sum(float(allocation.weights.get(code, 0.0)) for code in top4)
    satellite_weight = sum(
        float(allocation.weights.get(code, 0.0)) for code in actual_satellites
    )
    actual_satellite = _satellite_performance(
        actual_satellites,
        allocation,
        actual_result,
        metadata,
        starting_capital=actual_result.starting_capital,
    )
    deterministic_satellite = _satellite_performance(
        pit_deterministic_codes,
        (
            deterministic_allocation
            if isinstance(deterministic_allocation, PortfolioParseResult)
            else cash_allocation(metadata["stock_pool"])
        ),
        deterministic_at_actual_start,
        metadata,
        starting_capital=actual_result.starting_capital,
    )

    deterministic_same_start_return = (
        float(deterministic_at_actual_start.net_return)
        if deterministic_at_actual_start is not None
        else float("nan")
    )
    dilution_same_start_return = (
        float(dilution_at_actual_start.net_return)
        if dilution_at_actual_start is not None
        else float("nan")
    )
    pure_same_start_return = (
        float(pure_momentum_at_actual_start.net_return)
        if pure_momentum_at_actual_start is not None
        else float("nan")
    )
    arithmetic_vs_deterministic = (
        float(actual_result.net_return) - deterministic_same_start_return
    )
    arithmetic_vs_dilution = (
        float(actual_result.net_return) - dilution_same_start_return
    )
    arithmetic_vs_pure = float(actual_result.net_return) - pure_same_start_return
    deterministic_ending = (
        float(deterministic_result.ending_capital)
        if deterministic_result is not None
        else float("nan")
    )
    dilution_ending = (
        float(dilution_result.ending_capital)
        if dilution_result is not None
        else float("nan")
    )
    pure_ending = (
        float(pure_momentum_result.ending_capital)
        if pure_momentum_result is not None
        else float("nan")
    )
    satellite_win = (
        actual_satellite["weighted_excess_contribution"] > 0
        if satellite_weight > 0
        and math.isfinite(actual_satellite["weighted_excess_contribution"])
        else None
    )
    market_regime = str(
        (snapshot or {}).get("market_breadth", {}).get("regime", "")
        if isinstance(snapshot, Mapping)
        else ""
    )

    return {
        "policy_mode": "core_satellite",
        "core_satellite_context_ok": (
            not context.get("error")
            and isinstance(deterministic_allocation, PortfolioParseResult)
        ),
        "core_satellite_context_error": str(context.get("error") or ""),
        "market_regime": market_regime,
        "regime": market_regime,
        "candidate_limit": int(
            context.get("candidate_limit", CORE_SATELLITE_CANDIDATE_LIMIT)
        ),
        "candidate_count": len(candidates),
        "candidate_codes": candidates,
        "candidate_signal_date": candidate_signal_date,
        "candidate_target": candidate_target,
        "target": candidate_target,
        "pit_candidate_signal_date": pit_candidate_signal_date,
        "pit_candidate_target": pit_candidate_target,
        "top4_codes": top4,
        "core_satellite_compliant": validation_ok,
        "core_satellite_errors": validation_error,
        "dynamic_core_cash_dilution_compliant": dilution_compliant,
        "dynamic_core_cash_dilution_errors": dilution_error,
        "paired_satellite_dilution_comparison_valid": paired_dilution_valid,
        "mandatory_top4_compliant": mandatory_top4_compliant,
        "top4_compliant": mandatory_top4_compliant,
        "core_weight_compliant": core_weight_compliant,
        "top4_exposure": float(validation_metrics.get("top4_exposure", core_weight)),
        "core_weight": core_weight,
        "satellite_weight": satellite_weight,
        "cash_weight": float(allocation.cash),
        "expected_core_weight": float(sleeve.get("core_total_weight", float("nan"))),
        "expected_satellite_weight": float(
            sleeve.get("satellite_total_weight", float("nan"))
        ),
        "expected_cash_weight": float(sleeve.get("cash_weight", float("nan"))),
        "dynamic_core_cash_dilution_weights": (
            {
                code: weight
                for code, weight in dilution_allocation.weights.items()
                if weight > 0
            }
            if isinstance(dilution_allocation, PortfolioParseResult)
            else {}
        ),
        "dynamic_core_cash_dilution_cash": (
            dilution_allocation.cash
            if isinstance(dilution_allocation, PortfolioParseResult)
            else float("nan")
        ),
        "allocation_satellites": actual_satellites,
        "selected_satellites": selected_codes,
        "deterministic_satellites": reported_deterministic_codes,
        "pit_deterministic_satellites": pit_deterministic_codes,
        "selected_codes": selected_codes,
        "deterministic_codes": reported_deterministic_codes,
        "selection_origin": selection_origin,
        "selection_audit_present": audit_present,
        "selection_audit_step": str(audit_mapping.get("audit_step") or ""),
        "selection_source": selection_source,
        "source": selection_source,
        "selection_fallback": selection_fallback,
        "audit_deterministic_codes": audit_deterministic_codes,
        "selection_audit_matches_allocation": (
            set(selected_codes) == set(actual_satellites) if audit_present else None
        ),
        "audit_deterministic_matches_pit": (
            set(audit_deterministic_codes) == set(pit_deterministic_codes)
            if audit_present and audit_deterministic_codes
            else None
        ),
        "audit_candidate_signal_date": audit_candidate_signal_date,
        "audit_target": audit_target,
        "candidate_audit_matches_pit": (
            candidate_signal_date == pit_candidate_signal_date
            and candidate_target == pit_candidate_target
            if audit_present
            else None
        ),
        "satellite_gross_return": actual_satellite["gross_return"],
        "satellite_excess_return": actual_satellite["excess_return"],
        "satellite_net_return": actual_satellite["net_return"],
        "satellite_weighted_return_contribution": actual_satellite[
            "weighted_return_contribution"
        ],
        "satellite_gross_contribution": actual_satellite[
            "weighted_return_contribution"
        ],
        "satellite_weighted_excess_contribution": actual_satellite[
            "weighted_excess_contribution"
        ],
        "satellite_net_contribution": actual_satellite["net_contribution"],
        "satellite_gross_pnl": actual_satellite["gross_pnl"],
        "satellite_net_pnl": actual_satellite["net_pnl"],
        "satellite_gross_contribution_pnl": actual_satellite["gross_pnl"],
        "satellite_net_contribution_pnl": actual_satellite["net_pnl"],
        "satellite_excess_win": satellite_win,
        "satellite_excess_win_indicator": (
            int(satellite_win) if satellite_win is not None else None
        ),
        "deterministic_satellite_gross_return": deterministic_satellite["gross_return"],
        "deterministic_satellite_excess_return": deterministic_satellite[
            "excess_return"
        ],
        "deterministic_satellite_net_return": deterministic_satellite["net_return"],
        "deterministic_satellite_weighted_return_contribution": (
            deterministic_satellite["weighted_return_contribution"]
        ),
        "deterministic_satellite_weighted_excess_contribution": (
            deterministic_satellite["weighted_excess_contribution"]
        ),
        "deterministic_satellite_net_contribution": deterministic_satellite[
            "net_contribution"
        ],
        "deterministic_satellite_gross_pnl": deterministic_satellite["gross_pnl"],
        "deterministic_satellite_net_pnl": deterministic_satellite["net_pnl"],
        "actual_net_return": float(actual_result.net_return),
        "deterministic_counterfactual_available": (deterministic_result is not None),
        "deterministic_counterfactual_starting_capital": (
            float(deterministic_result.starting_capital)
            if deterministic_result is not None
            else float("nan")
        ),
        "deterministic_counterfactual_ending_capital": deterministic_ending,
        "deterministic_counterfactual_net_return": (
            float(deterministic_result.net_return)
            if deterministic_result is not None
            else float("nan")
        ),
        "deterministic_counterfactual_total_cost": (
            float(deterministic_result.total_cost)
            if deterministic_result is not None
            else float("nan")
        ),
        "deterministic_counterfactual_two_way_turnover": (
            deterministic_result.gross_traded_notional
            / deterministic_result.starting_capital
            if deterministic_result is not None
            else float("nan")
        ),
        "deterministic_counterfactual_net_return_at_actual_start": (
            deterministic_same_start_return
        ),
        "actual_minus_deterministic_net_return": arithmetic_vs_deterministic,
        "actual_minus_deterministic_pnl": (
            arithmetic_vs_deterministic * actual_result.starting_capital
        ),
        "arithmetic_agent_incremental_return_vs_deterministic": (
            arithmetic_vs_deterministic
        ),
        "arithmetic_agent_incremental_pnl_vs_deterministic": (
            arithmetic_vs_deterministic * actual_result.starting_capital
        ),
        "compounded_agent_incremental_return_vs_deterministic": (
            (actual_result.ending_capital - deterministic_ending) / initial_capital
        ),
        "compounded_agent_incremental_pnl_vs_deterministic": (
            actual_result.ending_capital - deterministic_ending
        ),
        "compounded_relative_nav_vs_deterministic": (
            actual_result.ending_capital / deterministic_ending - 1.0
            if deterministic_ending > 0
            else float("nan")
        ),
        "dynamic_core_cash_dilution_counterfactual_available": (
            dilution_result is not None
        ),
        "dynamic_core_cash_dilution_counterfactual_starting_capital": (
            float(dilution_result.starting_capital)
            if dilution_result is not None
            else float("nan")
        ),
        "dynamic_core_cash_dilution_counterfactual_ending_capital": (dilution_ending),
        "dynamic_core_cash_dilution_counterfactual_net_return": (
            float(dilution_result.net_return)
            if dilution_result is not None
            else float("nan")
        ),
        "dynamic_core_cash_dilution_counterfactual_total_cost": (
            float(dilution_result.total_cost)
            if dilution_result is not None
            else float("nan")
        ),
        "dynamic_core_cash_dilution_counterfactual_two_way_turnover": (
            dilution_result.gross_traded_notional / dilution_result.starting_capital
            if dilution_result is not None
            else float("nan")
        ),
        "dynamic_core_cash_dilution_net_return_at_actual_start": (
            dilution_same_start_return
        ),
        "actual_minus_dynamic_core_cash_dilution_net_return": (arithmetic_vs_dilution),
        "actual_minus_dynamic_core_cash_dilution_pnl": (
            arithmetic_vs_dilution * actual_result.starting_capital
        ),
        "arithmetic_agent_incremental_return_vs_dynamic_core_cash_dilution": (
            arithmetic_vs_dilution
        ),
        "arithmetic_agent_incremental_pnl_vs_dynamic_core_cash_dilution": (
            arithmetic_vs_dilution * actual_result.starting_capital
        ),
        "compounded_agent_incremental_return_vs_dynamic_core_cash_dilution": (
            (actual_result.ending_capital - dilution_ending) / initial_capital
        ),
        "compounded_agent_incremental_pnl_vs_dynamic_core_cash_dilution": (
            actual_result.ending_capital - dilution_ending
        ),
        "compounded_relative_nav_vs_dynamic_core_cash_dilution": (
            actual_result.ending_capital / dilution_ending - 1.0
            if dilution_ending > 0
            else float("nan")
        ),
        "arithmetic_satellite_incremental_return_vs_dilution": (
            arithmetic_vs_dilution if paired_dilution_valid else float("nan")
        ),
        "arithmetic_satellite_incremental_pnl_vs_dilution": (
            arithmetic_vs_dilution * actual_result.starting_capital
            if paired_dilution_valid
            else float("nan")
        ),
        "compounded_satellite_incremental_return_vs_dilution": (
            (actual_result.ending_capital - dilution_ending) / initial_capital
            if paired_dilution_valid
            else float("nan")
        ),
        "compounded_satellite_incremental_pnl_vs_dilution": (
            actual_result.ending_capital - dilution_ending
            if paired_dilution_valid
            else float("nan")
        ),
        "pure_momentum_counterfactual_available": (pure_momentum_result is not None),
        "pure_momentum_counterfactual_starting_capital": (
            float(pure_momentum_result.starting_capital)
            if pure_momentum_result is not None
            else float("nan")
        ),
        "pure_momentum_counterfactual_ending_capital": pure_ending,
        "pure_momentum_counterfactual_net_return": (
            float(pure_momentum_result.net_return)
            if pure_momentum_result is not None
            else float("nan")
        ),
        "pure_momentum_counterfactual_total_cost": (
            float(pure_momentum_result.total_cost)
            if pure_momentum_result is not None
            else float("nan")
        ),
        "pure_momentum_counterfactual_two_way_turnover": (
            pure_momentum_result.gross_traded_notional
            / pure_momentum_result.starting_capital
            if pure_momentum_result is not None
            else float("nan")
        ),
        "pure_momentum_net_return": pure_same_start_return,
        "actual_minus_pure_momentum_net_return": arithmetic_vs_pure,
        "actual_minus_pure_momentum_pnl": (
            arithmetic_vs_pure * actual_result.starting_capital
        ),
        "arithmetic_agent_incremental_return_vs_pure_momentum": (arithmetic_vs_pure),
        "arithmetic_agent_incremental_pnl_vs_pure_momentum": (
            arithmetic_vs_pure * actual_result.starting_capital
        ),
        "compounded_agent_incremental_return_vs_pure_momentum": (
            (actual_result.ending_capital - pure_ending) / initial_capital
        ),
        "compounded_agent_incremental_pnl_vs_pure_momentum": (
            actual_result.ending_capital - pure_ending
        ),
        "compounded_relative_nav_vs_pure_momentum": (
            actual_result.ending_capital / pure_ending - 1.0
            if pure_ending > 0
            else float("nan")
        ),
        "pure_momentum_weights": (
            {
                code: weight
                for code, weight in pure_momentum_allocation.weights.items()
                if weight > 0
            }
            if isinstance(pure_momentum_allocation, PortfolioParseResult)
            else {}
        ),
    }


def _core_regime_performance(
    monthly: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    performance: dict[str, dict[str, Any]] = {}
    for regime in ("risk_on", "neutral", "defensive"):
        rows = [row for row in monthly if row.get("market_regime") == regime]
        if not rows:
            performance[regime] = {
                "months": 0,
                "net_return": float("nan"),
                "deterministic_counterfactual_return": float("nan"),
                "dynamic_core_cash_dilution_counterfactual_return": float("nan"),
                "pure_momentum_counterfactual_return": float("nan"),
                "index_return": float("nan"),
            }
            continue
        actual = math.prod(1.0 + float(row["net_return"]) for row in rows)
        deterministic_values = [
            float(row["deterministic_counterfactual_net_return"]) for row in rows
        ]
        dilution_values = [
            float(row["dynamic_core_cash_dilution_counterfactual_net_return"])
            for row in rows
        ]
        pure_values = [
            float(row["pure_momentum_counterfactual_net_return"]) for row in rows
        ]
        benchmark = math.prod(1.0 + float(row["index_return"]) for row in rows)
        performance[regime] = {
            "months": len(rows),
            "net_return": actual - 1.0,
            "deterministic_counterfactual_return": (
                math.prod(1.0 + value for value in deterministic_values) - 1.0
                if all(math.isfinite(value) for value in deterministic_values)
                else float("nan")
            ),
            "dynamic_core_cash_dilution_counterfactual_return": (
                math.prod(1.0 + value for value in dilution_values) - 1.0
                if all(math.isfinite(value) for value in dilution_values)
                else float("nan")
            ),
            "pure_momentum_counterfactual_return": (
                math.prod(1.0 + value for value in pure_values) - 1.0
                if all(math.isfinite(value) for value in pure_values)
                else float("nan")
            ),
            "index_return": benchmark - 1.0,
        }
    return performance


def _code_counts(
    monthly: Sequence[Mapping[str, Any]],
    field: str,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in monthly:
        for code in _normalized_codes(row.get(field)):
            counts[code] = counts.get(code, 0) + 1
    return dict(sorted(counts.items()))


def _evaluate_legacy_allocations(
    name: str,
    tasks: Mapping[str, dict[str, Any]],
    allocations: Mapping[str, PortfolioParseResult],
    *,
    initial_capital: float,
    open_cost: float,
    close_cost: float,
    min_cost: float,
    is_oracle: bool = False,
    anchor_policy: AnchorPolicy | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    capital = float(initial_capital)
    benchmark_nav = 1.0
    capitals = [capital]
    monthly: list[dict[str, Any]] = []

    ordered_tasks = sorted(
        tasks.items(), key=lambda item: str(item[1]["metadata"]["entry_date"])
    )
    for task_id, task in ordered_tasks:
        metadata = task["metadata"]
        requested = allocations.get(task_id)
        parse_ok = bool(requested and requested.ok)
        allocation = (
            requested
            if requested is not None and requested.ok
            else cash_allocation(metadata["stock_pool"])
        )
        result = evaluate_portfolio_month(
            allocation.weights,
            allocation.cash,
            metadata["stock_returns"],
            float(metadata["index_return"]),
            starting_capital=capital,
            excess_returns=metadata.get("excess_returns"),
            open_cost=open_cost,
            close_cost=close_cost,
            min_cost=min_cost,
        )
        capital = result.ending_capital
        capitals.append(capital)
        benchmark_nav *= 1.0 + float(metadata["index_return"])
        ranked_contributions = sorted(
            result.contributions.items(),
            key=lambda item: (-item[1], item[0]),
        )
        row = {
            "task_id": task_id,
            "entry_date": metadata["entry_date"],
            "exit_date": metadata["exit_date"],
            "parse_ok": parse_ok,
            "parse_error": (
                ""
                if parse_ok
                else (requested.error if requested is not None else "missing")
            ),
            "weights": {
                code: weight
                for code, weight in allocation.weights.items()
                if weight > 0
            },
            "cash": allocation.cash,
            "starting_capital": result.starting_capital,
            "ending_capital": result.ending_capital,
            "gross_return": result.gross_return,
            "net_return": result.net_return,
            "index_return": result.index_return,
            "active_return": result.active_return,
            "buy_cost": result.buy_cost,
            "sell_cost": result.sell_cost,
            "total_cost": result.total_cost,
            "gross_traded_notional": result.gross_traded_notional,
            "invested_weight": result.invested_weight,
            "holding_count": result.holding_count,
            "concentration_hhi": result.concentration_hhi,
            "weight_rank_ic": result.weight_rank_ic,
            "top_contributor": (
                ranked_contributions[0][0] if ranked_contributions else ""
            ),
            "top_contribution": (
                ranked_contributions[0][1] if ranked_contributions else 0.0
            ),
            "worst_contributor": (
                ranked_contributions[-1][0] if ranked_contributions else ""
            ),
            "worst_contribution": (
                ranked_contributions[-1][1] if ranked_contributions else 0.0
            ),
        }
        row.update(
            _anchor_metrics_for_month(
                metadata,
                allocation,
                starting_capital=result.starting_capital,
                actual_result=result,
                open_cost=open_cost,
                close_cost=close_cost,
                min_cost=min_cost,
                anchor_policy=anchor_policy,
            )
        )
        monthly.append(row)

    net_returns = [float(row["net_return"]) for row in monthly]
    active_returns = [float(row["active_return"]) for row in monthly]
    benchmark_final = initial_capital * benchmark_nav
    anchor_rows = [row for row in monthly if "anchor_net_return" in row]
    anchor_nav = math.prod(1.0 + float(row["anchor_net_return"]) for row in anchor_rows)
    anchor_final = initial_capital * anchor_nav
    regime_performance: dict[str, dict[str, Any]] = {}
    for regime in ("risk_on", "neutral", "defensive"):
        regime_rows = [row for row in anchor_rows if row.get("market_regime") == regime]
        if not regime_rows:
            regime_performance[regime] = {
                "months": 0,
                "net_return": float("nan"),
                "anchor_return": float("nan"),
                "deviation_vs_anchor": float("nan"),
                "index_return": float("nan"),
                "active_vs_index": float("nan"),
            }
            continue
        actual_factor = math.prod(1.0 + float(row["net_return"]) for row in regime_rows)
        anchor_factor = math.prod(
            1.0 + float(row["anchor_net_return"]) for row in regime_rows
        )
        index_factor = math.prod(
            1.0 + float(row["index_return"]) for row in regime_rows
        )
        regime_performance[regime] = {
            "months": len(regime_rows),
            "net_return": actual_factor - 1.0,
            "anchor_return": anchor_factor - 1.0,
            "deviation_vs_anchor": (
                actual_factor / anchor_factor - 1.0
                if anchor_factor > 0
                else float("nan")
            ),
            "index_return": index_factor - 1.0,
            "active_vs_index": (
                actual_factor / index_factor - 1.0 if index_factor > 0 else float("nan")
            ),
        }
    summary = {
        "run": name,
        "oracle": bool(is_oracle),
        "anchor_floor": (
            float(anchor_rows[0]["anchor_floor"]) if anchor_rows else 0.60
        ),
        "months": len(monthly),
        "parsed": sum(bool(row["parse_ok"]) for row in monthly),
        "parse_rate": (
            sum(bool(row["parse_ok"]) for row in monthly) / len(monthly)
            if monthly
            else 0.0
        ),
        "initial_capital": initial_capital,
        "final_capital": capital,
        "net_return": capital / initial_capital - 1.0,
        "benchmark_final_capital": benchmark_final,
        "benchmark_return": benchmark_nav - 1.0,
        "relative_nav_return": (
            capital / benchmark_final - 1.0 if benchmark_final > 0 else float("nan")
        ),
        "annualized_sharpe": _annualized_ratio(net_returns),
        "information_ratio": _annualized_ratio(active_returns),
        "max_drawdown": _max_drawdown(capitals),
        "beat_month_rate": (
            sum(value > 0 for value in active_returns) / len(active_returns)
            if active_returns
            else float("nan")
        ),
        "total_cost": sum(float(row["total_cost"]) for row in monthly),
        "total_traded_notional": sum(
            float(row["gross_traded_notional"]) for row in monthly
        ),
        "average_cash": _mean([float(row["cash"]) for row in monthly]),
        "average_holding_count": _mean(
            [float(row["holding_count"]) for row in monthly]
        ),
        "average_concentration_hhi": _mean(
            [float(row["concentration_hhi"]) for row in monthly]
        ),
        "mean_weight_rank_ic": _finite_mean(
            [float(row["weight_rank_ic"]) for row in monthly]
        ),
        "anchor_compliance_rate": _finite_mean(
            [
                float(row["anchor_compliant"])
                for row in monthly
                if "anchor_compliant" in row
            ]
        ),
        "mean_top4_exposure": _finite_mean(
            [float(row["top4_exposure"]) for row in monthly if "top4_exposure" in row]
        ),
        "mean_top4_overlap_count": _finite_mean(
            [
                float(row["top4_overlap_count"])
                for row in monthly
                if "top4_overlap_count" in row
            ]
        ),
        "mean_replacement_count": _finite_mean(
            [
                float(row["replacement_count"])
                for row in monthly
                if "replacement_count" in row
            ]
        ),
        "anchor_final_capital": anchor_final,
        "anchor_return": anchor_nav - 1.0,
        "cumulative_deviation_vs_anchor": (
            capital / anchor_final - 1.0 if anchor_final > 0 else float("nan")
        ),
        "cumulative_deviation_pnl": capital - anchor_final,
        "breadth_regime_performance": regime_performance,
    }
    return summary, monthly


def _evaluate_core_satellite_allocations(
    name: str,
    tasks: Mapping[str, dict[str, Any]],
    allocations: Mapping[str, PortfolioParseResult],
    *,
    initial_capital: float,
    open_cost: float,
    close_cost: float,
    min_cost: float,
    is_oracle: bool = False,
    anchor_policy: AnchorPolicy | None = None,
    core_contexts: Mapping[str, Mapping[str, Any]] | None = None,
    selection_audits: Mapping[str, Mapping[str, Any]] | None = None,
    run_kind: str = "run",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    policy = anchor_policy or AnchorPolicy(enabled=True, mode="core_satellite")
    if not policy.enabled or policy.mode != "core_satellite":
        raise ValueError("core-satellite evaluation requires an enabled core policy")
    if run_kind not in {"run", "baseline"}:
        raise ValueError("run_kind must be 'run' or 'baseline'")
    contexts = (
        dict(core_contexts)
        if core_contexts is not None
        else build_core_satellite_contexts(tasks, policy=policy)
    )
    audits = selection_audits or {}

    capital = float(initial_capital)
    deterministic_capital = float(initial_capital)
    dilution_capital = float(initial_capital)
    pure_momentum_capital = float(initial_capital)
    benchmark_nav = 1.0
    capitals = [capital]
    deterministic_capitals = [deterministic_capital]
    dilution_capitals = [dilution_capital]
    pure_momentum_capitals = [pure_momentum_capital]
    deterministic_chain_complete = True
    dilution_chain_complete = True
    pure_momentum_chain_complete = True
    monthly: list[dict[str, Any]] = []

    ordered_tasks = sorted(
        tasks.items(), key=lambda item: str(item[1]["metadata"]["entry_date"])
    )
    for task_id, task in ordered_tasks:
        metadata = task["metadata"]
        requested = allocations.get(task_id)
        parse_ok = bool(requested and requested.ok)
        allocation = (
            requested
            if requested is not None and requested.ok
            else cash_allocation(metadata["stock_pool"])
        )
        result = evaluate_portfolio_month(
            allocation.weights,
            allocation.cash,
            metadata["stock_returns"],
            float(metadata["index_return"]),
            starting_capital=capital,
            excess_returns=metadata.get("excess_returns"),
            open_cost=open_cost,
            close_cost=close_cost,
            min_cost=min_cost,
        )
        capital = result.ending_capital
        capitals.append(capital)

        context = contexts.get(
            task_id,
            {
                "error": "missing core-satellite context",
                "top4": [],
                "candidate_codes": [],
                "deterministic_satellites": [],
            },
        )
        deterministic_allocation = context.get("deterministic_allocation")
        dilution_allocation = context.get("dynamic_core_cash_dilution_allocation")
        pure_momentum_allocation = context.get("pure_momentum_allocation")

        deterministic_result = None
        if deterministic_chain_complete:
            deterministic_result = _evaluate_counterfactual(
                deterministic_allocation,
                metadata,
                starting_capital=deterministic_capital,
                open_cost=open_cost,
                close_cost=close_cost,
                min_cost=min_cost,
            )
            if deterministic_result is None:
                deterministic_chain_complete = False
            else:
                deterministic_capital = deterministic_result.ending_capital
                deterministic_capitals.append(deterministic_capital)
        deterministic_at_actual_start = _evaluate_counterfactual(
            deterministic_allocation,
            metadata,
            starting_capital=result.starting_capital,
            open_cost=open_cost,
            close_cost=close_cost,
            min_cost=min_cost,
        )

        dilution_result = None
        if dilution_chain_complete:
            dilution_result = _evaluate_counterfactual(
                dilution_allocation,
                metadata,
                starting_capital=dilution_capital,
                open_cost=open_cost,
                close_cost=close_cost,
                min_cost=min_cost,
            )
            if dilution_result is None:
                dilution_chain_complete = False
            else:
                dilution_capital = dilution_result.ending_capital
                dilution_capitals.append(dilution_capital)
        dilution_at_actual_start = _evaluate_counterfactual(
            dilution_allocation,
            metadata,
            starting_capital=result.starting_capital,
            open_cost=open_cost,
            close_cost=close_cost,
            min_cost=min_cost,
        )

        pure_momentum_result = None
        if pure_momentum_chain_complete:
            pure_momentum_result = _evaluate_counterfactual(
                pure_momentum_allocation,
                metadata,
                starting_capital=pure_momentum_capital,
                open_cost=open_cost,
                close_cost=close_cost,
                min_cost=min_cost,
            )
            if pure_momentum_result is None:
                pure_momentum_chain_complete = False
            else:
                pure_momentum_capital = pure_momentum_result.ending_capital
                pure_momentum_capitals.append(pure_momentum_capital)
        pure_momentum_at_actual_start = _evaluate_counterfactual(
            pure_momentum_allocation,
            metadata,
            starting_capital=result.starting_capital,
            open_cost=open_cost,
            close_cost=close_cost,
            min_cost=min_cost,
        )

        benchmark_nav *= 1.0 + float(metadata["index_return"])
        ranked_contributions = sorted(
            result.contributions.items(),
            key=lambda item: (-item[1], item[0]),
        )
        row: dict[str, Any] = {
            "task_id": task_id,
            "entry_date": metadata["entry_date"],
            "exit_date": metadata["exit_date"],
            "parse_ok": parse_ok,
            "parse_error": (
                ""
                if parse_ok
                else (requested.error if requested is not None else "missing")
            ),
            "weights": {
                code: weight
                for code, weight in allocation.weights.items()
                if weight > 0
            },
            "cash": allocation.cash,
            "starting_capital": result.starting_capital,
            "ending_capital": result.ending_capital,
            "gross_return": result.gross_return,
            "net_return": result.net_return,
            "index_return": result.index_return,
            "active_return": result.active_return,
            "buy_cost": result.buy_cost,
            "sell_cost": result.sell_cost,
            "total_cost": result.total_cost,
            "gross_traded_notional": result.gross_traded_notional,
            "two_way_turnover": (
                result.gross_traded_notional / result.starting_capital
            ),
            "invested_weight": result.invested_weight,
            "holding_count": result.holding_count,
            "concentration_hhi": result.concentration_hhi,
            "weight_rank_ic": result.weight_rank_ic,
            "top_contributor": (
                ranked_contributions[0][0] if ranked_contributions else ""
            ),
            "top_contribution": (
                ranked_contributions[0][1] if ranked_contributions else 0.0
            ),
            "worst_contributor": (
                ranked_contributions[-1][0] if ranked_contributions else ""
            ),
            "worst_contribution": (
                ranked_contributions[-1][1] if ranked_contributions else 0.0
            ),
        }
        row.update(
            _core_satellite_metrics_for_month(
                name=name,
                run_kind=run_kind,
                allocation=allocation,
                actual_result=result,
                metadata=metadata,
                context=context,
                audit=audits.get(task_id),
                policy=policy,
                deterministic_result=deterministic_result,
                deterministic_at_actual_start=deterministic_at_actual_start,
                dilution_result=dilution_result,
                dilution_at_actual_start=dilution_at_actual_start,
                pure_momentum_result=pure_momentum_result,
                pure_momentum_at_actual_start=pure_momentum_at_actual_start,
                initial_capital=initial_capital,
            )
        )
        monthly.append(row)

    months = len(monthly)
    net_returns = [float(row["net_return"]) for row in monthly]
    active_returns = [float(row["active_return"]) for row in monthly]
    benchmark_final = initial_capital * benchmark_nav
    actual_factor = capital / initial_capital
    deterministic_complete = (
        deterministic_chain_complete and len(deterministic_capitals) == months + 1
    )
    dilution_complete = dilution_chain_complete and len(dilution_capitals) == months + 1
    pure_momentum_complete = (
        pure_momentum_chain_complete and len(pure_momentum_capitals) == months + 1
    )
    deterministic_final = (
        deterministic_capital if deterministic_complete else float("nan")
    )
    dilution_final = dilution_capital if dilution_complete else float("nan")
    pure_momentum_final = (
        pure_momentum_capital if pure_momentum_complete else float("nan")
    )
    deterministic_factor = deterministic_final / initial_capital
    dilution_factor = dilution_final / initial_capital
    pure_momentum_factor = pure_momentum_final / initial_capital

    fallback_values = [row.get("selection_fallback") for row in monthly]
    fallback_count = sum(value is True for value in fallback_values)
    fallback_known_count = sum(isinstance(value, bool) for value in fallback_values)
    fallback_unknown_count = months - fallback_known_count
    satellite_wins = [
        bool(row["satellite_excess_win"])
        for row in monthly
        if isinstance(row.get("satellite_excess_win"), bool)
    ]
    arithmetic_incremental_returns = [
        float(row["arithmetic_agent_incremental_return_vs_deterministic"])
        for row in monthly
        if math.isfinite(
            float(row["arithmetic_agent_incremental_return_vs_deterministic"])
        )
    ]
    arithmetic_incremental_pnl = [
        float(row["arithmetic_agent_incremental_pnl_vs_deterministic"])
        for row in monthly
        if math.isfinite(
            float(row["arithmetic_agent_incremental_pnl_vs_deterministic"])
        )
    ]
    arithmetic_vs_dilution_returns = [
        float(row["arithmetic_agent_incremental_return_vs_dynamic_core_cash_dilution"])
        for row in monthly
        if math.isfinite(
            float(
                row["arithmetic_agent_incremental_return_vs_dynamic_core_cash_dilution"]
            )
        )
    ]
    arithmetic_vs_dilution_pnl = [
        float(row["arithmetic_agent_incremental_pnl_vs_dynamic_core_cash_dilution"])
        for row in monthly
        if math.isfinite(
            float(row["arithmetic_agent_incremental_pnl_vs_dynamic_core_cash_dilution"])
        )
    ]
    arithmetic_satellite_vs_dilution_returns = [
        float(row["arithmetic_satellite_incremental_return_vs_dilution"])
        for row in monthly
        if math.isfinite(
            float(row["arithmetic_satellite_incremental_return_vs_dilution"])
        )
    ]
    arithmetic_satellite_vs_dilution_pnl = [
        float(row["arithmetic_satellite_incremental_pnl_vs_dilution"])
        for row in monthly
        if math.isfinite(float(row["arithmetic_satellite_incremental_pnl_vs_dilution"]))
    ]
    paired_dilution_months = sum(
        bool(row["paired_satellite_dilution_comparison_valid"]) for row in monthly
    )
    paired_dilution_complete = months > 0 and paired_dilution_months == months
    arithmetic_vs_pure_returns = [
        float(row["arithmetic_agent_incremental_return_vs_pure_momentum"])
        for row in monthly
        if math.isfinite(
            float(row["arithmetic_agent_incremental_return_vs_pure_momentum"])
        )
    ]
    arithmetic_vs_pure_pnl = [
        float(row["arithmetic_agent_incremental_pnl_vs_pure_momentum"])
        for row in monthly
        if math.isfinite(
            float(row["arithmetic_agent_incremental_pnl_vs_pure_momentum"])
        )
    ]
    compounded_vs_deterministic_return = (
        actual_factor - deterministic_factor
        if math.isfinite(deterministic_factor)
        else float("nan")
    )
    compounded_vs_deterministic_pnl = (
        capital - deterministic_final
        if math.isfinite(deterministic_final)
        else float("nan")
    )
    compounded_vs_dilution_return = (
        actual_factor - dilution_factor
        if math.isfinite(dilution_factor)
        else float("nan")
    )
    compounded_vs_dilution_pnl = (
        capital - dilution_final if math.isfinite(dilution_final) else float("nan")
    )
    compounded_vs_pure_return = (
        actual_factor - pure_momentum_factor
        if math.isfinite(pure_momentum_factor)
        else float("nan")
    )
    compounded_vs_pure_pnl = (
        capital - pure_momentum_final
        if math.isfinite(pure_momentum_final)
        else float("nan")
    )

    summary: dict[str, Any] = {
        "run": name,
        "run_kind": run_kind,
        "oracle": bool(is_oracle),
        "policy_mode": "core_satellite",
        "registered_random_satellite_seeds": list(RANDOM_SATELLITE_SEEDS),
        "random_satellite_seed": next(
            (
                seed
                for seed in RANDOM_SATELLITE_SEEDS
                if name == _random_satellite_baseline_name(seed)
            ),
            None,
        ),
        "months": months,
        "parsed": sum(bool(row["parse_ok"]) for row in monthly),
        "parse_rate": (
            sum(bool(row["parse_ok"]) for row in monthly) / months if months else 0.0
        ),
        "initial_capital": initial_capital,
        "final_capital": capital,
        "net_return": actual_factor - 1.0,
        "benchmark_final_capital": benchmark_final,
        "benchmark_return": benchmark_nav - 1.0,
        "relative_nav_return": (
            capital / benchmark_final - 1.0 if benchmark_final > 0 else float("nan")
        ),
        "annualized_net_return": _annualized_compound_return(
            actual_factor,
            months,
        ),
        "annualized_excess_return_vs_csi300": _annualized_excess_return(
            actual_factor,
            benchmark_nav,
            months,
        ),
        "annualized_sharpe": _annualized_ratio(net_returns),
        "information_ratio": _annualized_ratio(active_returns),
        "max_drawdown": _max_drawdown(capitals),
        "beat_month_rate": (
            sum(value > 0 for value in active_returns) / len(active_returns)
            if active_returns
            else float("nan")
        ),
        "total_cost": sum(float(row["total_cost"]) for row in monthly),
        "total_traded_notional": sum(
            float(row["gross_traded_notional"]) for row in monthly
        ),
        "average_two_way_turnover": _mean(
            [float(row["two_way_turnover"]) for row in monthly]
        ),
        "average_cash": _mean([float(row["cash"]) for row in monthly]),
        "average_holding_count": _mean(
            [float(row["holding_count"]) for row in monthly]
        ),
        "average_concentration_hhi": _mean(
            [float(row["concentration_hhi"]) for row in monthly]
        ),
        "mean_weight_rank_ic": _finite_mean(
            [float(row["weight_rank_ic"]) for row in monthly]
        ),
        "core_satellite_compliance_rate": _finite_mean(
            [float(row["core_satellite_compliant"]) for row in monthly]
        ),
        "dynamic_core_cash_dilution_compliance_rate": _finite_mean(
            [float(row["dynamic_core_cash_dilution_compliant"]) for row in monthly]
        ),
        "mandatory_top4_compliance_rate": _finite_mean(
            [float(row["mandatory_top4_compliant"]) for row in monthly]
        ),
        "core_weight_compliance_rate": _finite_mean(
            [float(row["core_weight_compliant"]) for row in monthly]
        ),
        "market_regime_counts": {
            regime: sum(row.get("market_regime") == regime for row in monthly)
            for regime in ("risk_on", "neutral", "defensive")
        },
        "average_core_weight": _mean([float(row["core_weight"]) for row in monthly]),
        "average_satellite_weight": _mean(
            [float(row["satellite_weight"]) for row in monthly]
        ),
        "average_core_satellite_cash_weight": _mean(
            [float(row["cash_weight"]) for row in monthly]
        ),
        "selected_satellite_counts": _code_counts(
            monthly,
            "selected_satellites",
        ),
        "deterministic_satellite_counts": _code_counts(
            monthly,
            "deterministic_satellites",
        ),
        "mean_satellite_gross_contribution": _finite_mean(
            [float(row["satellite_gross_contribution"]) for row in monthly]
        ),
        "mean_satellite_net_contribution": _finite_mean(
            [float(row["satellite_net_contribution"]) for row in monthly]
        ),
        "total_satellite_gross_pnl": sum(
            float(row["satellite_gross_pnl"])
            for row in monthly
            if math.isfinite(float(row["satellite_gross_pnl"]))
        ),
        "total_satellite_net_pnl": sum(
            float(row["satellite_net_pnl"])
            for row in monthly
            if math.isfinite(float(row["satellite_net_pnl"]))
        ),
        "satellite_excess_win_count": sum(satellite_wins),
        "satellite_excess_win_rate": (
            sum(satellite_wins) / len(satellite_wins)
            if satellite_wins
            else float("nan")
        ),
        "fallback_count": fallback_count,
        "fallback_known_count": fallback_known_count,
        "fallback_unknown_count": fallback_unknown_count,
        "fallback_rate": (
            fallback_count / fallback_known_count
            if fallback_known_count
            else float("nan")
        ),
        "agent_selection_count": sum(
            row.get("selection_source") == "agent" for row in monthly
        ),
        "deterministic_fallback_source_count": sum(
            row.get("selection_source") == "deterministic_fallback" for row in monthly
        ),
        "selection_source_unknown_count": sum(
            row.get("selection_source") == "unknown" for row in monthly
        ),
        "selection_audit_present_count": sum(
            bool(row.get("selection_audit_present")) for row in monthly
        ),
        "core_context_error_count": sum(
            not bool(row.get("core_satellite_context_ok")) for row in monthly
        ),
        "deterministic_counterfactual_months": len(deterministic_capitals) - 1,
        "deterministic_counterfactual_complete": deterministic_complete,
        "deterministic_counterfactual_final_capital": deterministic_final,
        "deterministic_counterfactual_return": deterministic_factor - 1.0,
        "deterministic_counterfactual_annualized_excess_vs_csi300": (
            _annualized_excess_return(
                deterministic_factor,
                benchmark_nav,
                months,
            )
        ),
        "deterministic_counterfactual_max_drawdown": (
            _max_drawdown(deterministic_capitals)
            if deterministic_complete
            else float("nan")
        ),
        "deterministic_counterfactual_average_two_way_turnover": (
            _finite_mean(
                [
                    float(row["deterministic_counterfactual_two_way_turnover"])
                    for row in monthly
                ]
            )
        ),
        "deterministic_counterfactual_total_cost": sum(
            float(row["deterministic_counterfactual_total_cost"])
            for row in monthly
            if math.isfinite(float(row["deterministic_counterfactual_total_cost"]))
        ),
        "arithmetic_agent_incremental_return_vs_deterministic_sum": sum(
            arithmetic_incremental_returns
        ),
        "arithmetic_agent_incremental_pnl_vs_deterministic_sum": sum(
            arithmetic_incremental_pnl
        ),
        "compounded_agent_incremental_return_vs_deterministic": (
            compounded_vs_deterministic_return
        ),
        "compounded_agent_incremental_pnl_vs_deterministic": (
            compounded_vs_deterministic_pnl
        ),
        "compounded_relative_nav_vs_deterministic": (
            capital / deterministic_final - 1.0
            if deterministic_final > 0
            else float("nan")
        ),
        "agent_incremental_return_vs_deterministic": (
            compounded_vs_deterministic_return
        ),
        "agent_incremental_pnl_vs_deterministic": (compounded_vs_deterministic_pnl),
        "dynamic_core_cash_dilution_counterfactual_months": (
            len(dilution_capitals) - 1
        ),
        "dynamic_core_cash_dilution_counterfactual_complete": (dilution_complete),
        "dynamic_core_cash_dilution_counterfactual_final_capital": (dilution_final),
        "dynamic_core_cash_dilution_counterfactual_return": (dilution_factor - 1.0),
        "dynamic_core_cash_dilution_counterfactual_annualized_excess_vs_csi300": (
            _annualized_excess_return(
                dilution_factor,
                benchmark_nav,
                months,
            )
        ),
        "dynamic_core_cash_dilution_counterfactual_max_drawdown": (
            _max_drawdown(dilution_capitals) if dilution_complete else float("nan")
        ),
        "dynamic_core_cash_dilution_counterfactual_average_two_way_turnover": (
            _finite_mean(
                [
                    float(
                        row[
                            "dynamic_core_cash_dilution_counterfactual_two_way_turnover"
                        ]
                    )
                    for row in monthly
                ]
            )
        ),
        "dynamic_core_cash_dilution_counterfactual_total_cost": sum(
            float(row["dynamic_core_cash_dilution_counterfactual_total_cost"])
            for row in monthly
            if math.isfinite(
                float(row["dynamic_core_cash_dilution_counterfactual_total_cost"])
            )
        ),
        "arithmetic_agent_incremental_return_vs_dynamic_core_cash_dilution_sum": (
            sum(arithmetic_vs_dilution_returns)
        ),
        "arithmetic_agent_incremental_pnl_vs_dynamic_core_cash_dilution_sum": (
            sum(arithmetic_vs_dilution_pnl)
        ),
        "compounded_agent_incremental_return_vs_dynamic_core_cash_dilution": (
            compounded_vs_dilution_return
        ),
        "compounded_agent_incremental_pnl_vs_dynamic_core_cash_dilution": (
            compounded_vs_dilution_pnl
        ),
        "compounded_relative_nav_vs_dynamic_core_cash_dilution": (
            capital / dilution_final - 1.0 if dilution_final > 0 else float("nan")
        ),
        "paired_satellite_dilution_comparison_months": (paired_dilution_months),
        "paired_satellite_dilution_comparison_complete": (paired_dilution_complete),
        "arithmetic_satellite_incremental_return_vs_dilution_sum": sum(
            arithmetic_satellite_vs_dilution_returns
        )
        if paired_dilution_months
        else float("nan"),
        "arithmetic_satellite_incremental_pnl_vs_dilution_sum": (
            sum(arithmetic_satellite_vs_dilution_pnl)
            if paired_dilution_months
            else float("nan")
        ),
        "compounded_satellite_incremental_return_vs_dilution": (
            compounded_vs_dilution_return if paired_dilution_complete else float("nan")
        ),
        "compounded_satellite_incremental_pnl_vs_dilution": (
            compounded_vs_dilution_pnl if paired_dilution_complete else float("nan")
        ),
        "pure_momentum_counterfactual_months": len(pure_momentum_capitals) - 1,
        "pure_momentum_counterfactual_complete": pure_momentum_complete,
        "pure_momentum_counterfactual_final_capital": pure_momentum_final,
        "pure_momentum_counterfactual_return": pure_momentum_factor - 1.0,
        "pure_momentum_counterfactual_annualized_excess_vs_csi300": (
            _annualized_excess_return(
                pure_momentum_factor,
                benchmark_nav,
                months,
            )
        ),
        "pure_momentum_counterfactual_max_drawdown": (
            _max_drawdown(pure_momentum_capitals)
            if pure_momentum_complete
            else float("nan")
        ),
        "pure_momentum_counterfactual_average_two_way_turnover": (
            _finite_mean(
                [
                    float(row["pure_momentum_counterfactual_two_way_turnover"])
                    for row in monthly
                ]
            )
        ),
        "pure_momentum_counterfactual_total_cost": sum(
            float(row["pure_momentum_counterfactual_total_cost"])
            for row in monthly
            if math.isfinite(float(row["pure_momentum_counterfactual_total_cost"]))
        ),
        "arithmetic_agent_incremental_return_vs_pure_momentum_sum": sum(
            arithmetic_vs_pure_returns
        ),
        "arithmetic_agent_incremental_pnl_vs_pure_momentum_sum": sum(
            arithmetic_vs_pure_pnl
        ),
        "compounded_agent_incremental_return_vs_pure_momentum": (
            compounded_vs_pure_return
        ),
        "compounded_agent_incremental_pnl_vs_pure_momentum": (compounded_vs_pure_pnl),
        "compounded_relative_nav_vs_pure_momentum": (
            capital / pure_momentum_final - 1.0
            if pure_momentum_final > 0
            else float("nan")
        ),
        "breadth_regime_performance": _core_regime_performance(monthly),
    }
    summary["core_satellite_ablation"] = {
        "deterministic_counterfactual": {
            "final_capital": deterministic_final,
            "return": summary["deterministic_counterfactual_return"],
            "annualized_excess_vs_csi300": summary[
                "deterministic_counterfactual_annualized_excess_vs_csi300"
            ],
            "max_drawdown": summary["deterministic_counterfactual_max_drawdown"],
            "average_two_way_turnover": summary[
                "deterministic_counterfactual_average_two_way_turnover"
            ],
        },
        "dynamic_core_cash_dilution_counterfactual": {
            "final_capital": dilution_final,
            "return": summary["dynamic_core_cash_dilution_counterfactual_return"],
            "annualized_excess_vs_csi300": summary[
                "dynamic_core_cash_dilution_counterfactual_annualized_excess_vs_csi300"
            ],
            "max_drawdown": summary[
                "dynamic_core_cash_dilution_counterfactual_max_drawdown"
            ],
            "average_two_way_turnover": summary[
                "dynamic_core_cash_dilution_counterfactual_average_two_way_turnover"
            ],
        },
        "pure_momentum_counterfactual": {
            "final_capital": pure_momentum_final,
            "return": summary["pure_momentum_counterfactual_return"],
            "annualized_excess_vs_csi300": summary[
                "pure_momentum_counterfactual_annualized_excess_vs_csi300"
            ],
            "max_drawdown": summary["pure_momentum_counterfactual_max_drawdown"],
        },
        "agent_vs_deterministic": {
            "arithmetic_monthly_return_difference_sum": summary[
                "arithmetic_agent_incremental_return_vs_deterministic_sum"
            ],
            "arithmetic_monthly_pnl_difference_sum": summary[
                "arithmetic_agent_incremental_pnl_vs_deterministic_sum"
            ],
            "compounded_return_difference": (compounded_vs_deterministic_return),
            "compounded_pnl_difference": compounded_vs_deterministic_pnl,
        },
        "satellite_vs_dynamic_core_cash_dilution": {
            "paired_comparison_complete": paired_dilution_complete,
            "arithmetic_monthly_return_difference_sum": summary[
                "arithmetic_satellite_incremental_return_vs_dilution_sum"
            ],
            "arithmetic_monthly_pnl_difference_sum": summary[
                "arithmetic_satellite_incremental_pnl_vs_dilution_sum"
            ],
            "compounded_return_difference": summary[
                "compounded_satellite_incremental_return_vs_dilution"
            ],
            "compounded_pnl_difference": summary[
                "compounded_satellite_incremental_pnl_vs_dilution"
            ],
        },
    }
    return summary, monthly


def evaluate_allocations(
    name: str,
    tasks: Mapping[str, dict[str, Any]],
    allocations: Mapping[str, PortfolioParseResult],
    *,
    initial_capital: float,
    open_cost: float,
    close_cost: float,
    min_cost: float,
    is_oracle: bool = False,
    anchor_policy: AnchorPolicy | None = None,
    policy_mode: str = "legacy",
    core_contexts: Mapping[str, Mapping[str, Any]] | None = None,
    selection_audits: Mapping[str, Mapping[str, Any]] | None = None,
    run_kind: str = "run",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate legacy allocations or the core-satellite ablation."""
    if policy_mode == "legacy":
        return _evaluate_legacy_allocations(
            name,
            tasks,
            allocations,
            initial_capital=initial_capital,
            open_cost=open_cost,
            close_cost=close_cost,
            min_cost=min_cost,
            is_oracle=is_oracle,
            anchor_policy=anchor_policy,
        )
    if policy_mode == "core_satellite":
        return _evaluate_core_satellite_allocations(
            name,
            tasks,
            allocations,
            initial_capital=initial_capital,
            open_cost=open_cost,
            close_cost=close_cost,
            min_cost=min_cost,
            is_oracle=is_oracle,
            anchor_policy=anchor_policy,
            core_contexts=core_contexts,
            selection_audits=selection_audits,
            run_kind=run_kind,
        )
    raise ValueError("policy_mode must be 'legacy' or 'core_satellite'")


def _fixed_allocation(
    pool: list[str],
    selected: list[str],
) -> PortfolioParseResult:
    if not selected:
        return cash_allocation(pool)
    weight = min(0.25, 1.0 / len(selected))
    weights = {code: (weight if code in selected else 0.0) for code in pool}
    cash = 1.0 - sum(weights.values())
    return PortfolioParseResult(weights, cash, True)


def build_baseline_allocations(
    tasks: Mapping[str, dict[str, Any]],
    *,
    policy_mode: str = "legacy",
    core_contexts: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[tuple[str, dict[str, PortfolioParseResult], bool]]:
    if policy_mode not in {"legacy", "core_satellite"}:
        raise ValueError("policy_mode must be 'legacy' or 'core_satellite'")
    contexts = dict(core_contexts or {}) if policy_mode == "core_satellite" else {}
    if policy_mode == "core_satellite" and not contexts:
        contexts = build_core_satellite_contexts(tasks)
    qlib = pd.read_csv(DATA_DIR / "qlib_signal.csv", dtype={"entry_date": str})
    baselines: dict[str, dict[str, PortfolioParseResult]] = {
        "cash(全现金)": {},
        "equal_weight(16股等权)": {},
        "momentum_top4(20日相对动量)": {},
        "qlib_top4(逐月walk-forward)": {},
        "oracle_top4(事后上界,不可交易)": {},
    }
    if policy_mode == "core_satellite":
        baselines[CORE_SATELLITE_BASELINE] = {}
        baselines[DYNAMIC_CORE_CASH_DILUTION_BASELINE] = {}
        for seed in RANDOM_SATELLITE_SEEDS:
            baselines[_random_satellite_baseline_name(seed)] = {}
        baselines[ORACLE_SATELLITE_BASELINE] = {}

    for task_id, task in tasks.items():
        metadata = task["metadata"]
        pool = list(metadata["stock_pool"])
        baselines["cash(全现金)"][task_id] = cash_allocation(pool)
        baselines["equal_weight(16股等权)"][task_id] = _fixed_allocation(pool, pool)

        context = contexts.get(task_id, {})
        pure_momentum = context.get("pure_momentum_allocation")
        if (
            policy_mode == "core_satellite"
            and isinstance(pure_momentum, PortfolioParseResult)
            and pure_momentum.ok
        ):
            baselines["momentum_top4(20日相对动量)"][task_id] = pure_momentum
        else:
            features = compute_trader_feature_rows(
                metadata["entry_date"],
                metadata["stock_info"],
                data_dir=DATA_DIR,
                lookback_days=250,
            )
            momentum_order = [
                row["ts_code"]
                for row in sorted(
                    features,
                    key=lambda row: (
                        -float(row["rel20"])
                        if row.get("rel20") not in ("", None)
                        else float("inf"),
                        row["ts_code"],
                    ),
                )
            ]
            baselines["momentum_top4(20日相对动量)"][task_id] = _fixed_allocation(
                pool, momentum_order[:4]
            )

        # Some trader entries are shifted to the previous portfolio's
        # liquidation close to prevent capital overlap.  Use the latest
        # walk-forward signal already available by that adjusted entry date.
        available_signal = qlib[
            (qlib["entry_date"] <= str(metadata["entry_date"]))
            & qlib["ts_code"].isin(pool)
        ].sort_values(["entry_date", "ts_code"])
        latest_signal = available_signal.groupby("ts_code", as_index=False).tail(1)
        qlib_order = latest_signal.sort_values(["rank", "ts_code"])["ts_code"].tolist()
        baselines["qlib_top4(逐月walk-forward)"][task_id] = _fixed_allocation(
            pool, qlib_order[:4]
        )
        baselines["oracle_top4(事后上界,不可交易)"][task_id] = _fixed_allocation(
            pool, list(metadata["ground_truth_rank"])[:4]
        )
        if policy_mode == "core_satellite":
            deterministic = context.get("deterministic_allocation")
            if isinstance(deterministic, PortfolioParseResult) and deterministic.ok:
                baselines[CORE_SATELLITE_BASELINE][task_id] = deterministic
            else:
                baselines[CORE_SATELLITE_BASELINE][task_id] = _invalid_allocation(
                    pool,
                    str(context.get("error") or "missing PIT candidates"),
                )
            dilution = context.get("dynamic_core_cash_dilution_allocation")
            if isinstance(dilution, PortfolioParseResult) and dilution.ok:
                baselines[DYNAMIC_CORE_CASH_DILUTION_BASELINE][task_id] = dilution
            else:
                baselines[DYNAMIC_CORE_CASH_DILUTION_BASELINE][task_id] = (
                    _invalid_allocation(
                        pool,
                        str(context.get("error") or "missing dilution sleeve"),
                    )
                )

            assembled = context.get("assembled_deterministic")
            candidates = [str(code) for code in context.get("candidate_codes", [])]
            satellite_count = (
                int(assembled.get("satellite_count", 0))
                if isinstance(assembled, Mapping)
                else 0
            )
            for seed in RANDOM_SATELLITE_SEEDS:
                label = _random_satellite_baseline_name(seed)
                try:
                    selected = _random_satellite_codes(
                        task_id,
                        candidates,
                        satellite_count,
                        seed,
                    )
                    baselines[label][task_id] = (
                        _core_satellite_allocation_for_selection(
                            context,
                            pool,
                            selected,
                        )
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    baselines[label][task_id] = _invalid_allocation(
                        pool,
                        str(exc),
                    )

            try:
                oracle_codes = sorted(
                    candidates,
                    key=lambda code: (
                        -float(metadata["stock_returns"][code]),
                        code,
                    ),
                )[:satellite_count]
                baselines[ORACLE_SATELLITE_BASELINE][task_id] = (
                    _core_satellite_allocation_for_selection(
                        context,
                        pool,
                        oracle_codes,
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                baselines[ORACLE_SATELLITE_BASELINE][task_id] = _invalid_allocation(
                    pool, str(exc)
                )

    return [
        (name, allocation, name.startswith("oracle_"))
        for name, allocation in baselines.items()
    ]


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _pct(value: float) -> str:
    return "-" if not math.isfinite(value) else f"{value * 100:.2f}%"


def _number(value: float, digits: int = 2) -> str:
    return "-" if not math.isfinite(value) else f"{value:.{digits}f}"


def _render_legacy_report(
    evaluated: list[tuple[dict[str, Any], list[dict[str, Any]]]],
    *,
    initial_capital: float,
    open_cost: float,
    close_cost: float,
    min_cost: float,
    anchor_floor: float = 0.60,
) -> str:
    lines = [
        "# A股统一交易员组合回测",
        "",
        (
            "口径：每月计划调仓；若与前一20日窗口重叠则顺延到前次平仓收盘，"
            "持有20个交易日后全部卖出；"
            f"初始资金 ¥{initial_capital:,.2f}，买入费 {open_cost:.3%}，"
            f"卖出费 {close_cost:.3%}，每笔最低 ¥{min_cost:.2f}，现金收益 0。"
        ),
        "无效或缺失仓位按全现金执行；oracle 仅是事后上界，不是可交易基线。",
        f"硬锚复核口径：rel20 top4 总仓位下限 {anchor_floor:.0%}。",
        "",
        "| 运行 | 合法月 | 硬锚合规 | 平均top4 | 平均重合 | 平均替换 | 相对锚点 | 偏离损益 | 最终资产 | 净收益 | 沪深300 | 相对净值 | Sharpe | 信息比 | 最大回撤 | 跑赢月 | 总费用 | 平均现金 | 平均持仓 | 权重RankIC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary, _ in evaluated:
        label = summary["run"] + (" [oracle]" if summary["oracle"] else "")
        lines.append(
            f"| {label} | {summary['parsed']}/{summary['months']} | "
            f"{_pct(summary.get('anchor_compliance_rate', float('nan')))} | "
            f"{_pct(summary.get('mean_top4_exposure', float('nan')))} | "
            f"{_number(summary.get('mean_top4_overlap_count', float('nan')), 2)} | "
            f"{_number(summary.get('mean_replacement_count', float('nan')), 2)} | "
            f"{_pct(summary.get('cumulative_deviation_vs_anchor', float('nan')))} | "
            f"¥{summary.get('cumulative_deviation_pnl', float('nan')):,.2f} | "
            f"¥{summary['final_capital']:,.2f} | {_pct(summary['net_return'])} | "
            f"{_pct(summary['benchmark_return'])} | "
            f"{_pct(summary['relative_nav_return'])} | "
            f"{_number(summary['annualized_sharpe'])} | "
            f"{_number(summary['information_ratio'])} | "
            f"{_pct(summary['max_drawdown'])} | "
            f"{_pct(summary['beat_month_rate'])} | "
            f"¥{summary['total_cost']:,.2f} | {_pct(summary['average_cash'])} | "
            f"{summary['average_holding_count']:.1f} | "
            f"{_number(summary['mean_weight_rank_ic'], 3)} |"
        )

    for summary, _ in evaluated:
        lines.extend(
            [
                "",
                f"## {summary['run']} 各市场广度状态",
                "",
                "| 状态 | 月数 | 实际净收益 | 动量锚点 | 相对锚点 | 沪深300 | 相对指数 |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        performance = summary.get("breadth_regime_performance", {})
        for regime in ("risk_on", "neutral", "defensive"):
            values = performance.get(regime, {})
            lines.append(
                f"| {regime} | {int(values.get('months', 0))} | "
                f"{_pct(float(values.get('net_return', float('nan'))))} | "
                f"{_pct(float(values.get('anchor_return', float('nan'))))} | "
                f"{_pct(float(values.get('deviation_vs_anchor', float('nan'))))} | "
                f"{_pct(float(values.get('index_return', float('nan'))))} | "
                f"{_pct(float(values.get('active_vs_index', float('nan'))))} |"
            )

    for summary, monthly in evaluated:
        lines.extend(
            [
                "",
                f"## {summary['run']} 月度明细",
                "",
                "| 买入日 | 卖出日 | 格式 | 硬锚 | top4暴露 | 重合数 | 替换数 | 市场状态 | 偏离锚点 | 偏离损益 | 股票仓位 | 现金 | 净收益 | 沪深300 | 主动收益 | 费用 | 期末资产 | 最大贡献 | 最大拖累 |",
                "|---|---|---|---|---:|---:|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|---|",
            ]
        )
        for row in monthly:
            positions = (
                ",".join(
                    f"{code}:{weight:.3f}" for code, weight in row["weights"].items()
                )
                or "-"
            )
            status = "OK" if row["parse_ok"] else "现金回退"
            lines.append(
                f"| {row['entry_date']} | {row['exit_date']} | {status} | "
                f"{'OK' if row.get('anchor_compliant') else '违规'} | "
                f"{_pct(float(row.get('top4_exposure', float('nan'))))} | "
                f"{row.get('top4_overlap_count', '-')} | "
                f"{row.get('replacement_count', '-')} | "
                f"{row.get('market_regime', '-')} | "
                f"{_pct(float(row.get('deviation_active_return', float('nan'))))} | "
                f"¥{float(row.get('deviation_pnl', float('nan'))):,.2f} | "
                f"{positions} | {_pct(float(row['cash']))} | "
                f"{_pct(float(row['net_return']))} | "
                f"{_pct(float(row['index_return']))} | "
                f"{_pct(float(row['active_return']))} | "
                f"¥{float(row['total_cost']):,.2f} | "
                f"¥{float(row['ending_capital']):,.2f} | "
                f"{row['top_contributor']} | {row['worst_contributor']} |"
            )
            if not row["parse_ok"]:
                lines.append(f"\n格式错误：`{row['parse_error']}`")
    lines.append("")
    return "\n".join(lines)


def _money(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    return "-" if not math.isfinite(number) else f"¥{number:,.2f}"


def _codes_cell(value: Any) -> str:
    codes = _normalized_codes(value)
    return ",".join(codes) if codes else "-"


def _fallback_cell(value: Any) -> str:
    return "是" if value is True else "否" if value is False else "未知"


def _markdown_cell(value: Any) -> str:
    return str(value).replace("|", r"\|").replace("\n", " ")


def _render_core_satellite_report(
    evaluated: list[tuple[dict[str, Any], list[dict[str, Any]]]],
    *,
    initial_capital: float,
    open_cost: float,
    close_cost: float,
    min_cost: float,
) -> str:
    lines = [
        "# A股核心-卫星消融回测",
        "",
        (
            "口径：每期从同一初始资金顺序复利；实际、PIT超额模型确定性组合、"
            "动态核心+现金稀释与纯rel20动量top4均独立延续净值。"
        ),
        (
            f"初始资金 ¥{initial_capital:,.2f}，买入费 {open_cost:.3%}，"
            f"卖出费 {close_cost:.3%}，每笔最低 ¥{min_cost:.2f}。"
        ),
        (
            "月度 arithmetic 差值在实际当月期初资金上重算两组合；"
            "compounded 差值比较从相同初始资金顺序复利后的净值，二者不混用。"
        ),
        (
            "`--anchor-floor` 在 core_satellite 模式不参与计算；"
            "每月重建严格点时动量快照并截取最多6个PIT超额候选。"
        ),
        (
            "dynamic_core_cash_dilution 保留固定核心、将全部卫星袖套移入现金："
            "risk_on=80/0/20，neutral=90/0/10，defensive=90/0/10；"
            "它按设计不满足完整核心-卫星约束。"
        ),
        (
            "随机卫星仅从同一PIT top-6候选抽样，固定种子="
            + ",".join(str(seed) for seed in RANDOM_SATELLITE_SEEDS)
            + "；oracle卫星也限定于该top-6，但使用实现收益，仅为不可交易上界。"
        ),
        "",
        "## Core-satellite ablation",
        "",
        "| 运行 | 合法月 | 核心卫星合规 | top4必选 | 平均核心/卫星/现金 | 最终资产 | 净收益 | 年化超额/沪深300 | 最大回撤 | 双向换手 | 卫星超额胜率 | 回退(次数/已知率/未知) | 确定性终值 | 确定性收益 | compounded增量收益 | compounded增量损益 | 纯动量终值 | 纯动量收益 |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for summary, _ in evaluated:
        label = summary["run"] + (" [oracle]" if summary["oracle"] else "")
        sleeve = (
            f"{_pct(float(summary['average_core_weight']))}/"
            f"{_pct(float(summary['average_satellite_weight']))}/"
            f"{_pct(float(summary['average_core_satellite_cash_weight']))}"
        )
        fallback = (
            f"{int(summary['fallback_count'])}/"
            f"{_pct(float(summary['fallback_rate']))}/"
            f"{int(summary['fallback_unknown_count'])}"
        )
        lines.append(
            f"| {_markdown_cell(label)} | "
            f"{summary['parsed']}/{summary['months']} | "
            f"{_pct(float(summary['core_satellite_compliance_rate']))} | "
            f"{_pct(float(summary['mandatory_top4_compliance_rate']))} | "
            f"{sleeve} | {_money(summary['final_capital'])} | "
            f"{_pct(float(summary['net_return']))} | "
            f"{_pct(float(summary['annualized_excess_return_vs_csi300']))} | "
            f"{_pct(float(summary['max_drawdown']))} | "
            f"{_pct(float(summary['average_two_way_turnover']))} | "
            f"{_pct(float(summary['satellite_excess_win_rate']))} | "
            f"{fallback} | "
            f"{_money(summary['deterministic_counterfactual_final_capital'])} | "
            f"{_pct(float(summary['deterministic_counterfactual_return']))} | "
            f"{_pct(float(summary['compounded_agent_incremental_return_vs_deterministic']))} | "
            f"{_money(summary['compounded_agent_incremental_pnl_vs_deterministic'])} | "
            f"{_money(summary['pure_momentum_counterfactual_final_capital'])} | "
            f"{_pct(float(summary['pure_momentum_counterfactual_return']))} |"
        )

    lines.extend(
        [
            "",
            "## Satellite sleeve vs dynamic core+cash dilution",
            "",
            (
                "arithmetic 使用实际组合当月期初资金同起点重算；"
                "compounded 使用各自顺序复利净值。配对卫星列仅在实际组合为"
                "合法核心-卫星或稀释臂时有定义。"
            ),
            "",
            "| 运行 | 稀释臂匹配 | 稀释终值 | 稀释收益 | arithmetic实际-稀释 | arithmetic实际-稀释损益 | compounded实际-稀释 | compounded实际-稀释损益 | 配对完整 | arithmetic卫星增量 | arithmetic卫星增量损益 | compounded卫星增量 | compounded卫星增量损益 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|",
        ]
    )
    for summary, _ in evaluated:
        label = summary["run"] + (" [oracle]" if summary["oracle"] else "")
        lines.append(
            f"| {_markdown_cell(label)} | "
            f"{_pct(float(summary['dynamic_core_cash_dilution_compliance_rate']))} | "
            f"{_money(summary['dynamic_core_cash_dilution_counterfactual_final_capital'])} | "
            f"{_pct(float(summary['dynamic_core_cash_dilution_counterfactual_return']))} | "
            f"{_pct(float(summary['arithmetic_agent_incremental_return_vs_dynamic_core_cash_dilution_sum']))} | "
            f"{_money(summary['arithmetic_agent_incremental_pnl_vs_dynamic_core_cash_dilution_sum'])} | "
            f"{_pct(float(summary['compounded_agent_incremental_return_vs_dynamic_core_cash_dilution']))} | "
            f"{_money(summary['compounded_agent_incremental_pnl_vs_dynamic_core_cash_dilution'])} | "
            f"{'是' if summary['paired_satellite_dilution_comparison_complete'] else '否'} | "
            f"{_pct(float(summary['arithmetic_satellite_incremental_return_vs_dilution_sum']))} | "
            f"{_money(summary['arithmetic_satellite_incremental_pnl_vs_dilution_sum'])} | "
            f"{_pct(float(summary['compounded_satellite_incremental_return_vs_dilution']))} | "
            f"{_money(summary['compounded_satellite_incremental_pnl_vs_dilution'])} |"
        )

    for summary, monthly in evaluated:
        lines.extend(
            [
                "",
                f"## {_markdown_cell(summary['run'])} 月度核心-卫星审计",
                "",
                "| 买入日 | 卖出日 | 格式 | 验证 | 状态 | 核心/卫星/现金 | 实际卫星 | 确定性卫星 | 来源/回退 | 候选日/目标 | 卫星毛贡献 | 卫星净贡献 | 卫星毛/净损益 | 超额胜 | 实际净收益 | 确定性CF净收益 | arithmetic增量收益/损益 | compounded增量损益 | 稀释CF净收益 | arithmetic卫星增量收益/损益 | compounded卫星增量损益 | 纯动量CF净收益 | 双向换手 | 实际/确定性/稀释/纯动量费用 |",
                "|---|---|---|---|---|---|---|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in monthly:
            status = "OK" if row["parse_ok"] else "现金回退"
            validation = (
                "Core-Sat OK"
                if row["core_satellite_compliant"]
                else (
                    "Dilution OK"
                    if row["dynamic_core_cash_dilution_compliant"]
                    else "违规"
                )
            )
            sleeve = (
                f"{_pct(float(row['core_weight']))}/"
                f"{_pct(float(row['satellite_weight']))}/"
                f"{_pct(float(row['cash_weight']))}"
            )
            source = (
                f"{row['selection_source']}/"
                f"{_fallback_cell(row['selection_fallback'])}"
            )
            candidate_audit = (
                f"{row['candidate_signal_date'] or '-'}/"
                f"{row['candidate_target'] or '-'}"
            )
            satellite_pnl = (
                f"{_money(row['satellite_gross_pnl'])}/"
                f"{_money(row['satellite_net_pnl'])}"
            )
            arithmetic = (
                f"{_pct(float(row['arithmetic_agent_incremental_return_vs_deterministic']))}/"
                f"{_money(row['arithmetic_agent_incremental_pnl_vs_deterministic'])}"
            )
            dilution_arithmetic = (
                f"{_pct(float(row['arithmetic_satellite_incremental_return_vs_dilution']))}/"
                f"{_money(row['arithmetic_satellite_incremental_pnl_vs_dilution'])}"
            )
            costs = (
                f"{_money(row['total_cost'])}/"
                f"{_money(row['deterministic_counterfactual_total_cost'])}/"
                f"{_money(row['dynamic_core_cash_dilution_counterfactual_total_cost'])}/"
                f"{_money(row['pure_momentum_counterfactual_total_cost'])}"
            )
            lines.append(
                f"| {row['entry_date']} | {row['exit_date']} | {status} | "
                f"{validation} | {row['market_regime'] or '-'} | {sleeve} | "
                f"{_codes_cell(row['selected_satellites'])} | "
                f"{_codes_cell(row['deterministic_satellites'])} | "
                f"{source} | {_markdown_cell(candidate_audit)} | "
                f"{_pct(float(row['satellite_gross_contribution']))} | "
                f"{_pct(float(row['satellite_net_contribution']))} | "
                f"{satellite_pnl} | "
                f"{'是' if row['satellite_excess_win'] is True else '否' if row['satellite_excess_win'] is False else '-'} | "
                f"{_pct(float(row['net_return']))} | "
                f"{_pct(float(row['deterministic_counterfactual_net_return']))} | "
                f"{arithmetic} | "
                f"{_money(row['compounded_agent_incremental_pnl_vs_deterministic'])} | "
                f"{_pct(float(row['dynamic_core_cash_dilution_counterfactual_net_return']))} | "
                f"{dilution_arithmetic} | "
                f"{_money(row['compounded_satellite_incremental_pnl_vs_dilution'])} | "
                f"{_pct(float(row['pure_momentum_counterfactual_net_return']))} | "
                f"{_pct(float(row['two_way_turnover']))} | {costs} |"
            )
            errors = [
                str(row.get("parse_error") or ""),
                str(row.get("core_satellite_context_error") or ""),
            ]
            if summary.get("run_kind") == "run":
                errors.append(str(row.get("core_satellite_errors") or ""))
            errors = list(dict.fromkeys(error for error in errors if error))
            if errors:
                message = _markdown_cell("; ".join(errors))
                if len(message) > 240:
                    message = message[:237] + "..."
                lines.append("\n审计提示：`" + message + "`")
    lines.append("")
    return "\n".join(lines)


def render_report(
    evaluated: list[tuple[dict[str, Any], list[dict[str, Any]]]],
    *,
    initial_capital: float,
    open_cost: float,
    close_cost: float,
    min_cost: float,
    anchor_floor: float = 0.60,
    policy_mode: str = "legacy",
) -> str:
    if policy_mode == "core_satellite":
        return _render_core_satellite_report(
            evaluated,
            initial_capital=initial_capital,
            open_cost=open_cost,
            close_cost=close_cost,
            min_cost=min_cost,
        )
    if policy_mode == "legacy":
        return _render_legacy_report(
            evaluated,
            initial_capital=initial_capital,
            open_cost=open_cost,
            close_cost=close_cost,
            min_cost=min_cost,
            anchor_floor=anchor_floor,
        )
    raise ValueError("policy_mode must be 'legacy' or 'core_satellite'")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", default=[], help="name=logs/path")
    parser.add_argument("--tasks", default=str(DEFAULT_TASKS))
    parser.add_argument("--out", default="logs/ashare_trader_report.md")
    parser.add_argument("--json-out", default="")
    parser.add_argument("--initial-capital", type=float, default=1_000_000.0)
    parser.add_argument("--open-cost", type=float, default=DEFAULT_OPEN_COST)
    parser.add_argument("--close-cost", type=float, default=DEFAULT_CLOSE_COST)
    parser.add_argument("--min-cost", type=float, default=DEFAULT_MIN_COST)
    parser.add_argument(
        "--policy-mode",
        choices=("legacy", "core_satellite"),
        default="legacy",
        help="Evaluation policy; legacy preserves the existing report.",
    )
    parser.add_argument(
        "--anchor-floor",
        type=float,
        choices=(0.60, 0.75),
        default=0.60,
        help=("Legacy hard-anchor top4 floor; ignored in core_satellite mode."),
    )
    args = parser.parse_args()

    tasks = load_tasks(args.tasks)
    anchor_policy = (
        AnchorPolicy(enabled=True, mode="core_satellite")
        if args.policy_mode == "core_satellite"
        else AnchorPolicy(
            enabled=True,
            min_top4_weight=args.anchor_floor,
            mode="legacy",
        )
    )
    core_contexts = (
        build_core_satellite_contexts(tasks, policy=anchor_policy)
        if args.policy_mode == "core_satellite"
        else None
    )
    evaluated: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for specification in args.run:
        name, separator, run_path = specification.partition("=")
        if not separator or not name or not run_path:
            raise ValueError(f"invalid --run value: {specification!r}")
        answers = load_run(run_path)
        allocations = parse_run_allocations(tasks, answers)
        audits = (
            load_run_audits(run_path) if args.policy_mode == "core_satellite" else None
        )
        evaluated.append(
            evaluate_allocations(
                name,
                tasks,
                allocations,
                initial_capital=args.initial_capital,
                open_cost=args.open_cost,
                close_cost=args.close_cost,
                min_cost=args.min_cost,
                anchor_policy=anchor_policy,
                policy_mode=args.policy_mode,
                core_contexts=core_contexts,
                selection_audits=audits,
                run_kind="run",
            )
        )

    for name, allocations, is_oracle in build_baseline_allocations(
        tasks,
        policy_mode=args.policy_mode,
        core_contexts=core_contexts,
    ):
        evaluated.append(
            evaluate_allocations(
                name,
                tasks,
                allocations,
                initial_capital=args.initial_capital,
                open_cost=args.open_cost,
                close_cost=args.close_cost,
                min_cost=args.min_cost,
                is_oracle=is_oracle,
                anchor_policy=anchor_policy,
                policy_mode=args.policy_mode,
                core_contexts=core_contexts,
                run_kind="baseline",
            )
        )

    report = render_report(
        evaluated,
        initial_capital=args.initial_capital,
        open_cost=args.open_cost,
        close_cost=args.close_cost,
        min_cost=args.min_cost,
        anchor_floor=args.anchor_floor,
        policy_mode=args.policy_mode,
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    json_output = Path(args.json_out) if args.json_out else output.with_suffix(".json")
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(
        json.dumps(
            _json_safe(
                [
                    {"summary": summary, "monthly": monthly}
                    for summary, monthly in evaluated
                ]
            ),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    print(f"wrote trader report -> {output}")
    print(f"wrote trader JSON -> {json_output}")


if __name__ == "__main__":
    main()
