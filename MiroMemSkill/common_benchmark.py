# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import math
import os
import signal
import sqlite3
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict
import random

import hydra
import openai
from omegaconf import DictConfig, OmegaConf

from src.utils.env_loader import load_project_env

from utils.eval_utils import verify_answer_for_datasets
from src.memory.context import get_memory_components
from src.memory.memory import extract_direction, parse_task_month
from src.memory.monthly_reflection import (
    compute_month_feature_rows,
    compute_month_features,
)
from src.utils.ashare_anchor import (
    AnchorPolicy,
    assemble_core_satellite_allocation,
    build_anchor_snapshot,
    render_anchor_policy_prompt,
)
from src.utils.ashare_satellite import load_excess_signal_candidates
from src.utils.ashare_trader import (
    cash_allocation,
    evaluate_anchor_deviation,
    evaluate_portfolio_month,
    parse_portfolio_weights,
    validate_portfolio_answer,
)
from src.utils.ashare_trader_features import compute_trader_feature_rows
from src.logging.logger import (
    bootstrap_logger,
    task_logging_context,
    init_logging_for_benchmark_evaluation,
)
from config import config_name, config_path
from src.core.pipeline import (
    create_pipeline_components,
    execute_task_pipeline,
)

init_logging_for_benchmark_evaluation(print_task_logs=False)

_REPO_ROOT = Path(__file__).resolve().parent


_SATELLITE_DECISION_FACTOR_FIELDS = (
    "rel5",
    "rel20",
    "rel60",
    "vol20_ann",
    "max_dd120",
    "ma20_gap",
    "amount20_vs120",
    "pe_pct250",
    "pb_pct250",
    "financial_ann_date",
    "or_yoy",
    "netprofit_yoy",
    "ml_rank",
)


def _open_market_holding_returns(
    database_path: str | Path,
    ts_codes: Sequence[str],
    entry_date: str,
    exit_date: str,
) -> dict[str, float]:
    """Compound frozen daily returns for selected open-market holdings."""
    codes = list(dict.fromkeys(str(code).strip().upper() for code in ts_codes if code))
    if not codes:
        return {}
    path = Path(database_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"ASHARE_OPEN_DB does not exist: {path}")
    placeholders = ",".join("?" for _ in codes)
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT ts_code,pct_chg FROM market_daily "
            f"WHERE ts_code IN ({placeholders}) "
            "AND trade_date>? AND trade_date<=? "
            "ORDER BY ts_code,trade_date",
            (*codes, str(entry_date), str(exit_date)),
        ).fetchall()
    finally:
        conn.close()
    navs: dict[str, float] = {}
    for code, change in rows:
        navs.setdefault(str(code), 1.0)
        if change is not None:
            navs[str(code)] *= 1.0 + float(change) / 100.0
    missing = [code for code in codes if code not in navs]
    if missing:
        raise ValueError(
            "missing open-market realized returns for selected holdings: "
            + ",".join(missing)
        )
    return {code: navs[code] - 1.0 for code in codes}


def _pit_panel_value(value: Any) -> Any:
    """Normalize missing PIT factor values for metadata and prompt rendering."""
    if value is None or value == "":
        return "NA"
    if isinstance(value, float) and not math.isfinite(value):
        return "NA"
    return value


def _enrich_satellite_candidate_rows(
    rows: list[dict[str, Any]],
    feature_rows: list[dict[str, Any]],
    *,
    decision_as_of: str,
) -> list[dict[str, Any]]:
    """Join the bounded PIT decision panel without changing signal order."""
    factors_by_code = {
        str(row.get("ts_code", "")).strip().upper(): row for row in feature_rows
    }
    enriched_rows: list[dict[str, Any]] = []
    for source_row in rows:
        row = dict(source_row)
        code = str(row.get("ts_code", "")).strip().upper()
        factors = factors_by_code.get(code, {})
        row["decision_as_of"] = decision_as_of
        for field_name in _SATELLITE_DECISION_FACTOR_FIELDS:
            row[field_name] = _pit_panel_value(factors.get(field_name))
        enriched_rows.append(row)
    return enriched_rows


def _render_satellite_candidate_block(
    rows: list[dict[str, Any]],
    *,
    required_count: int,
) -> str:
    """Render the bounded PIT candidate set injected into the Agent prompt."""
    if not rows:
        raise ValueError("core-satellite candidate block cannot be empty")
    if required_count <= 0 or required_count > len(rows):
        raise ValueError("core-satellite required_count must fit candidate rows")
    signal_date = str(rows[0]["signal_date"])
    target = str(rows[0]["target"])
    lines = [
        "## 超额收益卫星候选（系统点时注入）",
        (
            f"- signal_date={signal_date}（不晚于决策日）；target={target}；"
            "已排除 rel20 top4。"
        ),
        (
            f"- 必须只选 {required_count} 只卫星代码（不多不少），且只能从以下候选中选择；"
            "固定权重与无效输出回退由系统处理。"
        ),
        (
            "- 比较超额模型 score/rank 与严格点时的趋势、风险、估值、基本面证据，"
            "并结合提示中已到期的结构化记忆（若有），不得机械照抄排名；"
            "若无可信反向证据，仍可选择模型排名最前者。"
        ),
        (
            "- 数值口径：rel/vol/max_dd/ma/or_yoy/netprofit_yoy 为%；"
            "估值分位为 0-100；amount20_vs120 为成交额均值比；"
            "ml_rank 为全池绝对 Qlib 排名。"
        ),
    ]
    for position, row in enumerate(rows, start=1):
        lines.append(
            f"- {position}. {row['ts_code']} "
            f"(score={float(row['score']):.8g}, signal_rank={int(row['rank'])}, "
            f"signal_date={row['signal_date']}, train_end={row['train_end']}, "
            f"target={row['target']})"
        )
        lines.append(
            "  - PIT "
            f"decision_as_of={_pit_panel_value(row.get('decision_as_of'))}; "
            f"trend: rel5={_pit_panel_value(row.get('rel5'))}, "
            f"rel20={_pit_panel_value(row.get('rel20'))}, "
            f"rel60={_pit_panel_value(row.get('rel60'))}, "
            f"ma20_gap={_pit_panel_value(row.get('ma20_gap'))}; "
            f"risk: vol20_ann={_pit_panel_value(row.get('vol20_ann'))}, "
            f"max_dd120={_pit_panel_value(row.get('max_dd120'))}; "
            "liquidity: "
            f"amount20_vs120={_pit_panel_value(row.get('amount20_vs120'))}; "
            f"valuation: pe_pct250={_pit_panel_value(row.get('pe_pct250'))}, "
            f"pb_pct250={_pit_panel_value(row.get('pb_pct250'))}; "
            "fundamental: "
            "financial_ann_date="
            f"{_pit_panel_value(row.get('financial_ann_date'))}, "
            f"or_yoy={_pit_panel_value(row.get('or_yoy'))}, "
            f"netprofit_yoy={_pit_panel_value(row.get('netprofit_yoy'))}; "
            f"absolute_qlib: ml_rank={_pit_panel_value(row.get('ml_rank'))}"
        )
    return "\n".join(lines)


_SATELLITE_CANDIDATE_KEYS = (
    "ranked_prediction_candidates",
    "prediction_candidates",
    "eligible_prediction_candidates",
    "expected_satellite_candidates",
    "expected_candidates",
    "satellite_candidates",
    "prediction_candidate_details",
)


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _stock_code(value: Any) -> str:
    return str(value or "").strip().upper()


def _code_list(value: Any) -> list[str]:
    if isinstance(value, str):
        raw_codes: Sequence[Any] = [value]
    elif isinstance(value, Mapping):
        raw_codes = list(value)
    elif isinstance(value, Sequence):
        raw_codes = value
    else:
        return []
    return list(
        dict.fromkeys(code for item in raw_codes if (code := _stock_code(item)))
    )


def _candidate_code(row: Mapping[str, Any]) -> str:
    for key in ("ts_code", "code", "symbol"):
        code = _stock_code(row.get(key))
        if code:
            return code
    return ""


def _candidate_rows(value: Any) -> list[dict[str, Any]]:
    """Retain only structured, point-in-time candidate rank/score inputs."""
    if isinstance(value, Mapping):
        if _candidate_code(value):
            raw_items: list[Any] = [value]
        else:
            raw_items = []
            for code, item in value.items():
                if isinstance(item, Mapping):
                    row = dict(item)
                    row.setdefault("ts_code", code)
                else:
                    row = {"ts_code": code, "score": item}
                raw_items.append(row)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        raw_items = list(value)
    elif isinstance(value, str):
        raw_items = [value]
    else:
        return []

    rows: list[dict[str, Any]] = []
    for item in raw_items:
        if isinstance(item, str):
            rows.append({"ts_code": item})
        elif isinstance(item, Mapping):
            rows.append(dict(item))
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            parts = list(item)
            if parts:
                row = {"ts_code": parts[0]}
                if len(parts) > 1:
                    row["score"] = parts[1]
                rows.append(row)
    return rows


def _candidate_detail_lookup(
    snapshot: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Collect factual model rank/score fields without retaining other text."""
    sources: list[Any] = []
    selection = metadata.get("satellite_selection")
    if isinstance(selection, Mapping):
        for key in ("candidates", "candidate_details", "ranked_candidates"):
            if key in selection:
                sources.append(selection.get(key))
    for container in (metadata, snapshot):
        for key in _SATELLITE_CANDIDATE_KEYS:
            if key in container:
                sources.append(container.get(key))

    details: dict[str, dict[str, Any]] = {}
    for source in sources:
        for row in _candidate_rows(source):
            code = _candidate_code(row)
            if not code:
                continue
            detail = details.setdefault(code, {})
            if "model_rank" not in detail:
                for key in (
                    "model_rank",
                    "rank",
                    "candidate_rank",
                    "eligible_rank",
                    "prediction_rank",
                    "ml_rank",
                ):
                    rank = _finite_float(row.get(key))
                    if rank is not None and rank > 0:
                        detail["model_rank"] = (
                            int(rank) if rank.is_integer() else rank
                        )
                        break
            if "model_score" not in detail:
                for key in (
                    "model_score",
                    "score",
                    "prediction_score",
                    "predicted_excess_return",
                    "ml_score",
                ):
                    score = _finite_float(row.get(key))
                    if score is not None:
                        detail["model_score"] = score
                        break
    if isinstance(selection, Mapping):
        for keys, field in (
            (("candidate_ranks", "ranks", "model_ranks"), "model_rank"),
            (("candidate_scores", "scores", "model_scores"), "model_score"),
        ):
            for key in keys:
                values = selection.get(key)
                if not isinstance(values, Mapping):
                    continue
                for raw_code, raw_value in values.items():
                    code = _stock_code(raw_code)
                    number = _finite_float(raw_value)
                    if not code or number is None or (
                        field == "model_rank" and number <= 0
                    ):
                        continue
                    details.setdefault(code, {}).setdefault(
                        field,
                        int(number)
                        if field == "model_rank" and number.is_integer()
                        else number,
                    )
    return details


def _candidate_payload(
    snapshot: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> Any | None:
    """Return the first non-empty PIT candidate payload, snapshot first."""
    for container in (snapshot, metadata):
        for key in _SATELLITE_CANDIDATE_KEYS:
            raw = container.get(key)
            if isinstance(raw, Mapping) and raw:
                return raw
            if (
                isinstance(raw, Sequence)
                and not isinstance(raw, (str, bytes))
                and len(raw) > 0
            ):
                return raw
    selection = metadata.get("satellite_selection")
    if isinstance(selection, Mapping):
        for key in ("candidates", "candidate_details", "ranked_candidates"):
            raw = selection.get(key)
            if isinstance(raw, Mapping) and raw:
                return raw
            if (
                isinstance(raw, Sequence)
                and not isinstance(raw, (str, bytes))
                and len(raw) > 0
            ):
                return raw
    return None


def _normalized_fallback_flag(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        if normalized in {
            "1",
            "true",
            "yes",
            "used",
            "fallback",
            "deterministic",
            "deterministic_fallback",
            "model_top",
        }:
            return True
        if normalized in {
            "0",
            "false",
            "no",
            "unused",
            "none",
            "explicit",
            "model_selected",
        }:
            return False
    if isinstance(value, Mapping):
        for key in (
            "used",
            "fallback_used",
            "selection_fallback",
            "used_fallback",
        ):
            if key in value:
                return _normalized_fallback_flag(value.get(key))
    return None


def _selection_fallback(metadata: Mapping[str, Any]) -> bool | None:
    for key in (
        "selection_fallback",
        "satellite_selection_fallback",
        "satellite_fallback",
    ):
        if key in metadata:
            flag = _normalized_fallback_flag(metadata.get(key))
            if flag is not None:
                return flag
    selection = metadata.get("satellite_selection")
    if isinstance(selection, Mapping):
        return _normalized_fallback_flag(selection)
    return None


def _build_core_satellite_attribution(
    *,
    metadata: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    actual_weights: Mapping[str, float],
    actual_cash: float,
    actual_result: Any,
    stock_returns: Mapping[str, float],
    excess_returns: Mapping[str, float],
    index_return: float,
    starting_capital: float,
    open_cost: float,
    close_cost: float,
    min_cost: float,
    anchor_attribution: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Build a factual, liquidation-gated satellite audit or fail closed."""
    selection = metadata.get("satellite_selection")
    selection_mapping = selection if isinstance(selection, Mapping) else {}
    candidate_payload = _candidate_payload(snapshot, metadata)
    if candidate_payload is None:
        return None

    attribution_snapshot = dict(snapshot)
    for key in _SATELLITE_CANDIDATE_KEYS:
        attribution_snapshot.pop(key, None)
    attribution_snapshot["ranked_prediction_candidates"] = candidate_payload

    try:
        deterministic = assemble_core_satellite_allocation(attribution_snapshot)
    except (KeyError, TypeError, ValueError):
        return None

    top4 = [_stock_code(code) for code in deterministic.get("top4", [])]
    if len(top4) != 4 or len(set(top4)) != 4 or not all(top4):
        return None
    top4_set = set(top4)
    eligible = [
        _stock_code(code)
        for code in deterministic.get("eligible_prediction_candidates", [])
        if _stock_code(code)
    ]
    if not eligible or len(set(eligible)) != len(eligible):
        return None
    eligible_rank = {code: rank for rank, code in enumerate(eligible)}

    actual_satellites = sorted(
        (
            _stock_code(code)
            for code, weight in actual_weights.items()
            if _stock_code(code) not in top4_set and float(weight) > 1e-8
        ),
        key=lambda code: (eligible_rank.get(code, len(eligible_rank)), code),
    )
    declared_actual = _code_list(selection_mapping.get("selected_codes"))
    if declared_actual and set(declared_actual) != set(actual_satellites):
        return None
    try:
        actual_fixed = assemble_core_satellite_allocation(
            attribution_snapshot,
            selected_satellites=actual_satellites,
        )
    except (KeyError, TypeError, ValueError):
        return None

    # Attribution applies only to the canonical system-fixed portfolio.
    expected_actual_weights = actual_fixed.get("weights", {})
    if not isinstance(expected_actual_weights, Mapping):
        return None
    if any(
        abs(float(actual_weights.get(code, 0.0)) - float(weight)) > 1e-6
        for code, weight in expected_actual_weights.items()
    ):
        return None
    if abs(float(actual_cash) - float(actual_fixed.get("cash", -1.0))) > 1e-6:
        return None
    expected_codes = {
        _stock_code(code) for code in expected_actual_weights if _stock_code(code)
    }
    if any(
        _stock_code(code) not in expected_codes and float(weight) > 1e-8
        for code, weight in actual_weights.items()
    ):
        return None

    deterministic_weights = deterministic.get("weights", {})
    if not isinstance(deterministic_weights, Mapping):
        return None
    deterministic_satellites = [
        _stock_code(code) for code in deterministic.get("satellite_codes", [])
    ]
    if not deterministic_satellites or not all(deterministic_satellites):
        return None
    declared_deterministic = _code_list(
        selection_mapping.get("deterministic_codes")
    )
    if declared_deterministic and set(declared_deterministic) != set(
        deterministic_satellites
    ):
        return None
    selection_differs = set(actual_satellites) != set(deterministic_satellites)
    if selection_differs:
        try:
            deterministic_result = evaluate_portfolio_month(
                deterministic_weights,
                float(deterministic["cash"]),
                stock_returns,
                index_return,
                starting_capital=starting_capital,
                excess_returns=excess_returns,
                open_cost=open_cost,
                close_cost=close_cost,
                min_cost=min_cost,
            )
        except (KeyError, TypeError, ValueError):
            return None
    else:
        # The deterministic portfolio is byte-for-byte equivalent in economic
        # terms, so the already evaluated actual result is the exact baseline.
        deterministic_result = actual_result

    detail_lookup = _candidate_detail_lookup(snapshot, metadata)
    candidate_facts: list[dict[str, Any]] = []
    relevant_codes = set(actual_satellites) | set(deterministic_satellites)
    for candidate_order, code in enumerate(eligible, 1):
        if code not in relevant_codes:
            continue
        realized_return = _finite_float(stock_returns.get(code))
        realized_excess = _finite_float(excess_returns.get(code))
        if realized_return is None:
            return None
        if realized_excess is None:
            realized_excess = realized_return - float(index_return)
        actual_weight = float(actual_weights.get(code, 0.0))
        deterministic_weight = float(deterministic_weights.get(code, 0.0))
        actual_net_contribution = _finite_float(
            actual_result.contributions.get(code, 0.0)
        )
        deterministic_net_contribution = _finite_float(
            deterministic_result.contributions.get(code, 0.0)
        )
        if (
            actual_net_contribution is None
            or deterministic_net_contribution is None
        ):
            return None
        row: dict[str, Any] = {
            "code": code,
            "candidate_rank": candidate_order,
            "realized_stock_return": realized_return,
            "realized_excess_vs_csi300": realized_excess,
            "selected": code in actual_satellites,
            "deterministic_selected": code in deterministic_satellites,
            "actual_weight": actual_weight,
            "deterministic_weight": deterministic_weight,
            "weighted_return_contribution": actual_weight * realized_return,
            "weighted_excess_contribution": (
                actual_weight * realized_excess
            ),
            "net_contribution": actual_net_contribution,
            "deterministic_weighted_return_contribution": (
                deterministic_weight * realized_return
            ),
            "deterministic_weighted_excess_contribution": (
                deterministic_weight * realized_excess
            ),
            "deterministic_net_contribution": deterministic_net_contribution,
        }
        row.update(detail_lookup.get(code, {}))
        candidate_facts.append(row)
    if {row["code"] for row in candidate_facts} != relevant_codes:
        return None

    sleeve = deterministic.get("sleeve", {})
    if not isinstance(sleeve, Mapping):
        return None
    regime = str(deterministic.get("regime", "")).strip().lower()
    core_weight = _finite_float(deterministic.get("core_total_weight"))
    satellite_weight = sum(
        float(actual_weights.get(code, 0.0)) for code in actual_satellites
    )
    deterministic_satellite_weight = sum(
        float(deterministic_weights.get(code, 0.0))
        for code in deterministic_satellites
    )
    cash_weight = _finite_float(actual_cash)
    if (
        regime not in {"risk_on", "neutral", "defensive"}
        or core_weight is None
        or cash_weight is None
        or satellite_weight <= 0.0
        or deterministic_satellite_weight <= 0.0
    ):
        return None

    satellite_weighted_return = sum(
        row["weighted_return_contribution"] for row in candidate_facts
    )
    satellite_weighted_excess = sum(
        row["weighted_excess_contribution"] for row in candidate_facts
    )
    satellite_net_contribution = sum(
        row["net_contribution"] for row in candidate_facts
    )
    deterministic_weighted_return = sum(
        row["deterministic_weighted_return_contribution"]
        for row in candidate_facts
    )
    deterministic_weighted_excess = sum(
        row["deterministic_weighted_excess_contribution"]
        for row in candidate_facts
    )
    deterministic_net_contribution = sum(
        row["deterministic_net_contribution"] for row in candidate_facts
    )
    core_weighted_return_contribution = sum(
        float(actual_weights.get(code, 0.0)) * float(stock_returns[code])
        for code in top4
    )
    core_net_contribution = sum(
        float(actual_result.contributions.get(code, 0.0)) for code in top4
    )

    pure_momentum_net_return = _finite_float(
        (anchor_attribution or {}).get("anchor_net_return")
    )
    if pure_momentum_net_return is None:
        raw_anchor_weights = snapshot.get("anchor_weights")
        if isinstance(raw_anchor_weights, Mapping):
            pure_weights = {
                code: float(raw_anchor_weights.get(code, 0.0)) for code in top4
            }
            pure_cash = float(snapshot.get("anchor_cash", 0.0))
        else:
            pure_weights = {code: 1.0 / len(top4) for code in top4}
            pure_cash = 0.0
        try:
            pure_result = evaluate_portfolio_month(
                pure_weights,
                pure_cash,
                stock_returns,
                index_return,
                starting_capital=starting_capital,
                excess_returns=excess_returns,
                open_cost=open_cost,
                close_cost=close_cost,
                min_cost=min_cost,
            )
        except (KeyError, TypeError, ValueError):
            return None
        pure_momentum_net_return = float(pure_result.net_return)
    actual_minus_momentum = (
        float(actual_result.net_return) - pure_momentum_net_return
    )

    selection_source = (
        str(selection_mapping.get("source", "") or "").strip().lower()
    )
    if selection_source not in {"agent", "deterministic_fallback"}:
        selection_source = "unknown"
    selection_fallback = _selection_fallback(metadata)
    if selection_fallback is None and selection_source != "unknown":
        selection_fallback = selection_source == "deterministic_fallback"
    if selection_source == "unknown":
        if selection_fallback is True:
            selection_source = "deterministic_fallback"
        elif selection_fallback is False and selection_mapping:
            selection_source = "agent"

    candidate_signal_date = str(
        selection_mapping.get("candidate_signal_date", "") or ""
    ).strip()[:32]
    raw_target = selection_mapping.get("target")
    target = (
        str(raw_target).strip()[:120]
        if isinstance(raw_target, (str, int, float, bool))
        else ""
    )

    actual_minus_deterministic = (
        float(actual_result.net_return) - deterministic_result.net_return
    )
    numeric_facts = (
        satellite_weighted_return,
        satellite_weighted_excess,
        satellite_net_contribution,
        deterministic_weighted_return,
        deterministic_weighted_excess,
        deterministic_net_contribution,
        core_weighted_return_contribution,
        core_net_contribution,
        actual_minus_momentum,
        actual_minus_deterministic,
    )
    if not all(math.isfinite(value) for value in numeric_facts):
        return None

    attribution: dict[str, Any] = {
        "policy_mode": "core_satellite",
        "regime": regime,
        "top4_codes": top4,
        "selected_codes": actual_satellites,
        "deterministic_codes": deterministic_satellites,
        "selection_differs": selection_differs,
        "selection_fallback": selection_fallback,
        "selection_source": selection_source,
        "candidate_signal_date": candidate_signal_date,
        "target": target,
        "core_weight": core_weight,
        "satellite_weight": satellite_weight,
        "cash_weight": cash_weight,
        "candidate_facts": candidate_facts,
        "satellite_gross_return": satellite_weighted_return / satellite_weight,
        "satellite_excess_return": satellite_weighted_excess / satellite_weight,
        "satellite_net_return": satellite_net_contribution / satellite_weight,
        "satellite_weighted_return_contribution": satellite_weighted_return,
        "satellite_weighted_excess_contribution": satellite_weighted_excess,
        "satellite_net_contribution": satellite_net_contribution,
        "core_weighted_return_contribution": core_weighted_return_contribution,
        "core_net_contribution": core_net_contribution,
        "deterministic_satellite_gross_return": (
            deterministic_weighted_return / deterministic_satellite_weight
        ),
        "deterministic_satellite_excess_return": (
            deterministic_weighted_excess / deterministic_satellite_weight
        ),
        "deterministic_satellite_net_return": (
            deterministic_net_contribution / deterministic_satellite_weight
        ),
        "deterministic_satellite_weighted_return_contribution": (
            deterministic_weighted_return
        ),
        "deterministic_satellite_weighted_excess_contribution": (
            deterministic_weighted_excess
        ),
        "deterministic_satellite_net_contribution": (
            deterministic_net_contribution
        ),
        "actual_net_return": float(actual_result.net_return),
        "deterministic_counterfactual_net_return": float(
            deterministic_result.net_return
        ),
        "deterministic_counterfactual_total_cost": float(
            deterministic_result.total_cost
        ),
        "actual_minus_deterministic_net_return": actual_minus_deterministic,
        "actual_minus_deterministic_pnl": float(
            actual_result.ending_capital - deterministic_result.ending_capital
        ),
        "pure_momentum_net_return": pure_momentum_net_return,
        "actual_minus_pure_momentum_net_return": actual_minus_momentum,
        "actual_minus_pure_momentum_pnl": float(
            actual_minus_momentum * starting_capital
        ),
    }
    return attribution


class TaskStatus(StrEnum):
    PENDING = "pending"
    RUN_FAILED = "run_failed"
    RUN_COMPLETED = "run_completed"
    RESULT_JUDGED = "result_judged"


@dataclass
class BenchmarkTask:
    """Generic benchmark task data structure"""

    task_id: str
    task_question: str
    ground_truth: str
    file_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    model_response: str = ""
    model_boxed_answer: str = ""
    status: TaskStatus = TaskStatus.PENDING
    # status: str = "pending"  # pending, success, failed


class AttemptStats(TypedDict):
    attempt_number: int
    model_response: str
    model_boxed_answer: str
    status: TaskStatus
    log_file_path: Optional[Path]
    judge_result: Optional[str]
    is_correct: bool
    error_message: Optional[str]


@dataclass
class BenchmarkResult:
    """Generic benchmark evaluation result structure"""

    task_id: str
    task_question: str
    ground_truth: str
    file_path: Optional[str]
    model_response: str
    model_boxed_answer: str
    status: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: str = ""
    judge_result: Optional[str] = None
    log_file_path: Optional[Path] = None
    # Pass@K support fields
    attempts: List[AttemptStats] = field(default_factory=list)  # Store all attempts
    pass_at_k_success: bool = False  # Whether task passed using pass@k evaluation
    k_value: int = 1  # The k value used for this evaluation

    def to_dict(self):
        """Convert the object to a serializable dictionary."""
        result = self.__dict__.copy()  # Copy the object's dictionary
        # Convert Path objects to string
        if isinstance(result.get("log_file_path"), Path):
            result["log_file_path"] = str(result["log_file_path"])
        if isinstance(result.get("file_path"), Path):
            result["file_path"] = str(result["file_path"])
        # Convert any Path objects inside the attempts list
        for attempt in result.get("attempts", []):
            if isinstance(attempt.get("log_file_path"), Path):
                attempt["log_file_path"] = str(attempt["log_file_path"])
        return result


class BenchmarkEvaluator(ABC):
    """Abstract base class for benchmark evaluators"""

    def __init__(self, data_dir: str, benchmark_name: str, cfg: DictConfig):
        """
        Initialize benchmark evaluator

        Args:
            data_dir: Path to benchmark data directory
            benchmark_name: Name of the benchmark
            cfg: The Hydra configuration object
        """
        self.data_dir = Path(data_dir)
        self.benchmark_name = benchmark_name
        self.cfg = cfg
        self.pass_at_k = cfg.benchmark.execution.get("pass_at_k", 1)
        self.output_dir = Path(cfg.output_dir).absolute()
        if not self.output_dir.exists():
            os.makedirs(self.output_dir, exist_ok=True)
            print(f"Created output directory: {self.output_dir}")
        # Judge client: defaults to official OpenAI; override via EVAL_LLM_* envs
        # (we point it to DeepSeek since no OpenAI key is available).
        self.evaluation_llm = openai.AsyncOpenAI(
            api_key=os.getenv("EVAL_LLM_API_KEY") or cfg.benchmark.openai_api_key,
            base_url=os.getenv("EVAL_LLM_BASE_URL") or None,
        )
        self.tasks: List[BenchmarkTask] = []
        self.results: List[BenchmarkResult] = []

        # Initialize pipeline components
        logs_dir = self.get_log_dir()
        print("Initializing pipeline components...")
        (
            self.main_agent_tool_manager,
            self.sub_agent_tool_managers,
            self.output_formatter,
        ) = create_pipeline_components(cfg, logs_dir=str(logs_dir))
        print(
            f"Pipeline components initialized successfully! Using pass@{self.pass_at_k}"
        )

    @abstractmethod
    def load_tasks(self) -> List[BenchmarkTask]:
        """Load benchmark tasks from data files"""
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def prepare_task_description(
        self, task: BenchmarkTask
    ) -> Tuple[str, Optional[str]]:
        """Prepare task description and file path for the agent"""
        raise NotImplementedError("Subclasses must implement this method")

    def get_log_dir(self) -> Path:
        """Get the log directory for the current benchmark and model."""
        return Path(self.cfg.output_dir)

    def _attach_trader_anchor_policy(self, task: BenchmarkTask) -> None:
        """Attach one PIT-safe policy snapshot before prompt construction."""
        metadata = task.metadata or {}
        if metadata.get("task_type") != "portfolio_allocation":
            return
        configured = self.cfg.benchmark.get("anchor_policy", {})
        raw_policy = OmegaConf.to_container(configured, resolve=True)
        policy_values = raw_policy if isinstance(raw_policy, dict) else {}
        policy = AnchorPolicy.from_mapping(policy_values)
        policy_metadata = policy.to_dict()
        if policy.mode == "core_satellite":
            raw_candidate_limit = policy_values.get("candidate_limit", 6)
            if isinstance(raw_candidate_limit, bool):
                raise ValueError("core-satellite candidate_limit must be an integer")
            try:
                candidate_limit = int(raw_candidate_limit)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "core-satellite candidate_limit must be an integer"
                ) from exc
            if candidate_limit <= 0:
                raise ValueError(
                    "core-satellite candidate_limit must be greater than zero"
                )
            policy_metadata["candidate_limit"] = candidate_limit
        metadata["anchor_policy"] = policy_metadata
        task.metadata = metadata
        if not policy.enabled:
            metadata.pop("anchor_snapshot", None)
            return

        configured_data_dir = Path(str(self.cfg.get("data_dir", "data")))
        if not configured_data_dir.is_absolute():
            configured_data_dir = _REPO_ROOT / configured_data_dir
        ashare_data_dir = configured_data_dir / "ashare"
        as_of = str(metadata.get("entry_date") or metadata.get("as_of") or "")
        stock_info = metadata.get("stock_info", {})
        snapshot = build_anchor_snapshot(
            as_of,
            stock_info,
            data_dir=ashare_data_dir,
            policy=policy,
        )
        candidate_rows: list[dict[str, Any]] = []
        if policy.mode == "core_satellite":
            pool = {
                str(code).strip().upper()
                for code in metadata.get("stock_pool", [])
            }
            loaded_rows = load_excess_signal_candidates(
                as_of,
                snapshot.get("top4", []),
                data_dir=ashare_data_dir,
            )
            candidate_rows = [
                dict(row)
                for row in loaded_rows
                if str(row.get("ts_code", "")).strip().upper() in pool
            ][:candidate_limit]
            feature_rows = compute_trader_feature_rows(
                as_of,
                stock_info,
                data_dir=ashare_data_dir,
                lookback_days=250,
            )
            candidate_rows = _enrich_satellite_candidate_rows(
                candidate_rows,
                feature_rows,
                decision_as_of=str(snapshot.get("as_of", as_of)),
            )
            snapshot = build_anchor_snapshot(
                as_of,
                stock_info,
                data_dir=ashare_data_dir,
                policy=policy,
                prediction_candidates=candidate_rows,
            )
            # This validates the regime-specific candidate count and fails
            # closed before any Agent call when the sleeve cannot be formed.
            allocation = assemble_core_satellite_allocation(snapshot)
            required_count = int(allocation["satellite_count"])
            signal_dates = {
                str(row.get("signal_date", "")).strip() for row in candidate_rows
            }
            targets = {str(row.get("target", "")).strip() for row in candidate_rows}
            if len(signal_dates) != 1 or "" in signal_dates:
                raise ValueError(
                    "core-satellite candidates require one non-empty signal_date"
                )
            if len(targets) != 1 or "" in targets:
                raise ValueError(
                    "core-satellite candidates require one non-empty target"
                )
            metadata["satellite_candidates"] = candidate_rows
            metadata["satellite_signal_date"] = next(iter(signal_dates))
            metadata["satellite_target"] = next(iter(targets))
            metadata["satellite_candidate_limit"] = candidate_limit
            metadata["satellite_required_count"] = required_count
        metadata["anchor_snapshot"] = snapshot
        marker = "## 动量硬锚（系统将确定性复核）"
        if marker not in task.task_question:
            task.task_question = (
                task.task_question.rstrip()
                + "\n"
                + render_anchor_policy_prompt(policy)
                + "\n"
            )
        candidate_marker = "## 超额收益卫星候选（系统点时注入）"
        if (
            policy.mode == "core_satellite"
            and candidate_marker not in task.task_question
        ):
            task.task_question = (
                task.task_question.rstrip()
                + "\n\n"
                + _render_satellite_candidate_block(
                    candidate_rows,
                    required_count=int(metadata["satellite_required_count"]),
                )
                + "\n"
            )

    async def run_single_task(self, task: BenchmarkTask) -> BenchmarkResult:
        """
        Run inference for a single benchmark task with pass@k support

        Args:
            task: BenchmarkTask object

        Returns:
            BenchmarkResult object
        """
        print(f"Processing task {task.task_id} with pass@{self.pass_at_k}")

        result = BenchmarkResult(
            task_id=task.task_id,
            task_question=task.task_question,
            ground_truth=task.ground_truth,
            file_path=task.file_path,
            model_response="",
            model_boxed_answer="",
            status="pending",
            metadata=task.metadata.copy(),
            error_message="",
            judge_result=None,
            log_file_path=None,
            attempts=[],
            pass_at_k_success=False,
            k_value=self.pass_at_k,
        )

        found_correct_answer = False

        # Print debug info about log directory
        print(f"  Current result directory: {self.output_dir}")
        print(f"  Current task log directory: {self.output_dir}/task_logs")

        try:
            # Prepare task
            task_description, task_file_path = self.prepare_task_description(task)

            # Run up to k attempts (with early stopping when correct answer found)
            for attempt in range(1, self.pass_at_k + 1):
                print(f"  Attempt {attempt}/{self.pass_at_k} for task {task.task_id}")

                attempt_result = self.scan_latest_attempt(task, attempt)
                # Run inference if no existing result
                if attempt_result["status"] in (
                    TaskStatus.PENDING,
                    TaskStatus.RUN_FAILED,
                ):
                    try:
                        (
                            response,
                            final_boxed_answer,
                            log_file_path,
                        ) = await execute_task_pipeline(
                            cfg=self.cfg,
                            task_id=f"{task.task_id}",
                            task_name=f"{task.task_id}",
                            task_file_name=task_file_path,
                            task_description=task_description,
                            main_agent_tool_manager=self.main_agent_tool_manager,
                            sub_agent_tool_managers=self.sub_agent_tool_managers,
                            output_formatter=self.output_formatter,
                            ground_truth=task.ground_truth,
                            metadata=task.metadata,
                            log_path=self.output_dir
                            / f"task_{task.task_id}_attempt_{attempt}.json",
                        )

                        attempt_result["model_response"] = response if response else ""
                        attempt_result["log_file_path"] = log_file_path
                        if final_boxed_answer:
                            attempt_result["model_boxed_answer"] = final_boxed_answer
                            attempt_result["status"] = TaskStatus.RUN_COMPLETED
                        else:
                            attempt_result["model_boxed_answer"] = final_boxed_answer
                            attempt_result["status"] = TaskStatus.RUN_FAILED

                    except Exception as e:
                        attempt_result["status"] = TaskStatus.RUN_FAILED
                        attempt_result["error_message"] = str(e)
                        print(f"    Error in attempt {attempt}: {e}")

                # Perform LLM verification if we have an answer and haven't verified yet
                if (
                    attempt_result["status"] == TaskStatus.RUN_COMPLETED
                    or attempt_result["judge_result"] == "NOT_ATTEMPTED"
                ):
                    # if attempt_result["status"] == TaskStatus.RUN_COMPLETED:
                    print(f"    Verifying answer for attempt {attempt}...")
                    try:
                        evaluation_result = await verify_answer_for_datasets(
                            openai_client=self.evaluation_llm,
                            benchmark_name=self.benchmark_name,
                            question=task.task_question,
                            target=task.ground_truth,
                            predicted_answer=attempt_result["model_boxed_answer"],
                            metadata=task.metadata,
                        )
                        attempt_result["judge_result"] = evaluation_result
                        attempt_result["is_correct"] = evaluation_result == "CORRECT"

                        # Update the log file with verification result
                        if "log_file_path" in attempt_result and isinstance(
                            attempt_result["log_file_path"], Path
                        ):
                            await self._update_log_file_with_evaluation(
                                attempt_result["log_file_path"], evaluation_result
                            )

                        if attempt_result["is_correct"]:
                            print(f"    ✅ Attempt {attempt}: CORRECT!")
                            found_correct_answer = True
                        else:
                            print(
                                f"    ❌ Attempt {attempt}: INCORRECT ({evaluation_result})"
                            )

                        await self._run_post_judge_hooks(
                            task=task,
                            attempt_result=attempt_result,
                            evaluation_result=evaluation_result,
                        )

                    except Exception as e:
                        print(f"    Error verifying attempt {attempt}: {e}")
                        attempt_result["judge_result"] = "ERROR"
                        attempt_result["is_correct"] = False

                if attempt_result["is_correct"]:
                    print(f"    ✅ Attempt {attempt}: CORRECT (cached)")
                    found_correct_answer = True
                elif attempt_result["judge_result"]:
                    print(
                        f"    ❌ Attempt {attempt}: INCORRECT (cached: {attempt_result['judge_result']})"
                    )
                else:
                    print(f"    ⚠️  Attempt {attempt}: No valid answer to verify")

                result.attempts.append(attempt_result)

                # Update main result with the first successful attempt or best attempt so far
                if attempt == 1 or (
                    attempt_result["status"] == TaskStatus.RUN_COMPLETED
                    and not result.model_boxed_answer
                ):
                    result.model_response = attempt_result["model_response"]
                    result.model_boxed_answer = attempt_result["model_boxed_answer"]
                    result.log_file_path = attempt_result["log_file_path"]
                    result.status = attempt_result["status"]
                    if attempt_result["error_message"] is not None:
                        result.error_message = attempt_result["error_message"]

                # Early stopping: if we found a correct answer, we can stop
                if found_correct_answer:
                    print(
                        f"    🎯 Found correct answer! Stopping early after {attempt} attempts."
                    )
                    break

        except Exception as e:
            result.error_message = str(e)
            result.status = "failed"
            print(f"Error processing task {task.task_id}: {e}")

        finally:
            result.pass_at_k_success = found_correct_answer

            # Set main result LLM judge result based on pass@k outcome
            if found_correct_answer:
                result.judge_result = "PASS_AT_K_SUCCESS"
            else:
                result.judge_result = "PASS_AT_K_FAILED"

            print(f"Task {task.task_id} completed with {len(result.attempts)} attempts")
            print(
                f"    Pass@{self.pass_at_k} result: {'✅ SUCCESS' if found_correct_answer else '❌ FAILED'}"
            )

        return result

    def scan_latest_attempt(self, task: BenchmarkTask, attempt: int) -> AttemptStats:
        """check filesystem for latest attempt"""
        attempt_result: AttemptStats = {
            "attempt_number": attempt,
            "model_response": "",
            "model_boxed_answer": "",
            "status": TaskStatus.PENDING,
            "log_file_path": None,
            "judge_result": None,
            "is_correct": False,
            "error_message": None,
        }
        trace_filename_pattern = f"task_{task.task_id}_attempt_{attempt}.json"
        matched_logs = self.output_dir.glob(trace_filename_pattern)
        sorted_logs = sorted(matched_logs, reverse=True)
        if len(sorted_logs) == 0:
            return attempt_result
        latest_log = sorted_logs[-1]
        attempt_result["status"] = TaskStatus.RUN_FAILED
        attempt_result["log_file_path"] = latest_log
        print(f"    Found existing log for attempt {attempt}: {latest_log.name}")

        with open(latest_log) as f:
            log_data = json.loads(f.read())
            if log_data.get("final_boxed_answer"):
                if self.benchmark_name == "ashare-trader":
                    validation = validate_portfolio_answer(
                        log_data["final_boxed_answer"],
                        task.metadata,
                    )
                    if not validation.ok:
                        attempt_result["error_message"] = (
                            "cached trader allocation rejected: "
                            f"{validation.error}"
                        )
                        print(
                            "    Rejecting invalid cached trader result: "
                            f"{validation.error}"
                        )
                        return attempt_result
                attempt_result["status"] = TaskStatus.RUN_COMPLETED
                attempt_result["model_boxed_answer"] = log_data["final_boxed_answer"]
                attempt_result["model_response"] = log_data.get("output", "")
                # Check if we already have LLM judge result in log
                if log_data.get("judge_result"):
                    attempt_result["status"] = TaskStatus.RESULT_JUDGED
                    attempt_result["judge_result"] = log_data["judge_result"]
                    attempt_result["is_correct"] = log_data["judge_result"] == "CORRECT"
                print(
                    f"    Loaded existing result: {attempt_result['model_boxed_answer']}"
                )
        return attempt_result

    async def run_parallel_inference(
        self, tasks: List[BenchmarkTask], max_concurrent: int = 3
    ) -> List[BenchmarkResult]:
        """Run inference on multiple tasks in parallel"""
        print(
            f"Running inference on {len(tasks)} tasks with max_concurrent={max_concurrent}"
        )

        semaphore = asyncio.Semaphore(max_concurrent)

        async def run_with_semaphore(task):
            async with semaphore:
                with task_logging_context(task.task_id, self.get_log_dir()):
                    result = await self.run_single_task(task)
                return result

        # Task ordering:
        #   "shuffle" (default) avoids order bias;
        #   "sorted"  runs chronologically without barriers;
        #   "monthly" runs chronologically WITH a barrier between entry months.
        #     The legacy monthly reflector learns after each month; rolling v3
        #     refreshes before each month using only labels whose exit dates
        #     have actually matured (exact walk-forward semantics).
        task_order = str(
            OmegaConf.select(self.cfg, "benchmark.execution.task_order") or "shuffle"
        )
        if task_order == "monthly":
            groups: Dict[str, List[BenchmarkTask]] = {}
            for t in tasks:
                groups.setdefault(parse_task_month(t.task_id) or "unknown", []).append(t)
            shuffled_tasks = []
            results = []
            for month in sorted(groups):
                month_tasks = sorted(groups[month], key=lambda t: t.task_id)
                print(f"\n=== month {month}: {len(month_tasks)} tasks ===")
                await self._refresh_rolling_reflection(month, month_tasks)
                shuffled_tasks.extend(month_tasks)
                month_results = await asyncio.gather(
                    *[run_with_semaphore(task) for task in month_tasks],
                    return_exceptions=True,
                )
                results.extend(month_results)
                await self._run_monthly_reflection(month, month_tasks, month_results)
                await self._log_rolling_samples(month, month_tasks, month_results)
        else:
            shuffled_tasks = tasks.copy()
            if task_order == "sorted":
                shuffled_tasks.sort(key=lambda t: (t.task_id.rsplit("_", 1)[-1], t.task_id))
            else:
                random.shuffle(shuffled_tasks)
            results = await asyncio.gather(
                *[run_with_semaphore(task) for task in shuffled_tasks],
                return_exceptions=True,
            )

        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Exception in task {shuffled_tasks[i].task_id}: {result}")
                error_result = BenchmarkResult(
                    task_id=shuffled_tasks[i].task_id,
                    task_question=shuffled_tasks[i].task_question,
                    ground_truth=shuffled_tasks[i].ground_truth,
                    file_path=shuffled_tasks[i].file_path,
                    model_response="",
                    model_boxed_answer="",
                    status="failed",
                    metadata=shuffled_tasks[i].metadata.copy(),
                    error_message=str(result),
                    judge_result=None,
                    log_file_path=None,
                    attempts=[],
                    pass_at_k_success=False,
                    k_value=self.pass_at_k,
                )
                processed_results.append(error_result)
            else:
                processed_results.append(result)

        self.results = processed_results
        return processed_results

    def save_results(self, output_path: Path) -> Path:
        """Save evaluation results to JSONL file"""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for result in self.results:
                f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")

        print(f"Results saved to {output_path}")
        return output_path

    async def evaluate_accuracy(self) -> float:
        """Evaluate pass@k accuracy (verification already done in run_single_task)"""
        if not self.results:
            print("No results to evaluate")
            return 0.0

        print(
            f"Calculating pass@{self.pass_at_k} accuracy for {len(self.results)} results..."
        )

        correct_count = 0
        total_count = 0

        for result in self.results:
            total_count += 1

            # Display task results
            print(f"\nTask {result.task_id}:")
            print(f"  Attempts: {len(result.attempts)}")
            print(
                f"  Pass@{self.pass_at_k}: {'✅ SUCCESS' if result.pass_at_k_success else '❌ FAILED'}"
            )

            # Show details of each attempt
            for attempt in result.attempts:
                attempt_num = attempt.get("attempt_number", "?")
                judge_result = attempt.get("judge_result", "NOT_VERIFIED")
                is_correct = attempt.get("is_correct", False)
                status_icon = (
                    "✅"
                    if is_correct
                    else "❌"
                    if judge_result != "NOT_VERIFIED"
                    else "⚠️"
                )
                print(f"    Attempt {attempt_num}: {status_icon} {judge_result}")
                if attempt.get("model_boxed_answer"):
                    print(f"      Answer: {attempt['model_boxed_answer']}")

            print("  " + "=" * 50)
            print(f"  Reference: {result.ground_truth}")
            print("  " + "=" * 50)

            if result.pass_at_k_success:
                correct_count += 1

        pass_at_k_accuracy = correct_count / total_count if total_count > 0 else 0.0

        print(f"\nPass@{self.pass_at_k} Final Results:")
        print(f"Tasks passed: {correct_count}/{total_count}")
        print(f"Pass@{self.pass_at_k} Accuracy: {pass_at_k_accuracy:.2%}")

        return pass_at_k_accuracy

    async def _run_post_judge_hooks(
        self,
        task: "BenchmarkTask",
        attempt_result: dict,
        evaluation_result: str,
    ) -> None:
        """After judging: append the calibration ledger, then route reflection.

        reflection_mode "per_task" keeps the Mem0 per-task extraction (kept as
        an ablation arm); "monthly" defers learning to the month barrier in
        run_parallel_inference (nothing per-task beyond the ledger).
        """
        if not self.cfg.get("memory") or not self.cfg.memory.get("enabled", False):
            return
        if evaluation_result not in ("CORRECT", "INCORRECT"):
            return

        try:
            memory, _ = get_memory_components(self.cfg)
        except Exception as e:
            print(f"    Memory unavailable after judge: {e}")
            return
        if not memory:
            return

        answer = attempt_result.get("model_boxed_answer", "")
        if self.cfg.memory.get("outcome_logging_enabled", True):
            try:
                memory.log_outcome(
                    task_id=task.task_id,
                    month=parse_task_month(task.task_id),
                    predicted=extract_direction(answer),
                    judge_result=evaluation_result,
                    available_after=(task.metadata or {}).get("exit_date", ""),
                )
            except Exception as e:
                print(f"    Outcome ledger write failed: {e}")

        if not self.cfg.memory.get("reflection_enabled", False):
            return
        if str(self.cfg.memory.get("reflection_mode", "per_task")) != "per_task":
            return

        log_path = attempt_result.get("log_file_path")
        log_data: dict = {}
        if log_path and Path(log_path).exists():
            try:
                with open(log_path, "r", encoding="utf-8") as f:
                    log_data = json.load(f)
            except Exception:
                pass

        try:
            ops = memory.add(
                question=task.task_question,
                answer=answer,
                judge_result=evaluation_result,
                log_data=log_data,
                task_id=task.task_id,
                ts_code=(task.metadata or {}).get("ts_code", ""),
                available_after=(task.metadata or {}).get("exit_date", ""),
            )
            for op in ops:
                print(f"    📝 memory {op}")
        except Exception as e:
            print(f"    Reflection failed: {e}")

    async def _refresh_rolling_reflection(
        self,
        month: str,
        month_tasks: List["BenchmarkTask"],
    ) -> None:
        """Before a month starts, validate rules using labels matured by its as-of date."""
        if not self.cfg.get("memory") or not self.cfg.memory.get("enabled", False):
            return
        if not self.cfg.memory.get("reflection_enabled", False):
            return
        reflection_mode = str(self.cfg.memory.get("reflection_mode", "per_task"))
        if reflection_mode != "rolling":
            return
        if self._month_is_fully_judged_in_cache(month_tasks):
            print(f"    rolling[{month}] cached month; existing rule state retained")
            return

        entry_dates = {
            str((task.metadata or {}).get("entry_date", "")) for task in month_tasks
        }
        entry_dates.discard("")
        if len(entry_dates) != 1:
            print(
                f"    [rolling {month}] expected one decision date, got {sorted(entry_dates)}"
            )
            return
        as_of_date = next(iter(entry_dates))

        try:
            memory, _ = get_memory_components(self.cfg)
        except Exception as e:
            print(f"    Memory unavailable for rolling reflection: {e}")
            return
        if not memory:
            return

        try:
            config = OmegaConf.to_container(self.cfg.memory, resolve=True)
            ops = memory.refresh_rolling(as_of_date, config=config)
            if ops:
                for op in ops:
                    print(f"    rolling[{month}] {op}")
            else:
                print(f"    rolling[{month}] no statistically validated rule")
        except Exception as e:
            print(f"    [rolling {month}] refresh failed: {e}")

    def _month_is_fully_judged_in_cache(
        self, month_tasks: List["BenchmarkTask"]
    ) -> bool:
        """Avoid rewinding/replaying rule mutations for fully cached months."""
        for task in month_tasks:
            judged = False
            for attempt in range(1, self.pass_at_k + 1):
                path = self.output_dir / f"task_{task.task_id}_attempt_{attempt}.json"
                if not path.exists():
                    continue
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        result = json.load(f)
                    if result.get("judge_result") in ("CORRECT", "INCORRECT"):
                        judged = True
                        break
                except (OSError, json.JSONDecodeError):
                    continue
            if not judged:
                return False
        return bool(month_tasks)

    async def _log_rolling_samples(
        self,
        month: str,
        month_tasks: List["BenchmarkTask"],
        month_results: list,
    ) -> None:
        """After judging, upsert this cross-section into the rolling sample ledger."""
        if not self.cfg.get("memory") or not self.cfg.memory.get("enabled", False):
            return
        if not self.cfg.memory.get("reflection_enabled", False):
            return
        reflection_mode = str(self.cfg.memory.get("reflection_mode", "per_task"))
        if reflection_mode == "trader":
            await self._log_trader_month(month, month_tasks, month_results)
            return
        if reflection_mode == "rank_factor":
            await self._log_rank_factor_samples(month, month_tasks, month_results)
            return
        if reflection_mode != "rolling":
            return

        try:
            memory, _ = get_memory_components(self.cfg)
        except Exception as e:
            print(f"    Memory unavailable for rolling sample logging: {e}")
            return
        if not memory:
            return

        tasks_by_id = {task.task_id: task for task in month_tasks}
        stocks: list[dict[str, Any]] = []
        entry_date = ""
        for result in month_results:
            if isinstance(result, Exception) or not getattr(result, "attempts", None):
                continue
            task = tasks_by_id.get(result.task_id)
            if task is None:
                continue
            judged = next(
                (
                    attempt
                    for attempt in result.attempts
                    if attempt.get("judge_result") in ("CORRECT", "INCORRECT")
                ),
                None,
            )
            if judged is None:
                continue
            meta = task.metadata or {}
            entry_date = str(meta.get("entry_date", entry_date))
            predicted = extract_direction(judged.get("model_boxed_answer", ""))
            stocks.append(
                {
                    "task_id": result.task_id,
                    "ts_code": meta.get("ts_code", ""),
                    "stock_name": meta.get("stock_name", ""),
                    "entry_date": meta.get("entry_date", ""),
                    "exit_date": meta.get("exit_date", ""),
                    "label": task.ground_truth,
                    "predicted": predicted,
                    "judge_result": judged["judge_result"],
                }
            )
            # Cached attempts skip the live post-judge hook.
            try:
                memory.log_outcome(
                    task_id=result.task_id,
                    month=month,
                    predicted=predicted,
                    judge_result=judged["judge_result"],
                    available_after=meta.get("exit_date", ""),
                )
            except Exception:
                pass

        if not stocks or not entry_date:
            print(f"    [rolling {month}] no judged rows to log")
            return

        try:
            feature_rows = compute_month_feature_rows(entry_date, stocks)
            features_by_code = {row["ts_code"]: row for row in feature_rows}
            samples: list[dict[str, Any]] = []
            for stock in stocks:
                feature_row = features_by_code.get(stock["ts_code"])
                if not feature_row or not stock.get("exit_date"):
                    continue
                samples.append(
                    {
                        "task_id": stock["task_id"],
                        "entry_month": month,
                        "entry_date": stock["entry_date"],
                        "exit_date": stock["exit_date"],
                        **feature_row,
                    }
                )
            added, updated = memory.log_samples(samples)
            print(
                f"    rolling[{month}] samples upserted: "
                f"added={added}, updated={updated}, month_rows={len(samples)}"
            )
        except Exception as e:
            print(f"    [rolling {month}] sample logging failed: {e}")

    async def _log_rank_factor_samples(
        self,
        month: str,
        month_tasks: List["BenchmarkTask"],
        month_results: list,
    ) -> None:
        """Log one matured cross-section for rank-factor reliability memory."""
        successful_ids = {
            result.task_id
            for result in month_results
            if not isinstance(result, Exception)
            and getattr(result, "attempts", None)
            and any(
                attempt.get("judge_result") == "CORRECT"
                for attempt in result.attempts
            )
        }
        rank_tasks = [
            task
            for task in month_tasks
            if task.task_id in successful_ids
            and (task.metadata or {}).get("task_type") == "cross_section_rank"
        ]
        if len(rank_tasks) != 1:
            print(
                f"    [rank-factor {month}] expected one valid ranking task, "
                f"got {len(rank_tasks)}"
            )
            return
        task = rank_tasks[0]
        metadata = task.metadata or {}
        entry_date = str(metadata.get("entry_date", ""))
        exit_date = str(metadata.get("exit_date", ""))
        pool = list(metadata.get("stock_pool", []))
        excess_returns = metadata.get("excess_returns", {})
        if not entry_date or not exit_date or not pool or not excess_returns:
            print(f"    [rank-factor {month}] incomplete task metadata")
            return

        try:
            memory, _ = get_memory_components(self.cfg)
        except Exception as e:
            print(f"    Memory unavailable for rank-factor logging: {e}")
            return
        if not memory:
            return

        try:
            feature_rows = compute_month_feature_rows(
                entry_date,
                [
                    {
                        "ts_code": code,
                        "stock_name": "",
                        "label": "",
                        "predicted": "",
                        "judge_result": "",
                    }
                    for code in pool
                ],
            )
            samples = []
            for row in feature_rows:
                code = row["ts_code"]
                if code not in excess_returns:
                    continue
                samples.append(
                    {
                        "task_id": f"{task.task_id}_{code}",
                        "entry_month": month,
                        "entry_date": entry_date,
                        "exit_date": exit_date,
                        "excess_return": excess_returns[code],
                        **row,
                    }
                )
            added, updated = memory.log_samples(samples)
            print(
                f"    rank-factor[{month}] samples upserted: "
                f"added={added}, updated={updated}, month_rows={len(samples)}"
            )
        except Exception as e:
            print(f"    [rank-factor {month}] sample logging failed: {e}")

    async def _log_trader_month(
        self,
        month: str,
        month_tasks: List["BenchmarkTask"],
        month_results: list,
    ) -> None:
        """Persist one matured-gated portfolio episode and its factor panel."""
        trader_tasks = [
            task
            for task in month_tasks
            if (task.metadata or {}).get("task_type") == "portfolio_allocation"
        ]
        if len(trader_tasks) != 1:
            print(
                f"    [trader {month}] expected one allocation task, "
                f"got {len(trader_tasks)}"
            )
            return
        task = trader_tasks[0]
        result = next(
            (
                item
                for item in month_results
                if not isinstance(item, Exception)
                and getattr(item, "task_id", "") == task.task_id
            ),
            None,
        )
        if result is None or not getattr(result, "attempts", None):
            print(f"    [trader {month}] no completed result")
            return
        judged = next(
            (
                attempt
                for attempt in result.attempts
                if attempt.get("judge_result") in ("CORRECT", "INCORRECT")
            ),
            None,
        )
        if judged is None:
            print(f"    [trader {month}] no judged attempt")
            return

        metadata = task.metadata or {}
        pool = list(metadata.get("stock_pool", []))
        entry_date = str(metadata.get("entry_date", ""))
        exit_date = str(metadata.get("exit_date", ""))
        if not pool or not entry_date or not exit_date:
            print(f"    [trader {month}] incomplete task metadata")
            return

        answer = str(judged.get("model_boxed_answer", "") or "")
        parsed = parse_portfolio_weights(
            answer,
            pool,
            max_stock_weight=float(metadata.get("max_stock_weight", 0.25)),
        )
        allocation = parsed if parsed.ok else cash_allocation(pool)
        stock_returns = metadata.get("stock_returns", {})
        excess_returns = metadata.get("excess_returns", {})
        stock_info = metadata.get("stock_info", {})
        has_embedded_outcomes = bool(
            stock_returns and excess_returns and stock_info
        )
        open_market_episodes_only = (
            not has_embedded_outcomes
            and str(metadata.get("universe", "")) == "all_ashare_point_in_time"
        )
        if not has_embedded_outcomes and not open_market_episodes_only:
            print(f"    [trader {month}] incomplete task metadata")
            return
        if open_market_episodes_only:
            active_codes = [
                code
                for code, weight in allocation.weights.items()
                if float(weight) > 0.0
            ]
            database_path = os.getenv("ASHARE_OPEN_DB", "")
            if not database_path:
                print(
                    f"    [trader {month}] ASHARE_OPEN_DB is required "
                    "for open-market episode logging"
                )
                return
            try:
                stock_returns = _open_market_holding_returns(
                    database_path,
                    active_codes,
                    entry_date,
                    exit_date,
                )
            except Exception as exc:
                print(
                    f"    [trader {month}] open-market return loading failed: {exc}"
                )
                return
            index_return = float(metadata.get("index_return", 0.0))
            excess_returns = {
                code: value - index_return
                for code, value in stock_returns.items()
            }
            print(
                f"    [trader {month}] open-market episodes-only: "
                f"loaded_returns={len(stock_returns)}"
            )
        episode_notional = float(
            self.cfg.memory.get("trader_episode_notional", 1_000_000.0)
        )
        index_return = float(metadata.get("index_return", 0.0))
        open_cost = float(metadata.get("open_cost", 0.0005))
        close_cost = float(metadata.get("close_cost", 0.0015))
        min_cost = float(metadata.get("min_cost", 5.0))
        portfolio_result = evaluate_portfolio_month(
            allocation.weights,
            allocation.cash,
            stock_returns,
            index_return,
            starting_capital=episode_notional,
            excess_returns=excess_returns,
            open_cost=open_cost,
            close_cost=close_cost,
            min_cost=min_cost,
        )

        configured_data_dir = Path(str(self.cfg.get("data_dir", "data")))
        if not configured_data_dir.is_absolute():
            configured_data_dir = _REPO_ROOT / configured_data_dir
        ashare_data_dir = configured_data_dir / "ashare"
        policy: AnchorPolicy | None = None
        snapshot: Mapping[str, Any] | None = None
        anchor_attribution: dict[str, Any] | None = None
        if not open_market_episodes_only:
            try:
                policy = AnchorPolicy.from_mapping(metadata.get("anchor_policy"))
                raw_snapshot = metadata.get("anchor_snapshot")
                if raw_snapshot:
                    if not isinstance(raw_snapshot, Mapping):
                        raise TypeError("anchor_snapshot must be a mapping")
                    snapshot = raw_snapshot
                else:
                    snapshot = build_anchor_snapshot(
                        entry_date,
                        stock_info,
                        data_dir=ashare_data_dir,
                        policy=policy,
                    )
                anchor_attribution = evaluate_anchor_deviation(
                    allocation.weights,
                    allocation.cash,
                    snapshot["anchor_weights"],
                    float(snapshot.get("anchor_cash", 0.0)),
                    stock_returns,
                    index_return,
                    starting_capital=episode_notional,
                    excess_returns=excess_returns,
                    open_cost=open_cost,
                    close_cost=close_cost,
                    min_cost=min_cost,
                    actual_result=portfolio_result,
                )
                anchor_attribution["market_regime"] = str(
                    snapshot.get("market_breadth", {}).get("regime", "")
                )
            except Exception as exc:
                print(f"    [trader {month}] anchor attribution failed: {exc}")

        satellite_attribution: dict[str, Any] | None = None
        if (
            policy is not None
            and policy.mode == "core_satellite"
            and snapshot is not None
        ):
            satellite_attribution = _build_core_satellite_attribution(
                metadata=metadata,
                snapshot=snapshot,
                actual_weights=allocation.weights,
                actual_cash=allocation.cash,
                actual_result=portfolio_result,
                stock_returns=stock_returns,
                excess_returns=excess_returns,
                index_return=index_return,
                starting_capital=episode_notional,
                open_cost=open_cost,
                close_cost=close_cost,
                min_cost=min_cost,
                anchor_attribution=anchor_attribution,
            )
            if satellite_attribution is None:
                print(
                    f"    [trader {month}] satellite attribution omitted: "
                    "incomplete or inconsistent point-in-time metadata"
                )

        try:
            memory, _ = get_memory_components(self.cfg)
        except Exception as exc:
            print(f"    Memory unavailable for trader logging: {exc}")
            return
        if not memory:
            return

        try:
            operation = memory.add_trader_episode(
                task_id=task.task_id,
                month=month,
                available_after=exit_date,
                weights=allocation.weights,
                cash=allocation.cash,
                gross_return=portfolio_result.gross_return,
                net_return=portfolio_result.net_return,
                index_return=portfolio_result.index_return,
                active_return=portfolio_result.active_return,
                total_cost=portfolio_result.total_cost,
                contributions=portfolio_result.contributions,
                episode_kind=(
                    "open_market" if open_market_episodes_only else ""
                ),
                anchor_attribution=anchor_attribution,
                satellite_attribution=satellite_attribution,
                parse_ok=parsed.ok,
            )
            print(f"    trader[{month}] {operation}")
        except Exception as exc:
            print(f"    [trader {month}] episode logging failed: {exc}")

        if open_market_episodes_only:
            return

        try:
            feature_rows = compute_trader_feature_rows(
                entry_date,
                stock_info,
                data_dir=ashare_data_dir,
                lookback_days=250,
            )
            samples = [
                {
                    "task_id": f"{task.task_id}_{row['ts_code']}",
                    "record_type": "trader_factor_sample",
                    "entry_month": month,
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "excess_return": excess_returns[row["ts_code"]],
                    **row,
                }
                for row in feature_rows
                if row["ts_code"] in excess_returns
            ]
            added, updated = memory.log_samples(samples)
            print(
                f"    trader-factor[{month}] samples upserted: "
                f"added={added}, updated={updated}, month_rows={len(samples)}"
            )
        except Exception as exc:
            print(f"    [trader {month}] factor sample logging failed: {exc}")

    async def _run_monthly_reflection(
        self,
        month: str,
        month_tasks: List["BenchmarkTask"],
        month_results: list,
    ) -> None:
        """Month-barrier hook: cross-sectional reflection over the settled month."""
        if not self.cfg.get("memory") or not self.cfg.memory.get("enabled", False):
            return
        if not self.cfg.memory.get("reflection_enabled", False):
            return
        if str(self.cfg.memory.get("reflection_mode", "per_task")) != "monthly":
            return

        try:
            memory, _ = get_memory_components(self.cfg)
        except Exception as e:
            print(f"    Memory unavailable for monthly reflection: {e}")
            return
        if not memory:
            return

        tasks_by_id = {t.task_id: t for t in month_tasks}
        stocks: list[dict] = []
        entry_date = ""
        available_after = ""
        for result in month_results:
            if isinstance(result, Exception) or not getattr(result, "attempts", None):
                continue
            task = tasks_by_id.get(result.task_id)
            if task is None:
                continue
            judged = next(
                (a for a in result.attempts if a.get("judge_result") in ("CORRECT", "INCORRECT")),
                None,
            )
            if judged is None:
                continue
            meta = task.metadata or {}
            entry_date = meta.get("entry_date", entry_date)
            available_after = max(
                available_after,
                str(meta.get("exit_date", "") or ""),
            )
            predicted = extract_direction(judged.get("model_boxed_answer", ""))
            stocks.append(
                {
                    "ts_code": meta.get("ts_code", ""),
                    "stock_name": meta.get("stock_name", ""),
                    "label": task.ground_truth,
                    "predicted": predicted,
                    "judge_result": judged["judge_result"],
                }
            )
            # Backfill the ledger for cached/resumed tasks that skipped the
            # live post-judge hook (log_outcome dedupes by task_id).
            try:
                memory.log_outcome(
                    task_id=result.task_id,
                    month=month,
                    predicted=predicted,
                    judge_result=judged["judge_result"],
                    available_after=meta.get("exit_date", ""),
                )
            except Exception:
                pass

        if len(stocks) < 4 or not entry_date:
            print(f"    [monthly {month}] only {len(stocks)} judged tasks — reflection skipped")
            return

        try:
            features_csv, n = compute_month_features(entry_date, stocks)
        except Exception as e:
            print(f"    [monthly {month}] feature table failed: {e}")
            return

        try:
            ops = memory.add_monthly(
                month,
                features_csv,
                n,
                available_after=available_after,
            )
            if ops:
                for op in ops:
                    print(f"    📅 monthly[{month}] {op}")
            else:
                print(f"    📅 monthly[{month}] no clear cross-sectional pattern stored")
        except Exception as e:
            print(f"    [monthly {month}] reflection failed: {e}")

    async def _update_log_file_with_evaluation(
        self, log_file_path: Path, evaluation_result: str
    ):
        """Helper method to update log file with evaluation result"""
        try:
            log_file = Path(log_file_path)
            # Read existing data
            with open(log_file, "r", encoding="utf-8") as f:
                log_data = json.load(f)

            # Update with evaluation result
            log_data["judge_result"] = evaluation_result

            # Write to a temporary file and then atomically replace
            temp_log_file = log_file.with_suffix(f"{log_file.suffix}.tmp")
            with open(temp_log_file, "w", encoding="utf-8") as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)

            os.replace(temp_log_file, log_file)
            print(f"    Updated log file {log_file.name} with evaluation result.")
        except Exception as e:
            print(f"    Error updating log file {log_file_path}: {e}")


class JSONLDatasetEvaluator(BenchmarkEvaluator):
    """benchmark evaluator for Gaia like dataset."""

    def __init__(
        self,
        data_dir: str,
        benchmark_name: str,
        cfg: DictConfig,
        metadata_file: str,
        parse_func: Callable[[str], BenchmarkTask],
        filter_func: Callable[[BenchmarkTask], bool],
    ):
        """
        dataset format:
        - a FOLDER (`data_dir`) with a METADATA file (`metadata_file`) and many other binary files.
        - METADATA file are newline separated json objects, parsed by `parse_func` into `BenchmarkTask` objects.
        - `filter_func` is used to filter tasks based on a condition.
        - binary files are referenced by `BenchmarkTask.file_path`.

        Args:
            data_dir: Path to benchmark data directory
            benchmark_name: Name of the benchmark
            cfg: The Hydra configuration object
            parse_func: Function to parse a line of data into a BenchmarkTask object
            filter_func: Function to filter tasks based on a condition
        """
        super().__init__(data_dir=data_dir, benchmark_name=benchmark_name, cfg=cfg)
        self.metadata_file = self.data_dir / metadata_file
        self.parse_func = parse_func
        self.filter_func = filter_func
        self.tasks: List[BenchmarkTask] = []
        self.results: List[BenchmarkResult] = []

    def load_tasks(self) -> List[BenchmarkTask]:
        """
        Load benchmark tasks from metadata.jsonl

        Returns:
            List of BenchmarkTask objects
        """
        print(f"Loading tasks from {self.metadata_file}")

        if not self.metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_file}")

        tasks = []
        with open(self.metadata_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                try:
                    task = self.parse_func(line.strip())
                    if self.filter_func(task):
                        self._attach_trader_anchor_policy(task)
                        tasks.append(task)

                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line {i + 1}: {e}")
                    continue
        tasks = tasks[: self.cfg.benchmark.execution.max_tasks]
        self.tasks = tasks
        print(f"Loaded {len(tasks)} tasks")
        return tasks

    def prepare_task_description(
        self, task: BenchmarkTask
    ) -> Tuple[str, Optional[str]]:
        if task.file_path is None:
            return task.task_question, None

        path = Path(task.file_path)
        # check if task.file_path is a relative path
        if path.is_absolute():
            return task.task_question, str(path)

        # Build complete file path: data directory + relative path
        full_file_path = Path(self.data_dir) / path
        return task.task_question, str(full_file_path)


async def entrypoint(cfg: DictConfig) -> float:
    """
    Main entry point for running benchmarks with Hydra.
    """
    # Keep environment-backed secrets as interpolation expressions in console
    # output instead of resolving and printing API keys.
    print("Benchmark configuration:\n", OmegaConf.to_yaml(cfg, resolve=False))

    def parse_func(x: str) -> BenchmarkTask:
        data = json.loads(x)
        if isinstance(data.get("task_id"), (str, bytes, os.PathLike)) is False:
            try:
                data["task_id"] = str(data["task_id"])
            except TypeError:
                raise TypeError(
                    "expected task_id to be a string, bytes or os.PathLike object"
                )
        return BenchmarkTask(
            task_id=data["task_id"],
            task_question=data["task_question"],
            ground_truth=data["ground_truth"],
            file_path=data.get("file_path"),
            metadata=data.get("metadata", {}),
        )

    def filter_func(x: BenchmarkTask) -> bool:
        if len(cfg.benchmark.data.whitelist) > 0:
            return x.task_id in cfg.benchmark.data.whitelist
        else:
            return True

    evaluator = JSONLDatasetEvaluator(
        data_dir=cfg.benchmark.data.data_dir,
        benchmark_name=cfg.benchmark.name,
        cfg=cfg,
        metadata_file=cfg.benchmark.data.metadata_file,
        parse_func=parse_func,
        filter_func=filter_func,
    )

    """
    Run the full benchmark evaluation process
    """
    print(f"Starting evaluation for benchmark: {cfg.benchmark.name}")

    # Load tasks
    tasks = evaluator.load_tasks()
    if len(evaluator.tasks) == 0:
        print("No tasks loaded. Exiting.")
        return 0.0

    # Run inference
    print(
        f"\nStarting parallel inference with {cfg.benchmark.execution.max_concurrent} concurrent tasks..."
    )
    print(f"Using pass@{evaluator.pass_at_k} evaluation...")
    await evaluator.run_parallel_inference(
        tasks,
        max_concurrent=cfg.benchmark.execution.max_concurrent,
    )

    # Evaluate accuracy
    print("Evaluating accuracy...")
    accuracy = await evaluator.evaluate_accuracy()
    print(f"\nOverall pass@{evaluator.pass_at_k} accuracy: {accuracy:.2%}")
    # Save results

    output_filename = "benchmark_results.jsonl"

    # Construct the full path in the correct log directory
    log_dir = evaluator.output_dir
    results_path = log_dir / output_filename

    evaluator.save_results(results_path)
    print(f"\nEvaluation completed! Results saved to {results_path}")
    # save accuracy to a file
    accuracy_file = (
        results_path.parent
        / f"{results_path.stem}_pass_at_{evaluator.pass_at_k}_accuracy.txt"
    )
    with open(accuracy_file, "w") as f:
        f.write(f"{accuracy:.2%}")

    return accuracy


def setup_hydra_output_dir(cfg: DictConfig, overrides: List[str]) -> DictConfig:
    """Manually creates a Hydra-like output directory and saves the configuration."""
    # Get the base output directory from config
    base_output_dir = Path(cfg.output_dir)

    run_output_dir = base_output_dir
    run_output_dir.mkdir(parents=True, exist_ok=True)

    # Save the composed configuration
    hydra_dir = run_output_dir / ".hydra"
    hydra_dir.mkdir(exist_ok=True)

    with open(hydra_dir / "config.yaml", "w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=False))
    with open(hydra_dir / "overrides.yaml", "w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(overrides))

    print(f"Hydra-like output directory created at: {run_output_dir}")
    return cfg


def signal_handler(signum, frame):
    """Force exit signal handler"""
    print(f"\n⚠️  Received interrupt signal {signum}, forcing immediate exit...")
    print("Program will terminate all operations immediately")
    os._exit(1)  # Force immediate exit


def main(*args, config_file_name: str = ""):
    # Register signal handlers for immediate response to Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    load_project_env()
    LOGGER_LEVEL = os.getenv("LOGGER_LEVEL", "INFO")

    # Support load from config_file_name
    if config_file_name:
        chosen_config_name = config_file_name
    else:
        chosen_config_name = config_name()

    with hydra.initialize_config_dir(
        config_dir=os.path.abspath(config_path()), version_base=None
    ):
        cfg = hydra.compose(config_name=chosen_config_name, overrides=list(args))
        cfg = setup_hydra_output_dir(cfg, list(args))

        _ = bootstrap_logger(level=LOGGER_LEVEL)
        # Tracing functionality removed - miroflow-contrib deleted
        asyncio.run(entrypoint(cfg))
