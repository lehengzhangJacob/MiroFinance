"""Deterministic hard-anchor policy for the unified A-share trader."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence, cast

from src.utils.ashare_market_breadth import compute_market_breadth_regime
from src.utils.ashare_momentum import build_relative_momentum_baseline
from src.utils.ashare_trader_features import compute_trader_feature_rows


ANCHOR_POLICY_VERSION = 2
QLIB_SIGNAL = "qlib_conflict"
POSITION_EPSILON = 1e-8
CORE_SATELLITE_TOP_K = 4

AnchorMode = Literal["legacy", "core_satellite"]
MarketRegime = Literal["risk_on", "neutral", "defensive"]
PredictionCandidate = str | Mapping[str, Any] | tuple[str, float]
PredictionCandidates = Sequence[PredictionCandidate] | Mapping[str, Any]


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


@dataclass(frozen=True)
class CoreSatelliteSleeve:
    """Exact fixed weights for one market-breadth regime."""

    regime: MarketRegime
    core_stock_weight: float
    satellite_stock_weight: float
    satellite_count: int
    cash_weight: float

    def __post_init__(self) -> None:
        if self.regime not in ("risk_on", "neutral", "defensive"):
            raise ValueError(f"unsupported core-satellite regime {self.regime!r}")
        weights = (
            self.core_stock_weight,
            self.satellite_stock_weight,
            self.cash_weight,
        )
        if any(_number(value) is None or float(value) < 0.0 for value in weights):
            raise ValueError(
                "core-satellite sleeve weights must be finite and non-negative"
            )
        if int(self.satellite_count) <= 0 or float(self.satellite_count) != float(
            int(self.satellite_count)
        ):
            raise ValueError("core-satellite sleeve requires at least one satellite")
        if abs(self.total_weight - 1.0) > POSITION_EPSILON:
            raise ValueError(f"core-satellite sleeve for {self.regime} must sum to 1")

    @property
    def core_total_weight(self) -> float:
        return round(float(self.core_stock_weight) * CORE_SATELLITE_TOP_K, 10)

    @property
    def satellite_total_weight(self) -> float:
        return round(
            float(self.satellite_stock_weight) * int(self.satellite_count),
            10,
        )

    @property
    def total_weight(self) -> float:
        return (
            self.core_total_weight
            + self.satellite_total_weight
            + float(self.cash_weight)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "core_total_weight": self.core_total_weight,
            "satellite_total_weight": self.satellite_total_weight,
        }


CORE_SATELLITE_SLEEVES: dict[MarketRegime, CoreSatelliteSleeve] = {
    "risk_on": CoreSatelliteSleeve(
        regime="risk_on",
        core_stock_weight=0.20,
        satellite_stock_weight=0.10,
        satellite_count=2,
        cash_weight=0.0,
    ),
    "neutral": CoreSatelliteSleeve(
        regime="neutral",
        core_stock_weight=0.225,
        satellite_stock_weight=0.10,
        satellite_count=1,
        cash_weight=0.0,
    ),
    "defensive": CoreSatelliteSleeve(
        regime="defensive",
        core_stock_weight=0.225,
        satellite_stock_weight=0.05,
        satellite_count=1,
        cash_weight=0.05,
    ),
}


def get_core_satellite_sleeve(regime: str) -> CoreSatelliteSleeve:
    """Return the fixed sleeve for a normalized breadth regime."""
    normalized = str(regime or "").strip().lower()
    if normalized not in CORE_SATELLITE_SLEEVES:
        supported = ", ".join(CORE_SATELLITE_SLEEVES)
        raise ValueError(
            f"unsupported market regime {regime!r}; expected one of {supported}"
        )
    return CORE_SATELLITE_SLEEVES[cast(MarketRegime, normalized)]


@dataclass(frozen=True)
class AnchorPolicy:
    enabled: bool = False
    min_top4_weight: float = 0.60
    momentum_window: int = 20
    top_k: int = 4
    min_top4_holdings: int = 3
    max_non_top4_holdings: int = 1
    min_independent_risk_signals: int = 2
    max_stock_weight: float = 0.25
    mode: AnchorMode = "legacy"

    def __post_init__(self) -> None:
        if self.mode not in ("legacy", "core_satellite"):
            raise ValueError("mode must be 'legacy' or 'core_satellite'")
        if int(self.top_k) != CORE_SATELLITE_TOP_K:
            raise ValueError("the unified trader hard anchor requires top_k=4")
        if self.mode == "core_satellite":
            largest_fixed_weight = max(
                max(sleeve.core_stock_weight, sleeve.satellite_stock_weight)
                for sleeve in CORE_SATELLITE_SLEEVES.values()
            )
            max_stock_weight = _number(self.max_stock_weight)
            if (
                max_stock_weight is None
                or max_stock_weight + POSITION_EPSILON < largest_fixed_weight
                or max_stock_weight > 0.25 + POSITION_EPSILON
            ):
                raise ValueError(
                    "core_satellite max_stock_weight must be within [0.225, 0.25]"
                )
            return

        if not 0.0 <= float(self.min_top4_weight) <= 1.0:
            raise ValueError("min_top4_weight must be within [0, 1]")
        if not 1 <= int(self.min_top4_holdings) <= int(self.top_k):
            raise ValueError("min_top4_holdings must be between 1 and top_k")
        if int(self.max_non_top4_holdings) < 0:
            raise ValueError("max_non_top4_holdings cannot be negative")
        if int(self.min_independent_risk_signals) < 2:
            raise ValueError("at least two independent risk signals are required")
        capacity = float(self.max_stock_weight) * int(self.min_top4_holdings)
        if float(self.min_top4_weight) > capacity + POSITION_EPSILON:
            raise ValueError(
                "min_top4_weight exceeds the capacity of min_top4_holdings"
            )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "AnchorPolicy":
        values = dict(raw or {})
        allowed = set(cls.__dataclass_fields__)
        return cls(**{key: values[key] for key in allowed if key in values})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def required_retained_weight(self) -> float:
        """Legacy undeviated weight of each equal-weight momentum leader."""
        return min(
            float(self.max_stock_weight),
            1.0 / int(self.top_k),
        )


@dataclass(frozen=True)
class AnchorValidationResult:
    ok: bool
    errors: tuple[str, ...]
    metrics: dict[str, Any]

    @property
    def error(self) -> str:
        return "; ".join(self.errors)


def _normalize_code(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().upper()


def _candidate_code(row: Mapping[str, Any]) -> str:
    for key in ("ts_code", "code", "symbol"):
        code = _normalize_code(row.get(key))
        if code:
            return code
    return ""


def _candidate_value(
    row: Mapping[str, Any],
    keys: Sequence[str],
) -> float | None:
    for key in keys:
        value = _number(row.get(key))
        if value is not None:
            return value
    return None


def _deduplicate_codes(codes: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for raw_code in codes:
        code = _normalize_code(raw_code)
        if code and code not in seen:
            seen.add(code)
            result.append(code)
    return result


def _rank_prediction_candidates(
    candidates: PredictionCandidates | None,
) -> list[str]:
    """Normalize candidate inputs into a deterministic best-first code list.

    A sequence of codes is already ranked and keeps its order. Score mappings
    and structured rows are sorted by rank ascending or score descending, with
    the stock code as the stable tie-breaker.
    """
    if candidates is None:
        return []
    if isinstance(candidates, (str, bytes)):
        return _deduplicate_codes([str(candidates)])

    records: list[tuple[str, float | None, float | None, int]] = []
    preserve_unranked_order = not isinstance(candidates, Mapping)

    if isinstance(candidates, Mapping):
        single_code = _candidate_code(candidates)
        raw_items: Sequence[Any]
        if single_code:
            raw_items = [candidates]
        else:
            mapped_items: list[Mapping[str, Any] | tuple[str, Any]] = []
            for code, value in candidates.items():
                if isinstance(value, Mapping):
                    row = dict(value)
                    row.setdefault("ts_code", code)
                    mapped_items.append(row)
                else:
                    mapped_items.append((_normalize_code(code), value))
            raw_items = mapped_items
    else:
        raw_items = list(candidates)
        if all(isinstance(item, str) for item in raw_items):
            return _deduplicate_codes([_normalize_code(item) for item in raw_items])

    for position, item in enumerate(raw_items):
        code = ""
        rank: float | None = None
        score: float | None = None
        if isinstance(item, str):
            code = _normalize_code(item)
        elif isinstance(item, Mapping):
            code = _candidate_code(item)
            rank = _candidate_value(
                item,
                ("rank", "prediction_rank", "ml_rank"),
            )
            score = _candidate_value(
                item,
                (
                    "score",
                    "prediction_score",
                    "predicted_excess_return",
                    "ml_score",
                ),
            )
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            if item:
                code = _normalize_code(item[0])
            if len(item) > 1:
                score = _number(item[1])
        if code:
            records.append((code, rank, score, position))

    def sort_key(
        record: tuple[str, float | None, float | None, int],
    ) -> tuple[int, float, str]:
        code, rank, score, position = record
        if rank is not None:
            return (0, rank, code)
        if score is not None:
            return (1, -score, code)
        if preserve_unranked_order:
            return (2, float(position), code)
        return (2, 0.0, code)

    ordered = [record[0] for record in sorted(records, key=sort_key)]
    return _deduplicate_codes(ordered)


def _snapshot_regime(snapshot: Mapping[str, Any]) -> str:
    breadth = snapshot.get("market_breadth", {})
    if isinstance(breadth, Mapping):
        return str(breadth.get("regime", "")).strip().lower()
    return ""


def _snapshot_ranked_candidates(snapshot: Mapping[str, Any]) -> list[str]:
    for key in (
        "ranked_prediction_candidates",
        "prediction_candidates",
        "eligible_prediction_candidates",
        "expected_satellite_candidates",
        "expected_candidates",
    ):
        if key in snapshot:
            raw = snapshot.get(key)
            if isinstance(raw, (Mapping, Sequence)) and not isinstance(raw, bytes):
                return _rank_prediction_candidates(raw)
    return []


def _eligible_prediction_candidates(
    snapshot: Mapping[str, Any],
    top4: Sequence[str],
) -> list[str]:
    top4_set = {_normalize_code(code) for code in top4}
    return [
        code for code in _snapshot_ranked_candidates(snapshot) if code not in top4_set
    ]


def _percentile(values: list[float], quantile: float) -> float | None:
    clean = sorted(value for value in values if math.isfinite(value))
    if not clean:
        return None
    position = (len(clean) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return clean[lower]
    fraction = position - lower
    return clean[lower] * (1.0 - fraction) + clean[upper] * fraction


def classify_anchor_risk_signals(
    row: Mapping[str, Any],
    *,
    vol20_q75: float | None,
    pool_size: int,
) -> dict[str, Any]:
    """Return independent, pre-declared risk categories for one anchor stock."""
    rel5 = _number(row.get("rel5"))
    rel60 = _number(row.get("rel60"))
    ma20_gap = _number(row.get("ma20_gap"))
    vol20 = _number(row.get("vol20_ann"))
    max_dd120 = _number(row.get("max_dd120"))
    from_high250 = _number(row.get("from_high250"))
    amount_ratio = _number(row.get("amount20_vs120"))
    pe_pct = _number(row.get("pe_pct250"))
    pb_pct = _number(row.get("pb_pct250"))
    revenue_yoy = _number(row.get("or_yoy"))
    profit_yoy = _number(row.get("netprofit_yoy"))
    ml_rank = _number(row.get("ml_rank"))

    flags = {
        # Require a short-horizon reversal and a visible price/trend break;
        # two correlated observations remain one category.
        "trend_break": bool(
            rel5 is not None
            and rel5 < 0.0
            and (
                (ma20_gap is not None and ma20_gap < 0.0)
                or (rel60 is not None and rel60 <= 0.0)
            )
        ),
        # High volatility alone is not enough: it must accompany a material
        # drawdown or distance from the trailing high.
        "tail_risk": bool(
            vol20 is not None
            and vol20_q75 is not None
            and vol20 >= vol20_q75
            and (
                (max_dd120 is not None and max_dd120 <= -15.0)
                or (from_high250 is not None and from_high250 <= -15.0)
            )
        ),
        "volume_anomaly": bool(
            amount_ratio is not None
            and (
                amount_ratio <= 0.65
                or (amount_ratio >= 1.80 and rel5 is not None and rel5 < 0.0)
            )
        ),
        "valuation_overheat": bool(
            (pe_pct is not None and pe_pct >= 90.0)
            or (pb_pct is not None and pb_pct >= 90.0)
        ),
        "fundamental_deterioration": bool(
            (revenue_yoy is not None and revenue_yoy <= -10.0)
            or (profit_yoy is not None and profit_yoy <= -10.0)
        ),
        "qlib_conflict": bool(
            ml_rank is not None
            and pool_size > 0
            and ml_rank > math.ceil(pool_size * 0.75)
        ),
    }
    active = [name for name, enabled in flags.items() if enabled]
    return {
        "active": active,
        "count": len(active),
        "non_qlib_count": sum(name != QLIB_SIGNAL for name in active),
        "flags": flags,
    }


def build_anchor_snapshot(
    as_of: str,
    stock_pool: Mapping[str, Mapping[str, Any]],
    *,
    data_dir: str | Path,
    policy: AnchorPolicy,
    prediction_candidates: PredictionCandidates | None = None,
) -> dict[str, Any]:
    """Build a hidden, PIT-safe snapshot used by prompts and hard validation."""
    baseline = build_relative_momentum_baseline(
        as_of,
        stock_pool,
        data_dir=data_dir,
        window=policy.momentum_window,
        top_k=policy.top_k,
        max_stock_weight=policy.max_stock_weight,
    )
    rows = compute_trader_feature_rows(
        as_of,
        stock_pool,
        data_dir=data_dir,
        lookback_days=0,
    )
    by_code = {str(row["ts_code"]): row for row in rows}
    vol_values = [
        value for row in rows if (value := _number(row.get("vol20_ann"))) is not None
    ]
    vol_q75 = _percentile(vol_values, 0.75)
    top4 = list(baseline["weights"])
    if len(top4) != policy.top_k:
        raise ValueError(
            f"hard anchor requires {policy.top_k} eligible stocks, got {len(top4)}"
        )
    risk_signals = {
        code: classify_anchor_risk_signals(
            by_code.get(code, {}),
            vol20_q75=vol_q75,
            pool_size=len(stock_pool),
        )
        for code in top4
    }
    breadth = compute_market_breadth_regime(as_of, data_dir=data_dir)
    snapshot: dict[str, Any] = {
        "version": ANCHOR_POLICY_VERSION,
        "as_of": baseline["as_of"],
        "top4": top4,
        "anchor_weights": dict(baseline["weights"]),
        "anchor_cash": float(baseline["cash"]),
        "risk_signals": risk_signals,
        "market_breadth": breadth,
    }
    if policy.mode == "core_satellite":
        sleeve = get_core_satellite_sleeve(str(breadth.get("regime", "")))
        snapshot["policy_mode"] = policy.mode
        snapshot["core_satellite_sleeve"] = sleeve.to_dict()
    if prediction_candidates is not None:
        ranked_candidates = _rank_prediction_candidates(prediction_candidates)
        normalized_top4 = {_normalize_code(code) for code in top4}
        eligible_candidates = [
            code for code in ranked_candidates if code not in normalized_top4
        ]
        snapshot["prediction_candidates"] = ranked_candidates
        snapshot["ranked_prediction_candidates"] = ranked_candidates
        snapshot["eligible_prediction_candidates"] = eligible_candidates
        if policy.mode == "core_satellite":
            sleeve = get_core_satellite_sleeve(str(breadth.get("regime", "")))
            snapshot["expected_satellite_candidates"] = eligible_candidates[
                : sleeve.satellite_count
            ]
    return snapshot


def assemble_core_satellite_allocation(
    snapshot: Mapping[str, Any],
    selected_satellites: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build the canonical fixed-weight allocation for a core-satellite snapshot."""
    regime = _snapshot_regime(snapshot)
    sleeve = get_core_satellite_sleeve(regime)
    top4 = [_normalize_code(code) for code in snapshot.get("top4", [])]
    if (
        len(top4) != CORE_SATELLITE_TOP_K
        or len(set(top4)) != CORE_SATELLITE_TOP_K
        or not all(top4)
    ):
        raise ValueError(
            "core-satellite snapshot requires four unique momentum leaders"
        )

    eligible_candidates = _eligible_prediction_candidates(snapshot, top4)
    expected_candidates = eligible_candidates[: sleeve.satellite_count]
    if len(expected_candidates) != sleeve.satellite_count:
        raise ValueError(
            f"{regime} core-satellite sleeve requires "
            f"{sleeve.satellite_count} eligible prediction candidates, "
            f"got {len(eligible_candidates)}"
        )

    if selected_satellites is None:
        satellite_codes = expected_candidates
    else:
        raw_selected = (
            [selected_satellites]
            if isinstance(selected_satellites, str)
            else list(selected_satellites)
        )
        normalized_selected = [
            _normalize_code(code) for code in raw_selected if _normalize_code(code)
        ]
        if len(normalized_selected) != len(set(normalized_selected)):
            raise ValueError("selected satellites must be unique")
        if len(normalized_selected) != sleeve.satellite_count:
            raise ValueError(
                f"{regime} requires exactly {sleeve.satellite_count} satellites"
            )
        top4_set = set(top4)
        overlapping = [code for code in normalized_selected if code in top4_set]
        if overlapping:
            raise ValueError(
                "satellites must be non-top4 stocks: " + ",".join(overlapping)
            )
        ineligible = [
            code for code in normalized_selected if code not in eligible_candidates
        ]
        if ineligible:
            raise ValueError(
                "satellites are not eligible prediction candidates: "
                + ",".join(ineligible)
            )
        rank_by_code = {code: rank for rank, code in enumerate(eligible_candidates)}
        satellite_codes = sorted(
            normalized_selected,
            key=lambda code: (rank_by_code[code], code),
        )

    core_weights = {code: float(sleeve.core_stock_weight) for code in top4}
    satellite_weights = {
        code: float(sleeve.satellite_stock_weight) for code in satellite_codes
    }
    weights = {**core_weights, **satellite_weights}
    return {
        "mode": "core_satellite",
        "regime": regime,
        "weights": weights,
        "cash": float(sleeve.cash_weight),
        "top4": top4,
        "core_weights": core_weights,
        "core_weight": sleeve.core_total_weight,
        "core_total_weight": sleeve.core_total_weight,
        "satellite_codes": satellite_codes,
        "satellite_count": len(satellite_codes),
        "satellite_weights": satellite_weights,
        "satellite_weight": sleeve.satellite_total_weight,
        "satellite_total_weight": sleeve.satellite_total_weight,
        "eligible_prediction_candidates": eligible_candidates,
        "expected_candidates": expected_candidates,
        "sleeve": sleeve.to_dict(),
    }


def validate_core_satellite_allocation(
    weights: Mapping[str, float],
    cash: float,
    *,
    snapshot: Mapping[str, Any],
    policy: AnchorPolicy,
) -> AnchorValidationResult:
    """Validate the exact core, satellite, and cash sleeves for one regime."""
    if not policy.enabled:
        return AnchorValidationResult(True, (), {})
    if policy.mode != "core_satellite":
        return AnchorValidationResult(
            False,
            ("core-satellite validation requires mode='core_satellite'",),
            {"mode": str(policy.mode)},
        )

    errors: list[str] = []
    top4 = [_normalize_code(code) for code in snapshot.get("top4", [])]
    top4_set = set(top4)
    if (
        len(top4) != CORE_SATELLITE_TOP_K
        or len(top4_set) != CORE_SATELLITE_TOP_K
        or not all(top4)
    ):
        errors.append(
            "anchor snapshot is missing four unique eligible momentum leaders"
        )

    regime = _snapshot_regime(snapshot)
    try:
        sleeve = get_core_satellite_sleeve(regime)
    except ValueError as exc:
        sleeve = None
        errors.append(str(exc))

    normalized: dict[str, float] = {}
    for raw_code, raw_weight in weights.items():
        code = _normalize_code(raw_code)
        weight = _number(raw_weight)
        if not code:
            errors.append("allocation contains an empty stock code")
            continue
        if weight is None:
            errors.append(f"{code} has a non-finite weight")
            continue
        if code in normalized:
            errors.append(f"allocation contains duplicate normalized code {code}")
            continue
        normalized[code] = weight

    cash_value = _number(cash)
    if cash_value is None:
        errors.append("cash has a non-finite weight")
        cash_value = 0.0

    for code, weight in normalized.items():
        if weight < -POSITION_EPSILON:
            errors.append(f"{code} weight {weight:.2%} cannot be negative")
        if weight > float(policy.max_stock_weight) + POSITION_EPSILON:
            errors.append(
                f"{code} weight {weight:.2%} exceeds "
                f"{float(policy.max_stock_weight):.2%}"
            )

    eligible_candidates = _eligible_prediction_candidates(snapshot, top4)
    expected_count = sleeve.satellite_count if sleeve is not None else 0
    expected_candidates = eligible_candidates[:expected_count]
    if sleeve is not None and len(expected_candidates) != expected_count:
        errors.append(
            f"{regime} requires {expected_count} eligible prediction candidates, "
            f"but snapshot has {len(eligible_candidates)}"
        )

    candidate_rank = {code: rank for rank, code in enumerate(eligible_candidates)}
    satellite_codes = sorted(
        (
            code
            for code, weight in normalized.items()
            if code not in top4_set and weight > POSITION_EPSILON
        ),
        key=lambda code: (candidate_rank.get(code, len(candidate_rank)), code),
    )
    core_weights = {code: round(normalized.get(code, 0.0), 10) for code in top4}
    satellite_weights = {
        code: round(normalized.get(code, 0.0), 10) for code in satellite_codes
    }
    core_weight = sum(normalized.get(code, 0.0) for code in top4)
    satellite_weight = sum(normalized.get(code, 0.0) for code in satellite_codes)
    held_top4 = [code for code in top4 if normalized.get(code, 0.0) > POSITION_EPSILON]

    expected_core_stock_weight = (
        float(sleeve.core_stock_weight) if sleeve is not None else None
    )
    expected_satellite_stock_weight = (
        float(sleeve.satellite_stock_weight) if sleeve is not None else None
    )
    expected_core_weight = sleeve.core_total_weight if sleeve is not None else None
    expected_satellite_weight = (
        sleeve.satellite_total_weight if sleeve is not None else None
    )
    expected_cash = float(sleeve.cash_weight) if sleeve is not None else None

    metrics = {
        "mode": "core_satellite",
        "regime": regime,
        "market_regime": regime,
        "top4": top4,
        "top4_exposure": round(core_weight, 10),
        "top4_holding_count": len(held_top4),
        "core_weights": core_weights,
        "core_weight": round(core_weight, 10),
        "core_total_weight": round(core_weight, 10),
        "expected_core_stock_weight": expected_core_stock_weight,
        "expected_core_weight": expected_core_weight,
        "satellite_codes": satellite_codes,
        "satellite_count": len(satellite_codes),
        "satellite_weights": satellite_weights,
        "satellite_weight": round(satellite_weight, 10),
        "satellite_total_weight": round(satellite_weight, 10),
        "expected_satellite_count": expected_count,
        "expected_satellite_stock_weight": expected_satellite_stock_weight,
        "expected_satellite_weight": expected_satellite_weight,
        "non_top4_holdings": satellite_codes,
        "replacement_count": len(satellite_codes),
        "eligible_prediction_candidates": eligible_candidates,
        "expected_candidates": expected_candidates,
        "expected_satellite_candidates": expected_candidates,
        "cash": float(cash_value),
        "expected_cash": expected_cash,
        "total_weight": round(sum(normalized.values()) + cash_value, 10),
    }

    if sleeve is not None:
        if len(held_top4) != CORE_SATELLITE_TOP_K:
            errors.append(
                f"all four momentum leaders are mandatory; held {len(held_top4)}"
            )
        for code in top4:
            actual = normalized.get(code, 0.0)
            if abs(actual - sleeve.core_stock_weight) > POSITION_EPSILON:
                errors.append(
                    f"{code} core weight {actual:.3%} must equal "
                    f"{sleeve.core_stock_weight:.3%} in {regime}"
                )
        if abs(core_weight - sleeve.core_total_weight) > POSITION_EPSILON:
            errors.append(
                f"core weight {core_weight:.2%} must equal "
                f"{sleeve.core_total_weight:.2%} in {regime}"
            )

        if len(satellite_codes) != sleeve.satellite_count:
            errors.append(
                f"{regime} requires exactly {sleeve.satellite_count} "
                f"satellites; held {len(satellite_codes)}"
            )
        ineligible = [
            code for code in satellite_codes if code not in eligible_candidates
        ]
        if ineligible:
            errors.append(
                "satellites are not eligible prediction candidates: "
                + ",".join(ineligible)
            )
        for code in satellite_codes:
            actual = normalized[code]
            if abs(actual - sleeve.satellite_stock_weight) > POSITION_EPSILON:
                errors.append(
                    f"{code} satellite weight {actual:.3%} must equal "
                    f"{sleeve.satellite_stock_weight:.3%} in {regime}"
                )
        if abs(satellite_weight - sleeve.satellite_total_weight) > POSITION_EPSILON:
            errors.append(
                f"satellite weight {satellite_weight:.2%} must equal "
                f"{sleeve.satellite_total_weight:.2%} in {regime}"
            )
        if abs(cash_value - sleeve.cash_weight) > POSITION_EPSILON:
            errors.append(
                f"cash weight {cash_value:.2%} must equal "
                f"{sleeve.cash_weight:.2%} in {regime}"
            )
        total_weight = sum(normalized.values()) + cash_value
        if abs(total_weight - 1.0) > POSITION_EPSILON:
            errors.append(
                f"core-satellite allocation must sum to 100%, got "
                f"{total_weight:.2%}"
            )

    return AnchorValidationResult(not errors, tuple(errors), metrics)


def validate_anchor_allocation(
    weights: Mapping[str, float],
    cash: float,
    *,
    snapshot: Mapping[str, Any],
    policy: AnchorPolicy,
) -> AnchorValidationResult:
    """Validate economic hard-anchor constraints after syntax validation."""
    if not policy.enabled:
        return AnchorValidationResult(True, (), {})
    if policy.mode == "core_satellite":
        return validate_core_satellite_allocation(
            weights,
            cash,
            snapshot=snapshot,
            policy=policy,
        )

    top4 = [str(code) for code in snapshot.get("top4", [])]
    top4_set = set(top4)
    if len(top4) != policy.top_k:
        return AnchorValidationResult(
            False,
            ("anchor snapshot is missing four eligible momentum leaders",),
            {},
        )

    normalized = {str(code): float(value) for code, value in weights.items()}
    top4_exposure = sum(normalized.get(code, 0.0) for code in top4)
    held_top4 = [code for code in top4 if normalized.get(code, 0.0) > POSITION_EPSILON]
    non_top4 = [
        code
        for code, weight in normalized.items()
        if code not in top4_set and float(weight) > POSITION_EPSILON
    ]
    errors: list[str] = []
    if top4_exposure + POSITION_EPSILON < policy.min_top4_weight:
        errors.append(
            f"momentum top4 exposure {top4_exposure:.2%} is below "
            f"{policy.min_top4_weight:.0%}"
        )
    if len(held_top4) < policy.min_top4_holdings:
        errors.append(
            f"only {len(held_top4)} momentum top4 names retained; "
            f"need at least {policy.min_top4_holdings}"
        )
    if len(non_top4) > policy.max_non_top4_holdings:
        errors.append(
            f"{len(non_top4)} non-top4 stocks held ({','.join(non_top4)}); "
            f"at most {policy.max_non_top4_holdings} replacement is allowed"
        )

    risk_signals = snapshot.get("risk_signals", {})
    underweight_top4: list[str] = []
    for code in top4:
        weight = normalized.get(code, 0.0)
        if weight + POSITION_EPSILON >= policy.required_retained_weight:
            continue
        underweight_top4.append(code)
        audit = risk_signals.get(code, {})
        active = [str(name) for name in audit.get("active", [])]
        non_qlib = [name for name in active if name != QLIB_SIGNAL]
        if len(active) < policy.min_independent_risk_signals or not non_qlib:
            shown = ",".join(active) if active else "none"
            errors.append(
                f"{code} weight {weight:.2%} is below the retained-leader "
                f"threshold {policy.required_retained_weight:.2%}, but only "
                f"risk categories [{shown}] are active; need at least "
                f"{policy.min_independent_risk_signals} independent categories "
                "and Qlib cannot stand alone"
            )

    metrics = {
        "top4": top4,
        "top4_exposure": round(top4_exposure, 10),
        "top4_holding_count": len(held_top4),
        "non_top4_holdings": non_top4,
        "replacement_count": len(non_top4),
        "underweight_top4": underweight_top4,
        "cash": float(cash),
        "market_regime": str(snapshot.get("market_breadth", {}).get("regime", "")),
    }
    return AnchorValidationResult(not errors, tuple(errors), metrics)


def validate_anchor_from_metadata(
    weights: Mapping[str, float],
    cash: float,
    metadata: Mapping[str, Any] | None,
) -> AnchorValidationResult:
    raw = dict(metadata or {})
    policy = AnchorPolicy.from_mapping(raw.get("anchor_policy"))
    if not policy.enabled:
        return AnchorValidationResult(True, (), {})
    return validate_anchor_allocation(
        weights,
        cash,
        snapshot=raw.get("anchor_snapshot", {}),
        policy=policy,
    )


def render_anchor_policy_prompt(policy: AnchorPolicy) -> str:
    """Return the identical hard-policy block injected in both repositories."""
    if not policy.enabled:
        return ""
    if policy.mode == "core_satellite":
        risk_on = CORE_SATELLITE_SLEEVES["risk_on"]
        neutral = CORE_SATELLITE_SLEEVES["neutral"]
        defensive = CORE_SATELLITE_SLEEVES["defensive"]
        return f"""

## 动量硬锚（系统将确定性复核）—核心 + 预测卫星
- 必须先调用 `ashare_momentum_baseline(as_of, window=20, top_k=4)`，再调用
  `ashare_market_breadth(as_of)`；不得手算或跳过。
- 四只 rel20 动量 top4 全部必选且固定等权；它们是不可替换的核心仓。预测卫星
  必须来自系统提供的合格预测候选，且不得与 top4 重合。
- 固定仓位由市场状态唯一决定，不得自行改变权重：
  - risk_on：top4 每只 {risk_on.core_stock_weight:.0%}（核心
    {risk_on.core_total_weight:.0%}），预测卫星
    {risk_on.satellite_count} 只每只 {risk_on.satellite_stock_weight:.0%}，
    现金 {risk_on.cash_weight:.0%}。
  - neutral：top4 每只 {neutral.core_stock_weight:.1%}（核心
    {neutral.core_total_weight:.0%}），预测卫星
    {neutral.satellite_count} 只 {neutral.satellite_stock_weight:.0%}，
    现金 {neutral.cash_weight:.0%}。
  - defensive：top4 每只 {defensive.core_stock_weight:.1%}（核心
    {defensive.core_total_weight:.0%}），预测卫星
    {defensive.satellite_count} 只 {defensive.satellite_stock_weight:.0%}，
    现金 {defensive.cash_weight:.0%}。
- 若需要选择卫星，只选择规定数量的候选代码；系统将确定性组装并复核固定权重。
  缺失或无效选择应回退到候选排名最前的合格非 top4 股票。
- 正文说明市场状态、四只核心股和卫星选择；最后一行仍只输出原规定的 boxed 组合。
""".rstrip()
    return f"""

## 动量硬锚（系统将确定性复核）
- 必须先调用 `ashare_momentum_baseline(as_of, window=20, top_k=4)`，再调用
  `ashare_market_breadth(as_of)`；不得手算或跳过。
- 最终组合对 rel20 top4 的总仓位不得低于 {policy.min_top4_weight:.0%}，
  至少保留 {policy.min_top4_holdings} 只 top4，非 top4 股票最多
  {policy.max_non_top4_holdings} 只；每只股票仍不得超过 25%。
- top4 等权锚点为每只 {policy.required_retained_weight:.0%}；任何低于该权重都
  视为降配。每个被降配
  成分必须同时出现至少 {policy.min_independent_risk_signals} 类独立、当前可见的
  风险：趋势破坏、波动/尾部、量价异常、估值过热、已公告基本面恶化、Qlib 冲突。
  Qlib 最多算一类，绝不能单独否决动量股。
- 全A市场广度只决定 risk_on/neutral/defensive 语境和现金解释，不降低硬锚下限，
  也不能代替个股双风险证据。
- 正文逐项列出 top4 暴露、保留/替换项、每个降配项的独立风险类别及市场状态；
  最后一行仍只输出原规定的 boxed 组合。
""".rstrip()
