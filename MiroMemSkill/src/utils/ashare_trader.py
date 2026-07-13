"""Parsing and deterministic portfolio math for ``ashare-trader``."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from src.utils.ashare_anchor import (
    AnchorPolicy,
    AnchorValidationResult,
    assemble_core_satellite_allocation,
    validate_anchor_from_metadata,
)


DEFAULT_MAX_STOCK_WEIGHT = 0.25
DEFAULT_OPEN_COST = 0.0005
DEFAULT_CLOSE_COST = 0.0015
DEFAULT_MIN_COST = 5.0
WEIGHT_TOLERANCE = 1e-6


@dataclass(frozen=True)
class PortfolioParseResult:
    weights: dict[str, float]
    cash: float
    ok: bool
    error: str = ""


@dataclass(frozen=True)
class PortfolioValidationResult:
    parsed: PortfolioParseResult
    anchor: AnchorValidationResult | None = None

    @property
    def ok(self) -> bool:
        return self.parsed.ok and bool(self.anchor is None or self.anchor.ok)

    @property
    def error(self) -> str:
        if not self.parsed.ok:
            return self.parsed.error
        return self.anchor.error if self.anchor is not None else ""


@dataclass(frozen=True)
class CoreSatelliteCanonicalizationResult:
    canonical_boxed_answer: str
    weights: dict[str, float]
    cash: float
    selected_codes: tuple[str, ...]
    deterministic_codes: tuple[str, ...]
    selection_fallback: bool
    selection_source: str
    diagnostic: str

    @property
    def boxed_answer(self) -> str:
        """Compatibility alias for the canonical boxed allocation."""
        return self.canonical_boxed_answer

    @property
    def source(self) -> str:
        """Metadata-ready selection source."""
        return self.selection_source


@dataclass(frozen=True)
class FixedCoreCanonicalizationResult:
    """Canonical allocation assembled from fixed assets plus Agent stock intent."""

    canonical_boxed_answer: str
    weights: dict[str, float]
    cash: float
    selected_codes: tuple[str, ...]
    diagnostic: str


@dataclass(frozen=True)
class PortfolioMonthResult:
    starting_capital: float
    ending_capital: float
    gross_return: float
    net_return: float
    index_return: float
    active_return: float
    buy_cost: float
    sell_cost: float
    total_cost: float
    gross_traded_notional: float
    cash_weight: float
    invested_weight: float
    holding_count: int
    concentration_hhi: float
    weight_rank_ic: float
    contributions: dict[str, float] = field(default_factory=dict)


_LATEX_WRAPPER_PATTERN = re.compile(
    r"\\(?:text|textbf|textit|mathrm|mathtt|mathsf|mathbf|operatorname)"
    r"\s*\{([^{}]*)\}"
)
_LATEX_NOISE_PATTERN = re.compile(r"\\[,;!: ]|\$")
_ALLOCATION_TOKEN_PATTERN = re.compile(
    r"\s*(CASH|现金|\d{6}(?:\.(?:SH|SZ))?)\s*[:=：]\s*"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*",
    flags=re.IGNORECASE,
)


def _extract_last_boxed(text: str) -> str | None:
    """Return the last ``\\boxed{...}`` body using balanced-brace scanning.

    The previous regex ``\\boxed\\{([^{}]*)\\}`` silently failed on nested
    braces (e.g. ``\\boxed{600015.\\text{SH}, w=1.00}``), causing validation
    to fall back to the full prose and emit misleading errors.
    """
    marker = r"\boxed{"
    matches: list[str] = []
    index = 0
    while True:
        start = text.find(marker, index)
        if start == -1:
            break
        depth = 1
        cursor = start + len(marker)
        while cursor < len(text) and depth:
            char = text[cursor]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
            cursor += 1
        if depth:
            index = start + len(marker)
            continue
        matches.append(text[start + len(marker) : cursor - 1])
        index = cursor
    return matches[-1] if matches else None


def _normalize_boxed_latex(content: str) -> str:
    """Unwrap LaTeX text commands and strip spacing noise inside a boxed body."""
    result = content
    for _ in range(3):
        unwrapped = _LATEX_WRAPPER_PATTERN.sub(r"\1", result)
        if unwrapped == result:
            break
        result = unwrapped
    return _LATEX_NOISE_PATTERN.sub("", result).strip()


def _last_raw_allocation_line(raw: str) -> str | None:
    """Find a bare terminal allocation line when the model omitted ``\boxed``."""
    for line in reversed(raw.splitlines()):
        candidate = line.strip().strip("`").strip()
        if not candidate:
            continue
        tokens = [
            token.strip()
            for token in re.split(r"[,，;；]+", candidate)
            if token.strip()
        ]
        if (
            len(tokens) >= 2
            and all(_ALLOCATION_TOKEN_PATTERN.fullmatch(token) for token in tokens)
            and any(
                str(match.group(1)).upper() in {"CASH", "现金"}
                for token in tokens
                if (match := _ALLOCATION_TOKEN_PATTERN.fullmatch(token))
            )
        ):
            return candidate
    return None


def _boxed_candidate(text: str) -> str:
    raw = str(text or "").strip()
    boxed = _extract_last_boxed(raw)
    # The candidate may arrive already unwrapped (OutputFormatter strips the
    # \boxed{} shell before terminal validation), so normalize either way.
    if boxed is not None:
        return _normalize_boxed_latex(boxed.strip())
    bare = _last_raw_allocation_line(raw)
    return _normalize_boxed_latex((bare if bare is not None else raw).strip())


_STOCK_CODE_INTENT_PATTERN = re.compile(
    r"(?<!\d)(\d{6}(?:\.(?:SH|SZ))?)(?![A-Z0-9])",
    flags=re.IGNORECASE,
)


def _canonical_pool(
    valid_pool: Any,
) -> tuple[list[str], set[str], dict[str, str]]:
    if not isinstance(valid_pool, Sequence) or isinstance(
        valid_pool, (str, bytes)
    ):
        raise ValueError(
            "core-satellite canonicalization requires stock_pool to be a sequence"
        )

    pool: list[str] = []
    pool_set: set[str] = set()
    for raw_code in valid_pool:
        code = str(raw_code).strip().upper()
        if code and code not in pool_set:
            pool.append(code)
            pool_set.add(code)
    if not pool:
        raise ValueError(
            "core-satellite canonicalization requires a non-empty stock_pool"
        )

    suffix_by_digits = {
        match.group(1): code
        for code in pool
        if (match := re.fullmatch(r"(\d{6})\.(?:SH|SZ)", code))
    }
    return pool, pool_set, suffix_by_digits


def _pool_code(
    value: Any,
    pool_set: set[str],
    suffix_by_digits: Mapping[str, str],
) -> str:
    code = str(value or "").strip().upper()
    if code in pool_set:
        return code
    if re.fullmatch(r"\d{6}", code):
        return suffix_by_digits.get(code, "")
    return ""


def _answer_code_intent(
    text: str,
    pool_set: set[str],
    suffix_by_digits: Mapping[str, str],
) -> tuple[tuple[str, ...], bool]:
    seen: set[str] = set()
    codes: list[str] = []
    has_outside_pool_code = False
    for match in _STOCK_CODE_INTENT_PATTERN.finditer(_boxed_candidate(text)):
        code = _pool_code(match.group(1), pool_set, suffix_by_digits)
        if not code:
            has_outside_pool_code = True
            continue
        if code and code not in seen:
            seen.add(code)
            codes.append(code)
    return tuple(codes), has_outside_pool_code


def _canonical_weight(value: Any) -> str:
    rendered = f"{float(value):.10f}".rstrip("0").rstrip(".")
    if "." not in rendered:
        return rendered + ".00"
    whole, fraction = rendered.split(".", 1)
    return whole + "." + fraction.ljust(2, "0")


def parse_portfolio_weights(
    text: str,
    valid_pool: Sequence[str],
    *,
    max_stock_weight: float = DEFAULT_MAX_STOCK_WEIGHT,
    tolerance: float = WEIGHT_TOLERANCE,
) -> PortfolioParseResult:
    """Parse and strictly validate a stock/CASH allocation.

    Accepted syntax is ``CODE:0.25,CODE:0.20,CASH:0.55`` inside a final
    ``\\boxed{...}`` or as raw content.  Omitted pool members receive zero
    weight.  Invalid allocations are never renormalized.
    """
    pool = [str(code).upper() for code in valid_pool]
    pool_set = set(pool)
    suffix_by_digits = {code[:6]: code for code in pool}
    zero_weights = {code: 0.0 for code in pool}
    candidate = _boxed_candidate(text)
    if not candidate:
        return PortfolioParseResult(zero_weights, 1.0, False, "empty allocation")
    if "%" in candidate:
        return PortfolioParseResult(
            zero_weights,
            1.0,
            False,
            "weights must use decimal fractions, not percent signs",
        )

    tokens = [
        token.strip()
        for token in re.split(r"[,，;；\n]+", candidate)
        if token.strip()
    ]
    if not tokens:
        return PortfolioParseResult(zero_weights, 1.0, False, "empty allocation")

    parsed: dict[str, float] = {}
    for token in tokens:
        match = _ALLOCATION_TOKEN_PATTERN.fullmatch(token)
        if not match:
            return PortfolioParseResult(
                zero_weights,
                1.0,
                False,
                f"invalid allocation token: {token!r}",
            )
        raw_key, raw_value = match.groups()
        upper_key = raw_key.upper()
        key = (
            "CASH"
            if upper_key in {"CASH", "现金"}
            else suffix_by_digits.get(upper_key, upper_key)
        )
        if key in parsed:
            return PortfolioParseResult(
                zero_weights, 1.0, False, f"duplicate allocation key: {key}"
            )
        try:
            value = float(raw_value)
        except ValueError:
            return PortfolioParseResult(
                zero_weights, 1.0, False, f"invalid weight for {key}"
            )
        if not math.isfinite(value):
            return PortfolioParseResult(
                zero_weights, 1.0, False, f"non-finite weight for {key}"
            )
        parsed[key] = value

    if "CASH" not in parsed:
        return PortfolioParseResult(
            zero_weights, 1.0, False, "CASH weight must be explicit"
        )
    invalid_codes = [
        key for key in parsed if key != "CASH" and key not in pool_set
    ]
    if invalid_codes:
        return PortfolioParseResult(
            zero_weights,
            1.0,
            False,
            f"codes outside pool: {invalid_codes}",
        )

    for key, value in parsed.items():
        upper = 1.0 if key == "CASH" else float(max_stock_weight)
        if value < -tolerance or value > upper + tolerance:
            return PortfolioParseResult(
                zero_weights,
                1.0,
                False,
                f"{key} weight {value} outside [0, {upper}]",
            )

    total = sum(parsed.values())
    if abs(total - 1.0) > tolerance:
        return PortfolioParseResult(
            zero_weights,
            1.0,
            False,
            f"weights including CASH must sum to 1, got {total:.8f}",
        )

    weights = {
        code: max(0.0, float(parsed.get(code, 0.0))) for code in pool
    }
    cash = max(0.0, float(parsed["CASH"]))
    return PortfolioParseResult(weights, cash, True)


def canonicalize_core_satellite_answer(
    text: str,
    metadata: Mapping[str, Any] | None,
) -> CoreSatelliteCanonicalizationResult | None:
    """Canonicalize Agent satellite intent under the fixed core-satellite policy.

    Disabled and legacy policies return ``None``. Active core-satellite policies
    either return one fully assembled allocation or raise ``ValueError`` when
    the hidden snapshot cannot support a valid allocation.
    """
    raw = dict(metadata or {})
    policy = AnchorPolicy.from_mapping(raw.get("anchor_policy"))
    if not policy.enabled or policy.mode != "core_satellite":
        return None

    _, pool_set, suffix_by_digits = _canonical_pool(raw.get("stock_pool", []))
    snapshot = raw.get("anchor_snapshot")
    if not isinstance(snapshot, Mapping) or not snapshot:
        raise ValueError(
            "core-satellite canonicalization requires a non-empty anchor_snapshot"
        )

    try:
        snapshot_allocation = assemble_core_satellite_allocation(snapshot)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "core-satellite snapshot/candidates cannot form an allocation: "
            f"{exc}"
        ) from exc

    core_codes = tuple(
        _pool_code(code, pool_set, suffix_by_digits)
        for code in snapshot_allocation.get("top4", [])
    )
    if len(core_codes) != 4 or len(set(core_codes)) != 4 or not all(core_codes):
        raise ValueError(
            "core-satellite snapshot top4 must contain four unique stock_pool codes"
        )
    core_set = set(core_codes)

    eligible_codes: list[str] = []
    eligible_set: set[str] = set()
    for raw_code in snapshot_allocation.get(
        "eligible_prediction_candidates", []
    ):
        code = _pool_code(raw_code, pool_set, suffix_by_digits)
        if code and code not in core_set and code not in eligible_set:
            eligible_set.add(code)
            eligible_codes.append(code)

    canonical_snapshot = dict(snapshot)
    canonical_snapshot["top4"] = list(core_codes)
    canonical_snapshot["ranked_prediction_candidates"] = eligible_codes
    try:
        deterministic = assemble_core_satellite_allocation(canonical_snapshot)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "core-satellite snapshot lacks enough in-pool eligible candidates: "
            f"{exc}"
        ) from exc

    deterministic_codes = tuple(deterministic.get("satellite_codes", []))
    required_count = int(deterministic.get("satellite_count", 0))
    answer_codes, has_outside_pool_code = _answer_code_intent(
        text,
        pool_set,
        suffix_by_digits,
    )
    satellite_intent = tuple(
        code for code in answer_codes if code in eligible_set and code not in core_set
    )
    ineligible_non_core = tuple(
        code
        for code in answer_codes
        if code not in core_set and code not in eligible_set
    )

    selection_fallback = (
        has_outside_pool_code
        or bool(ineligible_non_core)
        or len(satellite_intent) != required_count
    )
    if selection_fallback:
        assembled = deterministic
        selection_source = "deterministic_fallback"
        reasons = [
            f"found {len(satellite_intent)} eligible satellite code(s); "
            f"required {required_count}"
        ]
        if ineligible_non_core:
            reasons.append(
                "ineligible non-core code(s): " + ",".join(ineligible_non_core)
            )
        if has_outside_pool_code:
            reasons.append("outside-pool stock code(s) present")
        diagnostic = "; ".join(reasons) + "; deterministic fallback was used"
    else:
        try:
            assembled = assemble_core_satellite_allocation(
                canonical_snapshot,
                selected_satellites=satellite_intent,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"eligible Agent satellite selection could not be assembled: {exc}"
            ) from exc
        selection_source = "agent"
        diagnostic = (
            f"accepted {required_count} eligible satellite code(s) from answer"
        )

    weights = {
        str(code): float(weight)
        for code, weight in assembled.get("weights", {}).items()
    }
    cash = float(assembled.get("cash", 0.0))
    selected_codes = tuple(assembled.get("satellite_codes", []))
    allocations = [
        *(f"{code}:{_canonical_weight(weight)}" for code, weight in weights.items()),
        f"CASH:{_canonical_weight(cash)}",
    ]
    canonical_answer = "\\boxed{" + ",".join(allocations) + "}"

    validation = validate_portfolio_answer(canonical_answer, raw)
    if not validation.ok:
        raise ValueError(
            "assembled core-satellite allocation failed validation: "
            f"{validation.error}"
        )

    return CoreSatelliteCanonicalizationResult(
        canonical_boxed_answer=canonical_answer,
        weights=weights,
        cash=cash,
        selected_codes=selected_codes,
        deterministic_codes=deterministic_codes,
        selection_fallback=selection_fallback,
        selection_source=selection_source,
        diagnostic=diagnostic,
    )


def canonicalize_fixed_core_answer(
    text: str,
    metadata: Mapping[str, Any] | None,
) -> FixedCoreCanonicalizationResult | None:
    """Assemble a fixed asset core and equal-weight Agent alpha sleeve.

    The Agent controls only the identities of ``alpha_count`` stocks.  Core,
    alpha, and cash weights are read from task metadata and cannot be changed
    by the model.
    """
    raw = dict(metadata or {})
    fixed_raw = raw.get("fixed_core_weights")
    if not isinstance(fixed_raw, Mapping) or not fixed_raw:
        return None

    stock_pool, stock_set, _ = _canonical_pool(
        raw.get("stock_pool", [])
    )
    fixed_weights: dict[str, float] = {}
    for raw_code, raw_weight in fixed_raw.items():
        code = str(raw_code).strip().upper()
        try:
            weight = float(raw_weight)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"fixed core has invalid weight for {code}") from exc
        if (
            not code
            or code in fixed_weights
            or not math.isfinite(weight)
            or weight <= 0.0
        ):
            raise ValueError("fixed core codes and weights must be unique and positive")
        fixed_weights[code] = weight

    alpha_count = int(raw.get("alpha_count", 0) or 0)
    alpha_weight = float(raw.get("alpha_weight", 0.0) or 0.0)
    cash = float(raw.get("cash_weight", 0.0) or 0.0)
    if alpha_count <= 0 or alpha_weight <= 0.0:
        raise ValueError("fixed-core policy requires positive alpha_count/alpha_weight")
    if cash < 0.0 or not math.isfinite(cash):
        raise ValueError("fixed-core policy requires finite non-negative cash")
    total = sum(fixed_weights.values()) + alpha_count * alpha_weight + cash
    if abs(total - 1.0) > WEIGHT_TOLERANCE:
        raise ValueError(f"fixed-core policy weights must sum to 1, got {total:.8f}")

    asset_pool = [
        *fixed_weights,
        *(code for code in stock_pool if code not in fixed_weights),
    ]
    _, asset_set, asset_suffix = _canonical_pool(asset_pool)
    answer_codes, has_outside_pool_code = _answer_code_intent(
        text,
        asset_set,
        asset_suffix,
    )
    if has_outside_pool_code:
        raise ValueError("fixed-core answer contains code(s) outside the asset pool")
    selected = tuple(code for code in answer_codes if code in stock_set)
    if len(selected) != alpha_count:
        raise ValueError(
            f"fixed-core answer must select exactly {alpha_count} unique "
            f"alpha stocks; found {len(selected)}"
        )
    if any(code in fixed_weights for code in selected):
        raise ValueError("fixed ETF core cannot be selected as stock alpha")

    weights = {
        **fixed_weights,
        **{code: alpha_weight for code in selected},
    }
    allocations = [
        *(f"{code}:{_canonical_weight(weight)}" for code, weight in weights.items()),
        f"CASH:{_canonical_weight(cash)}",
    ]
    canonical_answer = "\\boxed{" + ",".join(allocations) + "}"

    validation_metadata = {
        **raw,
        "asset_pool": asset_pool,
    }
    validation = validate_portfolio_answer(canonical_answer, validation_metadata)
    if not validation.ok:
        raise ValueError(
            "assembled fixed-core allocation failed validation: "
            f"{validation.error}"
        )
    return FixedCoreCanonicalizationResult(
        canonical_boxed_answer=canonical_answer,
        weights=weights,
        cash=cash,
        selected_codes=selected,
        diagnostic=(
            f"accepted {len(selected)} Agent alpha stocks and enforced "
            f"{len(fixed_weights)} fixed core assets"
        ),
    )


def validate_portfolio_answer(
    text: str,
    metadata: Mapping[str, Any] | None,
) -> PortfolioValidationResult:
    """Apply syntax, pool, weight, and optional momentum-anchor constraints."""
    raw = dict(metadata or {})
    valid_pool = raw.get("asset_pool") or raw.get("stock_pool", [])
    parsed = parse_portfolio_weights(
        text,
        valid_pool,
        max_stock_weight=float(raw.get("max_stock_weight", 0.25)),
    )
    if not parsed.ok:
        return PortfolioValidationResult(parsed)
    active_weights = [
        value for value in parsed.weights.values() if value > WEIGHT_TOLERANCE
    ]
    holding_count = len(active_weights)
    min_holdings = int(raw.get("min_holdings", 0) or 0)
    max_holdings = int(raw.get("max_holdings", 0) or 0)
    min_active_weight = float(raw.get("min_active_stock_weight", 0.0))
    min_cash = float(raw.get("min_cash_weight", 0.0))
    max_cash = float(raw.get("max_cash_weight", 1.0))
    constraint_error = ""
    if min_holdings and holding_count < min_holdings:
        constraint_error = (
            f"portfolio must hold at least {min_holdings} stocks; held {holding_count}"
        )
    elif max_holdings and holding_count > max_holdings:
        constraint_error = (
            f"portfolio must hold at most {max_holdings} stocks; held {holding_count}"
        )
    elif min_active_weight and any(
        value < min_active_weight - WEIGHT_TOLERANCE for value in active_weights
    ):
        smallest = min(active_weights)
        constraint_error = (
            f"active stock weight {smallest:.8f} is below minimum "
            f"{min_active_weight:.8f}"
        )
    elif parsed.cash < min_cash - WEIGHT_TOLERANCE:
        constraint_error = (
            f"CASH weight {parsed.cash:.8f} is below minimum {min_cash:.8f}"
        )
    elif parsed.cash > max_cash + WEIGHT_TOLERANCE:
        constraint_error = (
            f"CASH weight {parsed.cash:.8f} exceeds maximum {max_cash:.8f}"
        )
    if constraint_error:
        invalid = PortfolioParseResult(
            parsed.weights,
            parsed.cash,
            False,
            constraint_error,
        )
        return PortfolioValidationResult(invalid)

    fixed_raw = raw.get("fixed_core_weights")
    if isinstance(fixed_raw, Mapping) and fixed_raw:
        fixed_weights = {
            str(code).strip().upper(): float(weight)
            for code, weight in fixed_raw.items()
        }
        stock_set = {
            str(code).strip().upper() for code in raw.get("stock_pool", [])
        }
        alpha_count = int(raw.get("alpha_count", 0) or 0)
        alpha_weight = float(raw.get("alpha_weight", 0.0) or 0.0)
        expected_cash = float(raw.get("cash_weight", 0.0) or 0.0)
        fixed_error = ""
        for code, expected in fixed_weights.items():
            actual = float(parsed.weights.get(code, 0.0))
            if abs(actual - expected) > WEIGHT_TOLERANCE:
                fixed_error = (
                    f"fixed core {code} weight {actual:.8f} must equal "
                    f"{expected:.8f}"
                )
                break
        alpha_codes = [
            code
            for code, weight in parsed.weights.items()
            if code not in fixed_weights and weight > WEIGHT_TOLERANCE
        ]
        if not fixed_error and any(code not in stock_set for code in alpha_codes):
            fixed_error = "fixed-core alpha contains a non-stock asset"
        elif not fixed_error and len(alpha_codes) != alpha_count:
            fixed_error = (
                f"fixed-core portfolio must hold exactly {alpha_count} alpha "
                f"stocks; held {len(alpha_codes)}"
            )
        elif not fixed_error and any(
            abs(float(parsed.weights[code]) - alpha_weight) > WEIGHT_TOLERANCE
            for code in alpha_codes
        ):
            fixed_error = (
                f"every fixed-core alpha stock must have weight "
                f"{alpha_weight:.8f}"
            )
        elif (
            not fixed_error
            and abs(parsed.cash - expected_cash) > WEIGHT_TOLERANCE
        ):
            fixed_error = (
                f"fixed-core CASH weight {parsed.cash:.8f} must equal "
                f"{expected_cash:.8f}"
            )
        eligible_raw = raw.get("growth_quality_eligible_codes")
        if not fixed_error and isinstance(eligible_raw, list):
            eligible_set = {
                str(code).strip().upper()
                for code in eligible_raw
                if str(code).strip()
            }
            ineligible = [code for code in alpha_codes if code not in eligible_set]
            if ineligible:
                fixed_error = (
                    "alpha stocks failed the growth-quality hard filter: "
                    + ",".join(ineligible)
                )
        if fixed_error:
            invalid = PortfolioParseResult(
                parsed.weights,
                parsed.cash,
                False,
                fixed_error,
            )
            return PortfolioValidationResult(invalid)
    anchor = validate_anchor_from_metadata(parsed.weights, parsed.cash, raw)
    return PortfolioValidationResult(parsed, anchor)


def cash_allocation(valid_pool: Sequence[str]) -> PortfolioParseResult:
    """Return the deterministic fallback for an invalid or missing decision."""
    return PortfolioParseResult(
        {str(code).upper(): 0.0 for code in valid_pool},
        1.0,
        True,
    )


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        finish = start + 1
        while finish < len(order) and values[order[finish]] == values[order[start]]:
            finish += 1
        average = (start + 1 + finish) / 2.0
        for position in range(start, finish):
            ranks[order[position]] = average
        start = finish
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return float("nan")
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum(
        (x - left_mean) * (y - right_mean) for x, y in zip(left, right)
    )
    left_var = sum((x - left_mean) ** 2 for x in left)
    right_var = sum((y - right_mean) ** 2 for y in right)
    denominator = math.sqrt(left_var * right_var)
    return numerator / denominator if denominator > 0 else float("nan")


def weight_rank_ic(
    weights: Mapping[str, float],
    excess_returns: Mapping[str, float],
) -> float:
    """Spearman correlation between allocation weights and realized excess."""
    codes = sorted(set(weights) & set(excess_returns))
    if len(codes) < 2:
        return float("nan")
    return _pearson(
        _average_ranks([float(weights[code]) for code in codes]),
        _average_ranks([float(excess_returns[code]) for code in codes]),
    )


def evaluate_portfolio_month(
    weights: Mapping[str, float],
    cash: float,
    stock_returns: Mapping[str, float],
    index_return: float,
    *,
    starting_capital: float,
    excess_returns: Mapping[str, float] | None = None,
    open_cost: float = DEFAULT_OPEN_COST,
    close_cost: float = DEFAULT_CLOSE_COST,
    min_cost: float = DEFAULT_MIN_COST,
) -> PortfolioMonthResult:
    """Buy at the entry close, liquidate after 20 sessions, and apply fees."""
    if starting_capital <= 0:
        raise ValueError("starting_capital must be positive")
    if open_cost < 0 or close_cost < 0 or min_cost < 0:
        raise ValueError("transaction costs cannot be negative")
    invested_weight = sum(float(value) for value in weights.values())
    if cash < -WEIGHT_TOLERANCE or abs(invested_weight + cash - 1.0) > 1e-5:
        raise ValueError("weights and cash must form a fully funded portfolio")

    ending_capital = starting_capital * max(0.0, float(cash))
    gross_ending_capital = ending_capital
    buy_cost_total = 0.0
    sell_cost_total = 0.0
    traded_notional = 0.0
    contributions: dict[str, float] = {}

    for code, raw_weight in weights.items():
        weight = float(raw_weight)
        if weight <= WEIGHT_TOLERANCE:
            continue
        if code not in stock_returns:
            raise ValueError(f"missing realized return for {code}")
        period_return = float(stock_returns[code])
        if not math.isfinite(period_return) or period_return < -1.0:
            raise ValueError(f"invalid realized return for {code}: {period_return}")

        allocated = starting_capital * weight
        buy_cost = min(allocated, max(allocated * open_cost, min_cost))
        invested_after_buy = max(0.0, allocated - buy_cost)
        value_before_sale = invested_after_buy * (1.0 + period_return)
        sell_cost = (
            min(value_before_sale, max(value_before_sale * close_cost, min_cost))
            if value_before_sale > 0
            else 0.0
        )
        net_value = max(0.0, value_before_sale - sell_cost)

        ending_capital += net_value
        gross_ending_capital += allocated * (1.0 + period_return)
        buy_cost_total += buy_cost
        sell_cost_total += sell_cost
        traded_notional += allocated + value_before_sale
        contributions[code] = net_value / starting_capital - weight

    net_return = ending_capital / starting_capital - 1.0
    gross_return = gross_ending_capital / starting_capital - 1.0
    excess = (
        excess_returns
        if excess_returns is not None
        else {
            code: float(stock_returns[code]) - float(index_return)
            for code in weights
            if code in stock_returns
        }
    )
    return PortfolioMonthResult(
        starting_capital=float(starting_capital),
        ending_capital=float(ending_capital),
        gross_return=float(gross_return),
        net_return=float(net_return),
        index_return=float(index_return),
        active_return=float(net_return - index_return),
        buy_cost=float(buy_cost_total),
        sell_cost=float(sell_cost_total),
        total_cost=float(buy_cost_total + sell_cost_total),
        gross_traded_notional=float(traded_notional),
        cash_weight=float(cash),
        invested_weight=float(invested_weight),
        holding_count=sum(
            float(weight) > WEIGHT_TOLERANCE for weight in weights.values()
        ),
        concentration_hhi=sum(float(weight) ** 2 for weight in weights.values()),
        weight_rank_ic=weight_rank_ic(weights, excess),
        contributions=contributions,
    )


def evaluate_anchor_deviation(
    actual_weights: Mapping[str, float],
    actual_cash: float,
    anchor_weights: Mapping[str, float],
    anchor_cash: float,
    stock_returns: Mapping[str, float],
    index_return: float,
    *,
    starting_capital: float,
    excess_returns: Mapping[str, float] | None = None,
    open_cost: float = DEFAULT_OPEN_COST,
    close_cost: float = DEFAULT_CLOSE_COST,
    min_cost: float = DEFAULT_MIN_COST,
    actual_result: PortfolioMonthResult | None = None,
) -> dict[str, Any]:
    """Attribute matured performance to deviation from a fixed anchor."""
    actual = actual_result or evaluate_portfolio_month(
        actual_weights,
        actual_cash,
        stock_returns,
        index_return,
        starting_capital=starting_capital,
        excess_returns=excess_returns,
        open_cost=open_cost,
        close_cost=close_cost,
        min_cost=min_cost,
    )
    anchor = evaluate_portfolio_month(
        anchor_weights,
        anchor_cash,
        stock_returns,
        index_return,
        starting_capital=starting_capital,
        excess_returns=excess_returns,
        open_cost=open_cost,
        close_cost=close_cost,
        min_cost=min_cost,
    )
    actual_codes = sorted(
        code for code, weight in actual_weights.items() if float(weight) > WEIGHT_TOLERANCE
    )
    anchor_codes = sorted(
        code for code, weight in anchor_weights.items() if float(weight) > WEIGHT_TOLERANCE
    )
    all_codes = sorted(set(actual_weights) | set(anchor_weights))
    weight_delta_contributions = {
        code: (
            float(actual_weights.get(code, 0.0))
            - float(anchor_weights.get(code, 0.0))
        )
        * float(stock_returns.get(code, 0.0))
        for code in all_codes
        if abs(
            float(actual_weights.get(code, 0.0))
            - float(anchor_weights.get(code, 0.0))
        )
        > WEIGHT_TOLERANCE
    }
    return {
        "anchor_weights": {
            str(code): float(weight) for code, weight in anchor_weights.items()
        },
        "anchor_cash": float(anchor_cash),
        "anchor_net_return": float(anchor.net_return),
        "anchor_active_return": float(anchor.active_return),
        "anchor_total_cost": float(anchor.total_cost),
        "actual_net_return": float(actual.net_return),
        "actual_active_return": float(actual.active_return),
        "actual_total_cost": float(actual.total_cost),
        "deviation_net_return": float(actual.net_return - anchor.net_return),
        "deviation_active_return": float(
            actual.active_return - anchor.active_return
        ),
        "cost_delta": float(actual.total_cost - anchor.total_cost),
        "cash_delta": float(actual_cash - anchor_cash),
        "overlap_count": len(set(actual_codes) & set(anchor_codes)),
        "dropped_from_anchor": sorted(set(anchor_codes) - set(actual_codes)),
        "added_vs_anchor": sorted(set(actual_codes) - set(anchor_codes)),
        "weight_delta_contributions": weight_delta_contributions,
    }
