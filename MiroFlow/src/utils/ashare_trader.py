"""Parsing and deterministic portfolio math for ``ashare-trader``."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Mapping, Sequence


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


def _boxed_candidate(text: str) -> str:
    raw = str(text or "").strip()
    boxed = re.findall(r"\\boxed\{([^{}]*)\}", raw, flags=re.DOTALL)
    return boxed[-1].strip() if boxed else raw


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
        match = re.fullmatch(
            r"\s*(CASH|现金|\d{6}(?:\.(?:SH|SZ))?)\s*[:=：]\s*"
            r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*",
            token,
            flags=re.IGNORECASE,
        )
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
