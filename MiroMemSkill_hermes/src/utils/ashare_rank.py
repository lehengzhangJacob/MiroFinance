"""Deterministic parsing and metrics for A-share cross-sectional rankings."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class RankParseResult:
    codes: list[str]
    ok: bool
    error: str = ""


def parse_ranked_codes(text: str, valid_pool: Sequence[str]) -> RankParseResult:
    """Parse a complete ordered stock-code list from boxed or raw output."""
    pool = [str(code).upper() for code in valid_pool]
    pool_set = set(pool)
    suffix_by_digits = {code[:6]: code for code in pool}
    raw = str(text or "")
    boxed = re.findall(r"\\boxed\{([^{}]*)\}", raw, flags=re.DOTALL)
    candidate = boxed[-1] if boxed else raw
    tokens = re.findall(r"(?<!\d)(\d{6}(?:\.(?:SH|SZ))?)(?!\d)", candidate.upper())
    normalized = [suffix_by_digits.get(token, token) for token in tokens]

    if len(normalized) != len(pool):
        return RankParseResult(
            normalized,
            False,
            f"expected {len(pool)} codes, found {len(normalized)}",
        )
    if len(set(normalized)) != len(normalized):
        return RankParseResult(normalized, False, "duplicate stock code")
    invalid = [code for code in normalized if code not in pool_set]
    if invalid:
        return RankParseResult(normalized, False, f"codes outside pool: {invalid}")
    return RankParseResult(normalized, True)


def spearman_rank_ic(predicted: Sequence[str], ground_truth: Sequence[str]) -> float:
    """Spearman correlation for two complete permutations (no tie ambiguity)."""
    if len(predicted) != len(ground_truth) or len(predicted) < 2:
        return float("nan")
    truth_rank = {code: index for index, code in enumerate(ground_truth, 1)}
    if len(truth_rank) != len(ground_truth) or set(predicted) != set(ground_truth):
        return float("nan")
    squared = sum(
        (index - truth_rank[code]) ** 2 for index, code in enumerate(predicted, 1)
    )
    n = len(predicted)
    return 1.0 - 6.0 * squared / (n * (n * n - 1))


def evaluate_ranking(
    predicted: Sequence[str],
    ground_truth: Sequence[str],
    excess_returns: Mapping[str, float],
    *,
    top_k: int = 4,
) -> dict[str, float]:
    """Compute rank and portfolio metrics for one monthly cross-section."""
    rank_ic = spearman_rank_ic(predicted, ground_truth)
    if math.isnan(rank_ic):
        raise ValueError("predicted and ground-truth rankings must be complete permutations")
    if top_k <= 0 or 2 * top_k > len(predicted):
        raise ValueError("top_k must define non-overlapping top and bottom legs")

    def mean_return(codes: Sequence[str]) -> float:
        return sum(float(excess_returns[code]) for code in codes) / len(codes)

    top = mean_return(predicted[:top_k])
    bottom = mean_return(predicted[-top_k:])
    return {
        "rank_ic": rank_ic,
        "top_excess": top,
        "bottom_excess": bottom,
        "spread": top - bottom,
    }
