"""Point-in-time relative-momentum baseline for the unified A-share trader."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.utils.ashare_trader_features import (
    RETURN_WINDOWS,
    compute_trader_feature_rows,
    normalize_date,
)


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def build_relative_momentum_baseline(
    as_of: str,
    stock_pool: Mapping[str, Mapping[str, Any]],
    *,
    data_dir: str | Path,
    window: int = 20,
    top_k: int = 4,
    max_stock_weight: float = 0.25,
) -> dict[str, Any]:
    """Rank the pool by trailing relative return and build a capped top-k anchor.

    All features are computed after hard-cutting source rows at ``as_of``.  The
    ranking therefore uses only information available on the decision date.
    """
    if window not in RETURN_WINDOWS:
        raise ValueError(f"window must be one of {RETURN_WINDOWS}, got {window}")
    if not 1 <= int(top_k) <= 4:
        raise ValueError(f"top_k must be between 1 and 4, got {top_k}")
    if not 0.0 < float(max_stock_weight) <= 0.25:
        raise ValueError(
            f"max_stock_weight must be in (0, 0.25], got {max_stock_weight}"
        )

    entry_date = normalize_date(as_of)
    rows = compute_trader_feature_rows(
        entry_date,
        stock_pool,
        data_dir=data_dir,
        lookback_days=0,
    )
    relative_key = f"rel{window}"
    return_key = f"ret{window}"
    eligible = [
        (row, value)
        for row in rows
        if (value := _finite_number(row.get(relative_key))) is not None
    ]
    eligible.sort(key=lambda item: (-item[1], str(item[0]["ts_code"])))

    ranking: list[dict[str, Any]] = []
    for rank, (row, relative_return) in enumerate(eligible, start=1):
        ranking.append(
            {
                "rank": rank,
                "ts_code": str(row["ts_code"]),
                "name": str(row.get("name", "")),
                "industry": str(row.get("industry", "")),
                return_key: row.get(return_key, ""),
                relative_key: relative_return,
                "rel5": row.get("rel5", ""),
                "rel60": row.get("rel60", ""),
                "rel120": row.get("rel120", ""),
                "vol20_ann": row.get("vol20_ann", ""),
                "max_dd120": row.get("max_dd120", ""),
                "from_high250": row.get("from_high250", ""),
                "ma20_gap": row.get("ma20_gap", ""),
                "amount20_vs120": row.get("amount20_vs120", ""),
                "pe_pct250": row.get("pe_pct250", ""),
                "pb_pct250": row.get("pb_pct250", ""),
                "ml_rank": row.get("ml_rank", ""),
                "financial_ann_date": row.get("financial_ann_date", ""),
                "or_yoy": row.get("or_yoy", ""),
                "netprofit_yoy": row.get("netprofit_yoy", ""),
            }
        )

    selected = ranking[: int(top_k)]
    weight = (
        min(float(max_stock_weight), 1.0 / len(selected))
        if selected
        else 0.0
    )
    weights = {
        str(row["ts_code"]): round(weight, 10)
        for row in selected
    }
    cash = round(max(0.0, 1.0 - sum(weights.values())), 10)
    return {
        "as_of": entry_date,
        "window": int(window),
        "top_k": int(top_k),
        "max_stock_weight": float(max_stock_weight),
        "weights": weights,
        "cash": cash,
        "ranking": ranking,
    }


def render_relative_momentum_baseline(
    as_of: str,
    stock_pool: Mapping[str, Mapping[str, Any]],
    *,
    data_dir: str | Path,
    window: int = 20,
    top_k: int = 4,
) -> str:
    """Render the point-in-time momentum anchor for an agent tool response."""
    result = build_relative_momentum_baseline(
        as_of,
        stock_pool,
        data_dir=data_dir,
        window=window,
        top_k=top_k,
    )
    allocations = [
        *(f"{code}:{weight:.2f}" for code, weight in result["weights"].items()),
        f"CASH:{result['cash']:.2f}",
    ]
    frame = pd.DataFrame(result["ranking"])
    return "\n".join(
        [
            f"# A股 {window} 日相对动量 top{top_k} 硬锚基线（严格点时）",
            f"决策日: {result['as_of']}；排序指标: 个股近{window}日复权收益"
            f"减沪深300同期收益；仅使用决策日及以前数据。",
            "参考组合: " + ",".join(allocations),
            "该组合是可交易锚点而非未来收益标签；硬锚实验中，top4 降配需至少"
            "两类独立当前风险，且 Qlib 不得单独否决。",
            "## 全池排名与诊断 (CSV)",
            frame.to_csv(index=False),
        ]
    )
