"""Strict point-in-time full-market breadth features for A-share trading."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd


BREADTH_FILE = "market_breadth_daily.csv"
INDEX_FILE = "index_000300.SH.csv"


def normalize_ashare_date(value: str) -> str:
    """Return a YYYYMMDD cutoff accepted by all A-share utilities."""
    normalized = str(value or "").strip().replace("-", "").replace("/", "")
    if len(normalized) != 8 or not normalized.isdigit():
        raise ValueError(
            f"invalid A-share date {value!r}; expected YYYYMMDD or YYYY-MM-DD"
        )
    return normalized


@lru_cache(maxsize=8)
def _load_csv(path_text: str) -> pd.DataFrame:
    path = Path(path_text)
    if not path.exists():
        raise FileNotFoundError(f"market breadth data file not found: {path}")
    return pd.read_csv(path, dtype={"trade_date": str})


def clear_market_breadth_cache() -> None:
    """Clear process-local CSV caches (primarily for deterministic tests)."""
    _load_csv.cache_clear()


def _latest_row(frame: pd.DataFrame, cutoff: str, *, source: str) -> pd.Series:
    eligible = frame[frame["trade_date"].astype(str) <= cutoff].sort_values(
        "trade_date"
    )
    if eligible.empty:
        raise ValueError(f"no {source} rows on or before {cutoff}")
    return eligible.iloc[-1]


def _finite_number(row: pd.Series, key: str) -> float:
    value = pd.to_numeric(pd.Series([row.get(key)]), errors="coerce").iloc[0]
    if pd.isna(value):
        raise ValueError(f"market breadth row has no finite {key}")
    return float(value)


def compute_market_breadth_regime(
    as_of: str,
    *,
    data_dir: str | Path,
) -> dict[str, Any]:
    """Compute a transparent risk-on/neutral/defensive regime as of a cutoff.

    Every input row is filtered to ``trade_date <= as_of``.  The thresholds are
    fixed ex ante; no future cross-section, realized holding-period return, or
    fitted parameter is used.
    """
    cutoff = normalize_ashare_date(as_of)
    root = Path(data_dir)
    breadth = _load_csv(str((root / BREADTH_FILE).resolve()))
    breadth_row = _latest_row(breadth, cutoff, source="market breadth")

    index = _load_csv(str((root / INDEX_FILE).resolve()))
    index = index[index["trade_date"].astype(str) <= cutoff].sort_values("trade_date")
    if index.empty:
        raise ValueError(f"no CSI300 rows on or before {cutoff}")
    closes = pd.to_numeric(index["close"], errors="coerce").dropna()
    if closes.empty:
        raise ValueError(f"no finite CSI300 close on or before {cutoff}")
    last_close = float(closes.iloc[-1])
    index_ma20 = (
        float(closes.tail(20).mean()) if len(closes) >= 20 else float("nan")
    )
    index_ret20 = (
        last_close / float(closes.iloc[-21]) - 1.0
        if len(closes) >= 21 and float(closes.iloc[-21]) != 0.0
        else float("nan")
    )

    metrics = {
        "adv_ratio_1d": _finite_number(breadth_row, "adv_ratio_1d"),
        "adv_ratio_5d": _finite_number(breadth_row, "adv_ratio_5d"),
        "above_ma20_ratio": _finite_number(breadth_row, "above_ma20_ratio"),
        "above_ma60_ratio": _finite_number(breadth_row, "above_ma60_ratio"),
        "positive_ret20_ratio": _finite_number(
            breadth_row, "positive_ret20_ratio"
        ),
        "index_ret20": index_ret20,
        "index_above_ma20": bool(
            pd.notna(index_ma20) and last_close >= index_ma20
        ),
    }

    positive_votes = {
        "five_day_advancers": metrics["adv_ratio_5d"] >= 0.55,
        "above_ma20": metrics["above_ma20_ratio"] >= 0.55,
        "above_ma60": metrics["above_ma60_ratio"] >= 0.50,
        "positive_ret20": metrics["positive_ret20_ratio"] >= 0.55,
        "csi300_trend": bool(
            pd.notna(index_ret20)
            and index_ret20 > 0.0
            and metrics["index_above_ma20"]
        ),
    }
    defensive_votes = {
        "five_day_decliners": metrics["adv_ratio_5d"] <= 0.45,
        "below_ma20": metrics["above_ma20_ratio"] <= 0.45,
        "below_ma60": metrics["above_ma60_ratio"] <= 0.50,
        "negative_ret20": metrics["positive_ret20_ratio"] <= 0.45,
        "csi300_downtrend": bool(
            pd.notna(index_ret20)
            and index_ret20 < 0.0
            and not metrics["index_above_ma20"]
        ),
    }
    risk_on_score = sum(positive_votes.values())
    defensive_score = sum(defensive_votes.values())
    if risk_on_score >= 3 and defensive_score < 3:
        regime = "risk_on"
    elif defensive_score >= 3 and risk_on_score < 3:
        regime = "defensive"
    else:
        regime = "neutral"

    return {
        "as_of": cutoff,
        "data_date": str(breadth_row["trade_date"]),
        "regime": regime,
        "risk_on_score": int(risk_on_score),
        "defensive_score": int(defensive_score),
        "universe_count": int(_finite_number(breadth_row, "universe_count")),
        "metrics": metrics,
        "positive_votes": positive_votes,
        "defensive_votes": defensive_votes,
    }


def render_market_breadth_regime(
    as_of: str,
    *,
    data_dir: str | Path,
) -> str:
    """Render the strict point-in-time breadth regime for an MCP response."""
    result = compute_market_breadth_regime(as_of, data_dir=data_dir)
    metrics = result["metrics"]
    return "\n".join(
        [
            f"# 全A市场广度（严格点时），决策日 {result['as_of']}",
            (
                f"数据日={result['data_date']}，有效股票数={result['universe_count']}，"
                f"状态={result['regime']}，risk_on票数={result['risk_on_score']}/5，"
                f"defensive票数={result['defensive_score']}/5"
            ),
            (
                "adv_ratio_1d={:.2%},adv_ratio_5d={:.2%},"
                "above_ma20={:.2%},above_ma60={:.2%},"
                "positive_ret20={:.2%},csi300_ret20={:+.2%},"
                "csi300_above_ma20={}"
            ).format(
                metrics["adv_ratio_1d"],
                metrics["adv_ratio_5d"],
                metrics["above_ma20_ratio"],
                metrics["above_ma60_ratio"],
                metrics["positive_ret20_ratio"],
                metrics["index_ret20"],
                str(metrics["index_above_ma20"]).lower(),
            ),
            (
                "判定规则预先固定；所有截面和指数数据均满足 trade_date<=as_of。"
                "市场广度只用于趋势/防守语境，不是未来收益标签，也不能单独否决个股。"
            ),
        ]
    )
