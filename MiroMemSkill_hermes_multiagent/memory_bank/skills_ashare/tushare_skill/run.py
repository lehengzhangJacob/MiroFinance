#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

"""tushare_skill CLI — point-in-time safe Tushare Pro data access.

Subcommands (see SKILL.md / README.md):
    daily | index | valuation | financials | stock-info | trade-cal

Every subcommand prints a JSON envelope to stdout, or writes CSV with --out.
`--as-of` enforces the point-in-time discipline used across this repo's
A-share backtests: market data cut on trade_date, financials cut on ann_date.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests
import yaml

SKILL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SKILL_DIR))

from schema import (  # noqa: E402
    VALID_ADJUST,
    VALID_FORMAT,
    envelope,
    normalize_date,
    validate_choice,
    validate_ts_code,
)

# Tushare error messages that indicate a transient rate limit worth retrying.
_RATE_LIMIT_MARKERS = ("每分钟", "每小时", "频率", "too many", "rate limit")


def load_config() -> dict[str, Any]:
    with open(SKILL_DIR / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Token resolution: env var -> config token.file -> walk-up file search
# ---------------------------------------------------------------------------


def _parse_token_text(raw: str) -> str:
    raw = raw.strip()
    # Accept `TUSHARE_TOKEN=xxx` / `KEY = "xxx"` or a bare token.
    if "=" in raw:
        raw = raw.split("=", 1)[1].strip().strip('"').strip("'")
    return raw


def resolve_token(cfg: dict[str, Any]) -> str:
    token_cfg = cfg.get("token", {})

    env_var = token_cfg.get("env_var", "TUSHARE_TOKEN")
    if os.getenv(env_var, "").strip():
        return os.getenv(env_var).strip()

    explicit = (token_cfg.get("file") or "").strip()
    if explicit:
        p = Path(explicit).expanduser()
        if p.exists():
            return _parse_token_text(p.read_text(encoding="utf-8"))

    walk_name = token_cfg.get("token_walk_filename", "tushare_token")
    cur = SKILL_DIR
    for _ in range(8):
        candidate = cur / walk_name
        if candidate.exists():
            return _parse_token_text(candidate.read_text(encoding="utf-8"))
        if cur.parent == cur:
            break
        cur = cur.parent

    sys.exit(
        f"tushare token not found: set ${env_var}, or config.yaml token.file, "
        f"or place a '{walk_name}' file in an ancestor directory."
    )


# ---------------------------------------------------------------------------
# HTTP query with retry/backoff
# ---------------------------------------------------------------------------


class _RateLimited(Exception):
    pass


def tushare_query(
    cfg: dict[str, Any],
    token: str,
    api_name: str,
    params: dict[str, Any],
    fields: str,
) -> pd.DataFrame:
    req_cfg = cfg.get("request", {})
    timeout = int(req_cfg.get("timeout_seconds", 60))
    max_retries = int(req_cfg.get("max_retries", 3))
    backoff = float(req_cfg.get("backoff_seconds", 5))

    payload = {
        "api_name": api_name,
        "token": token,
        "params": {k: v for k, v in params.items() if v not in (None, "")},
        "fields": fields,
    }

    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(cfg["api_url"], json=payload, timeout=timeout)
            resp.raise_for_status()
            body = resp.json()
            if body.get("code") != 0:
                msg = str(body.get("msg", ""))
                if any(m in msg for m in _RATE_LIMIT_MARKERS):
                    raise _RateLimited(msg)
                raise RuntimeError(f"tushare {api_name} error: {msg}")
            data = body["data"]
            return pd.DataFrame(data["items"], columns=data["fields"])
        except (_RateLimited, requests.exceptions.RequestException) as exc:
            last_err = exc
            if attempt < max_retries:
                time.sleep(backoff * (2**attempt))
                continue
            break
    raise RuntimeError(f"tushare {api_name} failed after {max_retries + 1} attempts: {last_err}")


# ---------------------------------------------------------------------------
# Point-in-time helpers
# ---------------------------------------------------------------------------


def effective_end(end: Optional[str], as_of: Optional[str]) -> Optional[str]:
    """Server-side cutoff: never request rows past as_of."""
    if end and as_of:
        return min(end, as_of)
    return as_of or end


def cut_by_column(df: pd.DataFrame, column: str, as_of: Optional[str]) -> pd.DataFrame:
    """Client-side cutoff (belt and braces on top of effective_end)."""
    if as_of is None or column not in df.columns or df.empty:
        return df
    kept = df[df[column].astype(str) <= as_of]
    return kept.reset_index(drop=True)


def apply_qfq(daily: pd.DataFrame, adj: pd.DataFrame) -> pd.DataFrame:
    """Forward-adjust OHLC: price * adj_factor / factor_at_window_end.

    Same convention as the repo's backtest cache builder — the window end
    (i.e. as_of) is the adjustment basis, so returns inside the window are
    correct. Results from different as_of queries must not be concatenated.
    """
    daily = daily.sort_values("trade_date").reset_index(drop=True)
    adj = adj.sort_values("trade_date").reset_index(drop=True)
    merged = daily.merge(adj, on="trade_date", how="left")
    merged["adj_factor"] = merged["adj_factor"].ffill()
    if merged["adj_factor"].isna().all():
        raise RuntimeError("adj_factor unavailable for the requested window")
    latest = merged["adj_factor"].iloc[-1]
    for col in ("open", "high", "low", "close"):
        merged[f"{col}_qfq"] = merged[col] * merged["adj_factor"] / latest
    return merged


# ---------------------------------------------------------------------------
# Subcommand implementations
# ---------------------------------------------------------------------------


def cmd_daily(args: argparse.Namespace, cfg: dict, token: str) -> tuple[pd.DataFrame, dict]:
    ts_code = validate_ts_code(args.ts_code)
    params = {
        "ts_code": ts_code,
        "start_date": args.start,
        "end_date": effective_end(args.end, args.as_of),
    }
    df = tushare_query(cfg, token, "daily", params, args.fields or cfg["fields"]["daily"])
    df = df.sort_values("trade_date").reset_index(drop=True)
    df = cut_by_column(df, "trade_date", args.as_of)

    if args.adjust == "qfq" and not df.empty:
        adj = tushare_query(cfg, token, "adj_factor", params, cfg["fields"]["adj_factor"])
        adj = cut_by_column(adj.sort_values("trade_date"), "trade_date", args.as_of)
        df = apply_qfq(df, adj)
    return df, params


def cmd_index(args: argparse.Namespace, cfg: dict, token: str) -> tuple[pd.DataFrame, dict]:
    params = {
        "ts_code": validate_ts_code(args.ts_code),
        "start_date": args.start,
        "end_date": effective_end(args.end, args.as_of),
    }
    df = tushare_query(cfg, token, "index_daily", params, args.fields or cfg["fields"]["index_daily"])
    df = df.sort_values("trade_date").reset_index(drop=True)
    return cut_by_column(df, "trade_date", args.as_of), params


def cmd_valuation(args: argparse.Namespace, cfg: dict, token: str) -> tuple[pd.DataFrame, dict]:
    params = {
        "ts_code": validate_ts_code(args.ts_code),
        "start_date": args.start,
        "end_date": effective_end(args.end, args.as_of),
    }
    df = tushare_query(cfg, token, "daily_basic", params, args.fields or cfg["fields"]["daily_basic"])
    df = df.sort_values("trade_date").reset_index(drop=True)
    return cut_by_column(df, "trade_date", args.as_of), params


def cmd_financials(args: argparse.Namespace, cfg: dict, token: str) -> tuple[pd.DataFrame, dict]:
    # NOTE: fina_indicator's start/end params filter by REPORT PERIOD
    # (end_date), not announcement date. Point-in-time correctness therefore
    # comes from the client-side ann_date <= as_of cut below.
    params = {
        "ts_code": validate_ts_code(args.ts_code),
        "start_date": args.start,
        "end_date": args.end,
    }
    df = tushare_query(
        cfg, token, "fina_indicator", params, args.fields or cfg["fields"]["fina_indicator"]
    )
    if "ann_date" in df.columns:
        df = df[df["ann_date"].notna() & (df["ann_date"].astype(str) != "")]
        df = df.sort_values("ann_date").reset_index(drop=True)
        df = cut_by_column(df, "ann_date", args.as_of)
    return df, params


def cmd_stock_info(args: argparse.Namespace, cfg: dict, token: str) -> tuple[pd.DataFrame, dict]:
    params: dict[str, Any] = {}
    if args.ts_code:
        params["ts_code"] = validate_ts_code(args.ts_code)
    df = tushare_query(cfg, token, "stock_basic", params, args.fields or cfg["fields"]["stock_basic"])
    return df.reset_index(drop=True), params


def cmd_trade_cal(args: argparse.Namespace, cfg: dict, token: str) -> tuple[pd.DataFrame, dict]:
    params = {
        "exchange": args.exchange,
        "start_date": args.start,
        "end_date": effective_end(args.end, args.as_of),
    }
    df = tushare_query(cfg, token, "trade_cal", params, args.fields or cfg["fields"]["trade_cal"])
    df = df.sort_values("cal_date").reset_index(drop=True)
    return cut_by_column(df, "cal_date", args.as_of), params


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def emit(df: pd.DataFrame, api: str, params: dict, args: argparse.Namespace) -> None:
    records = json.loads(df.to_json(orient="records", force_ascii=False))

    if args.out:
        out_path = Path(args.out).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        body = envelope(api, params, records, as_of=args.as_of, out=str(out_path))
    elif args.format == "csv":
        df.to_csv(sys.stdout, index=False)
        return
    else:
        body = envelope(api, params, records, as_of=args.as_of)

    print(json.dumps(body, ensure_ascii=False, indent=None if len(records) > 50 else 2))


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def _add_common(p: argparse.ArgumentParser, cfg: dict, *, dates: bool = True) -> None:
    if dates:
        p.add_argument("--start", help="start date YYYYMMDD / YYYY-MM-DD")
        p.add_argument("--end", help="end date YYYYMMDD / YYYY-MM-DD")
        p.add_argument(
            "--as-of",
            dest="as_of",
            help="point-in-time cutoff (decision date); market data cut on "
            "trade_date, financials on ann_date",
        )
    p.add_argument("--fields", help="override default field list for this API")
    p.add_argument("--format", default="json", help="stdout format: json|csv (default json)")
    p.add_argument("--out", help="write items to this CSV file instead of stdout")


def build_parser(cfg: dict) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tushare_skill",
        description="Point-in-time safe Tushare Pro CLI (see SKILL.md)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("daily", help="stock daily bars (+qfq adjustment)")
    p.add_argument("--ts-code", required=True, help="e.g. 600519.SH")
    p.add_argument(
        "--adjust",
        default=cfg["defaults"].get("adjust", "qfq"),
        help="qfq (forward-adjusted, default) | raw",
    )
    _add_common(p, cfg)
    p.set_defaults(func=cmd_daily)

    p = sub.add_parser("index", help="index daily bars (default CSI300)")
    p.add_argument("--ts-code", default=cfg["defaults"].get("index_code", "000300.SH"))
    _add_common(p, cfg)
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("valuation", help="daily_basic: PE/PB/PS/turnover/mv")
    p.add_argument("--ts-code", required=True)
    _add_common(p, cfg)
    p.set_defaults(func=cmd_valuation)

    p = sub.add_parser("financials", help="fina_indicator, cut by ann_date <= as-of")
    p.add_argument("--ts-code", required=True)
    _add_common(p, cfg)
    p.set_defaults(func=cmd_financials)

    p = sub.add_parser("stock-info", help="stock_basic snapshot (NOT point-in-time)")
    p.add_argument("--ts-code", help="optional filter, e.g. 600519.SH")
    _add_common(p, cfg, dates=False)
    p.set_defaults(func=cmd_stock_info)

    p = sub.add_parser("trade-cal", help="trading calendar")
    p.add_argument("--exchange", default=cfg["defaults"].get("exchange", "SSE"))
    _add_common(p, cfg)
    p.set_defaults(func=cmd_trade_cal)

    return parser


def normalize_args(args: argparse.Namespace) -> None:
    for attr in ("start", "end", "as_of"):
        if getattr(args, attr, None):
            setattr(args, attr, normalize_date(getattr(args, attr), attr))
    if getattr(args, "adjust", None):
        args.adjust = validate_choice(args.adjust, VALID_ADJUST, "adjust")
    if getattr(args, "format", None):
        args.format = validate_choice(args.format, VALID_FORMAT, "format")


def main(argv: Optional[list[str]] = None) -> None:
    cfg = load_config()
    parser = build_parser(cfg)
    args = parser.parse_args(argv)

    try:
        normalize_args(args)
    except ValueError as exc:
        parser.error(str(exc))

    token = resolve_token(cfg)
    df, params = args.func(args, cfg, token)
    emit(df, args.command, params, args)


if __name__ == "__main__":
    main()
