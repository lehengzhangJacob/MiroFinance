#!/usr/bin/env python3
"""Offline CLI for the point-in-time A-share relative-momentum soft anchor."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import yaml

SKILL_DIR = Path(__file__).resolve().parent
REPO_ROOT = SKILL_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.ashare_momentum import build_relative_momentum_baseline  # noqa: E402


def load_config() -> dict[str, Any]:
    return yaml.safe_load((SKILL_DIR / "config.yaml").read_text(encoding="utf-8"))


def resolve_data_dir(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (SKILL_DIR / path).resolve()


def load_stock_pool(data_dir: Path) -> dict[str, dict[str, Any]]:
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    pool = meta.get("stock_pool")
    if not isinstance(pool, dict) or not pool:
        raise ValueError(f"stock_pool missing or empty in {data_dir / 'meta.json'}")
    return pool


def build_parser(cfg: dict[str, Any]) -> argparse.ArgumentParser:
    defaults = cfg["defaults"]
    parser = argparse.ArgumentParser(
        prog="ashare_momentum_relative_strength",
        description="Point-in-time relative-momentum top-k soft anchor",
    )
    parser.add_argument("--as-of", required=True, help="decision date YYYYMMDD or YYYY-MM-DD")
    parser.add_argument(
        "--window",
        type=int,
        choices=cfg["valid_windows"],
        default=int(defaults["window"]),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        choices=range(1, 5),
        default=int(defaults["top_k"]),
    )
    parser.add_argument(
        "--data-dir",
        default=cfg["data_dir"],
        help="directory containing meta.json and local A-share CSV files",
    )
    parser.add_argument(
        "--format",
        choices=("json", "csv"),
        default=str(defaults["format"]),
    )
    parser.add_argument("--out", help="optional output file")
    return parser


def emit(result: dict[str, Any], output_format: str, out: Optional[str]) -> None:
    if output_format == "csv":
        text = pd.DataFrame(result["ranking"]).to_csv(index=False)
    else:
        text = json.dumps(result, ensure_ascii=False, indent=2)

    if out:
        output = Path(out).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
        print(f"wrote momentum baseline -> {output}")
    else:
        print(text, end="" if text.endswith("\n") else "\n")


def main(argv: Optional[list[str]] = None) -> None:
    cfg = load_config()
    args = build_parser(cfg).parse_args(argv)
    data_dir = resolve_data_dir(args.data_dir)
    result = build_relative_momentum_baseline(
        args.as_of,
        load_stock_pool(data_dir),
        data_dir=data_dir,
        window=args.window,
        top_k=args.top_k,
        max_stock_weight=float(cfg["defaults"]["max_stock_weight"]),
    )
    emit(result, args.format, args.out)


if __name__ == "__main__":
    main()
