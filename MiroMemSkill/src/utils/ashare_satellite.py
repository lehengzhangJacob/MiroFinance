"""Strict loader for point-in-time A-share excess-return satellite signals."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd


DEFAULT_EXCESS_SIGNAL_FILE = "qlib_excess_signal.csv"
EXCESS_TARGET = "excess_vs_000300.SH"
_CODE_RE = re.compile(r"^\d{6}\.(?:SH|SZ|BJ)$")
_REQUIRED_COLUMNS = {
    "signal_date",
    "ts_code",
    "score",
    "rank",
    "n_stocks",
    "train_end",
    "target",
}


def normalize_signal_date(value: Any) -> str:
    """Normalize a signal cutoff to compact YYYYMMDD."""
    compact = str(value).strip().replace("-", "").replace("/", "")
    if len(compact) != 8 or not compact.isdigit():
        raise ValueError(
            f"invalid signal date {value!r}; expected YYYYMMDD or YYYY-MM-DD"
        )
    return compact


def _resolve_signal_path(
    data_dir: str | Path,
    signal_file: str | Path,
) -> Path:
    root = Path(data_dir).resolve()
    supplied = Path(signal_file)
    path = supplied.resolve() if supplied.is_absolute() else (root / supplied).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"signal CSV must be under data_dir: {path}") from exc
    if not path.exists():
        raise FileNotFoundError(f"excess signal CSV not found: {path}")
    return path


def _manifest_candidates(signal_path: Path) -> list[Path]:
    candidates = [
        signal_path.with_name(f"{signal_path.stem}_manifest.json"),
        signal_path.with_suffix(f"{signal_path.suffix}.manifest.json"),
        signal_path.with_suffix(".manifest.json"),
    ]
    unique: list[Path] = []
    for candidate in candidates:
        if candidate not in unique and candidate.exists():
            unique.append(candidate)
    return unique


def _manifest_checksum(payload: Mapping[str, Any], signal_path: Path) -> str:
    checksum = payload.get("sha256")
    if checksum:
        declared_file = payload.get("file") or payload.get("filename")
        if declared_file and Path(str(declared_file)).name != signal_path.name:
            raise ValueError(
                f"manifest names {declared_file!r}, expected {signal_path.name!r}"
            )
        return str(checksum).lower()

    files = payload.get("files")
    if isinstance(files, list):
        matches = [
            item
            for item in files
            if isinstance(item, Mapping)
            and Path(str(item.get("path") or item.get("file") or "")).name
            == signal_path.name
        ]
        if len(matches) == 1 and matches[0].get("sha256"):
            return str(matches[0]["sha256"]).lower()
    raise ValueError(f"manifest has no checksum for {signal_path.name}")


def validate_excess_signal_manifest(signal_path: str | Path) -> list[Path]:
    """Validate every recognized adjacent manifest, if one exists."""
    path = Path(signal_path)
    manifests = _manifest_candidates(path)
    if not manifests:
        return []
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    for manifest in manifests:
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid excess signal manifest: {manifest}") from exc
        expected = _manifest_checksum(payload, path)
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError(f"invalid sha256 in excess signal manifest: {manifest}")
        if actual != expected:
            raise ValueError(
                f"checksum mismatch for {path.name} against {manifest.name}"
            )
    return manifests


def _normalize_code(value: Any) -> str:
    code = str(value).strip().upper()
    if not _CODE_RE.fullmatch(code):
        raise ValueError(f"invalid A-share code in excess signal: {value!r}")
    return code


def _excluded_codes(momentum_top4: Iterable[Any]) -> set[str]:
    excluded: set[str] = set()
    for item in momentum_top4:
        value = item.get("ts_code") if isinstance(item, Mapping) else item
        excluded.add(_normalize_code(value))
    return excluded


def _validated_signal_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        dtype={
            "signal_date": str,
            "ts_code": str,
            "train_end": str,
            "target": str,
        },
    )
    missing = _REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"{path.name} missing columns: {sorted(missing)}")
    if frame.empty:
        return frame

    frame = frame.copy()
    frame["signal_date"] = frame["signal_date"].map(normalize_signal_date)
    frame["train_end"] = frame["train_end"].map(normalize_signal_date)
    frame["ts_code"] = frame["ts_code"].map(_normalize_code)
    frame["score"] = pd.to_numeric(frame["score"], errors="coerce")
    if not frame["score"].map(
        lambda value: pd.notna(value) and math.isfinite(float(value))
    ).all():
        raise ValueError(f"{path.name} contains a non-finite score")

    for column in ("rank", "n_stocks"):
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.isna().any() or (numeric <= 0).any() or (numeric % 1 != 0).any():
            raise ValueError(f"{path.name} contains invalid {column}")
        frame[column] = numeric.astype(int)

    if not frame["target"].eq(EXCESS_TARGET).all():
        unexpected = sorted(set(frame.loc[frame["target"] != EXCESS_TARGET, "target"]))
        raise ValueError(f"{path.name} contains unexpected targets: {unexpected}")
    if not frame["train_end"].lt(frame["signal_date"]).all():
        raise ValueError(f"{path.name} contains train_end on/after signal_date")
    if frame.duplicated(["signal_date", "ts_code"]).any():
        raise ValueError(f"{path.name} contains duplicate signal-date stock rows")
    return frame


def load_excess_signal_candidates(
    as_of: str,
    momentum_top4: Iterable[Any],
    *,
    data_dir: str | Path,
    exact_date: bool = False,
    signal_file: str | Path = DEFAULT_EXCESS_SIGNAL_FILE,
) -> list[dict[str, Any]]:
    """Load the latest eligible excess-return candidates at a point in time.

    By default the latest ``signal_date <= as_of`` is selected. With
    ``exact_date=True``, stale monthly signals are rejected. Momentum top-four
    names are removed, then ties are deterministically ordered by stock code.
    """
    cutoff = normalize_signal_date(as_of)
    path = _resolve_signal_path(data_dir, signal_file)
    validate_excess_signal_manifest(path)
    frame = _validated_signal_frame(path)
    if frame.empty:
        return []

    eligible = frame[
        frame["signal_date"].eq(cutoff)
        if exact_date
        else frame["signal_date"].le(cutoff)
    ]
    if eligible.empty:
        return []
    signal_date = cutoff if exact_date else str(eligible["signal_date"].max())
    eligible = eligible[eligible["signal_date"].eq(signal_date)]
    eligible = eligible[
        ~eligible["ts_code"].isin(_excluded_codes(momentum_top4))
    ].copy()
    if eligible.empty:
        return []

    # Two stable passes give the explicit ordering: score descending, then code
    # ascending for ties.
    eligible = eligible.sort_values(
        "ts_code", ascending=True, kind="mergesort"
    ).sort_values("score", ascending=False, kind="mergesort")
    return eligible.to_dict(orient="records")


def render_excess_signal_candidates(
    as_of: str,
    momentum_top4: Iterable[Any],
    *,
    data_dir: str | Path,
    exact_date: bool = False,
    signal_file: str | Path = DEFAULT_EXCESS_SIGNAL_FILE,
) -> str:
    """Render eligible satellite candidates without any realized labels."""
    rows = load_excess_signal_candidates(
        as_of,
        momentum_top4,
        data_dir=data_dir,
        exact_date=exact_date,
        signal_file=signal_file,
    )
    if not rows:
        return f"No eligible excess-return satellite signal on or before {as_of}."
    frame = pd.DataFrame(rows)
    signal_date = str(frame.iloc[0]["signal_date"])
    return "\n".join(
        [
            "# A股超额收益卫星候选（严格点时）",
            (
                f"as_of={normalize_signal_date(as_of)}；signal_date={signal_date}；"
                f"target={EXCESS_TARGET}；已排除相对动量 top4。"
            ),
            "仅包含预测分数与训练截止信息，不包含未来或已实现标签。",
            frame.to_csv(index=False),
        ]
    )
