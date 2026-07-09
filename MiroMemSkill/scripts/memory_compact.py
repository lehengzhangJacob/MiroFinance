# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

"""One-off memory compaction: drop near-duplicate entries (keep first seen).

Historical memory banks were written without write-side dedup, so retrieval
slots were often wasted on repeated generic lessons. This rewrites the JSONL
keeping the first occurrence of each near-duplicate cluster (token-set
Jaccard on the same tokenizer used by retrieval). A timestamped .bak copy of
the original file is kept next to it.

Usage:
    uv run python scripts/memory_compact.py --namespace ashare [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.memory.store import _tokenize  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--namespace", required=True)
    ap.add_argument("--store-dir", default="memory_bank")
    ap.add_argument("--kind", default="episodic", choices=["episodic", "semantic"])
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    path = REPO_ROOT / args.store_dir / f"{args.namespace}_{args.kind}.jsonl"
    if not path.exists():
        sys.exit(f"not found: {path}")

    entries = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    kept: list[dict] = []
    kept_tokens: list[set[str]] = []
    dropped = 0
    for entry in entries:
        tokens = set(_tokenize(entry.get("content", "")))
        is_dup = False
        if tokens:
            for existing in kept_tokens:
                if existing and len(tokens & existing) / len(tokens | existing) >= args.threshold:
                    is_dup = True
                    break
        if is_dup:
            dropped += 1
        else:
            kept.append(entry)
            kept_tokens.append(tokens)

    print(
        f"{path.name}: {len(entries)} -> {len(kept)} "
        f"(dropped {dropped} near-duplicates, threshold={args.threshold})"
    )
    if args.dry_run:
        return

    backup = path.with_name(f"{path.name}.bak-{time.strftime('%Y%m%d-%H%M%S')}")
    shutil.copy2(path, backup)
    with open(path, "w", encoding="utf-8") as f:
        for entry in kept:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"backup -> {backup.name}")


if __name__ == "__main__":
    main()
