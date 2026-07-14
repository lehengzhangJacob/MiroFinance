# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

"""Offline re-judge benchmark logs without re-running agents."""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import sys
from collections import Counter
from pathlib import Path

import openai

from src.utils.env_loader import load_project_env
from utils.eval_utils import verify_answer_for_datasets


def _collect_log_files(input_dir: str, pattern: str) -> list[str]:
    return sorted(glob.glob(os.path.join(input_dir, pattern)))


async def rejudge_logs(
    input_dir: str,
    benchmark_name: str,
    pattern: str = "*_attempt_1.json",
    only_not_attempted: bool = True,
    update_files: bool = True,
) -> Counter:
    load_project_env()
    client = openai.AsyncOpenAI(
        api_key=os.getenv("EVAL_LLM_API_KEY"),
        base_url=os.getenv("EVAL_LLM_BASE_URL") or None,
    )

    log_files = _collect_log_files(input_dir, pattern)
    if not log_files:
        print(f"No log files matching {pattern} in {input_dir}")
        sys.exit(1)

    results: Counter = Counter()
    for log_file in log_files:
        with open(log_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        prior = data.get("judge_result") or ""
        if only_not_attempted and prior not in ("NOT_ATTEMPTED", ""):
            results[prior] += 1
            continue

        predicted = data.get("final_boxed_answer") or data.get("model_boxed_answer") or ""
        if not predicted:
            print(f"SKIP {Path(log_file).name}: no predicted answer")
            results["SKIP_NO_ANSWER"] += 1
            continue

        question = data.get("task_question") or data.get("question") or ""
        ground_truth = data.get("ground_truth") or ""
        metadata = data.get("input", {}).get("metadata", data.get("metadata", {}))

        result = await verify_answer_for_datasets(
            openai_client=client,
            benchmark_name=benchmark_name,
            question=question,
            target=ground_truth,
            predicted_answer=predicted,
            metadata=metadata,
        )
        print(f"{Path(log_file).name}: {prior or 'MISSING'} -> {result}")

        if update_files:
            data["judge_result"] = result
            with open(log_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        results[result] += 1

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline LLM re-judge for benchmark logs")
    parser.add_argument("input_dir", help="Directory containing attempt JSON logs")
    parser.add_argument(
        "--benchmark-name",
        default="gaia-validation-text-only",
        help="Benchmark name passed to verify_answer_for_datasets",
    )
    parser.add_argument(
        "--pattern",
        default="*_attempt_1.json",
        help="Glob pattern for log files",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Re-judge all logs, not only NOT_ATTEMPTED/empty",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print results without updating log files",
    )
    args = parser.parse_args()

    summary = asyncio.run(
        rejudge_logs(
            input_dir=args.input_dir,
            benchmark_name=args.benchmark_name,
            pattern=args.pattern,
            only_not_attempted=not args.all,
            update_files=not args.dry_run,
        )
    )

    print("\n=== Re-judge summary ===")
    for key, count in sorted(summary.items()):
        print(f"{key}: {count}")


if __name__ == "__main__":
    main()
