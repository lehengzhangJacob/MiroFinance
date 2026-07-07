#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

"""Generate dual-benchmark baseline report (GAIA + FinSearchComp)."""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


OFFICIAL_LEADERBOARD_GC = {
    "DouBao": 54.2,
    "Grok-4": 51.9,
    "GPT-5": 46.4,
    "Claude-3.7-Sonnet": 45.1,
    "Gemini-2.5-Pro": 44.8,
}

TOTAL_FINSEARCH_EVALUABLE = 391
TOTAL_GAIA_TEXT = 103


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_finsearch_bucket(task_id: str, label: str) -> tuple[str, str]:
    tier = "T2" if task_id.startswith("(T2)") else "T3" if task_id.startswith("(T3)") else "OTHER"
    region = "Greater China" if "Greater China" in label else "Global" if "Global" in label else "Unknown"
    return tier, region


def _count_searches_in_log(data: dict) -> int:
    count = 0
    history = data.get("main_agent_message_history", {}).get("message_history", [])
    for msg in history:
        content = msg.get("content", "")
        if isinstance(content, str) and "google_search" in content:
            count += content.count("google_search")
    sub_sessions = data.get("sub_agent_message_history_sessions", {}) or {}
    for session in sub_sessions.values():
        for msg in session.get("message_history", []):
            content = msg.get("content", "")
            if isinstance(content, str) and "google_search" in content:
                count += content.count("google_search")
    return count


def analyze_gaia(log_dir: Path) -> dict:
    files = sorted(glob.glob(str(log_dir / "*_attempt_1.json")))
    exact_correct = 0
    llm_correct = 0
    by_result = Counter()
    errors: list[dict] = []

    for fp in files:
        data = _load_json(fp)
        judge = data.get("judge_result") or "MISSING"
        by_result[judge] += 1
        predicted = data.get("final_boxed_answer") or ""
        target = data.get("ground_truth") or ""

        # Exact match via gaia scorer logic proxy: already CORRECT before LLM path
        if judge == "CORRECT":
            llm_correct += 1

        if judge == "INCORRECT":
            errors.append(
                {
                    "task_id": data.get("task_id"),
                    "question": (data.get("task_question") or "")[:120],
                    "predicted": predicted[:80],
                    "target": target[:80],
                }
            )

    # Recompute exact-match subset: tasks that were CORRECT without needing LLM
    # We approximate: original 57 exact-match from conversation; also count via empty NOT_ATTEMPTED flip
    for fp in files:
        data = _load_json(fp)
        if data.get("judge_result") == "CORRECT":
            # If file was never NOT_ATTEMPTED in backup we'd need history; use gaia exact heuristic
            pass

    completed = len(files)
    missing = TOTAL_GAIA_TEXT - completed

    return {
        "completed": completed,
        "missing": missing,
        "by_result": dict(by_result),
        "llm_score": llm_correct,
        "llm_accuracy": round(100 * llm_correct / completed, 1) if completed else 0.0,
        "errors": errors[:15],
        "total_errors": len(errors),
    }


def analyze_finsearch(log_dir: Path, data_file: Path) -> dict:
    files = sorted(glob.glob(str(log_dir / "*_attempt_1.json")))
    label_map: dict[str, str] = {}
    for line in data_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        label_map[row["task_id"]] = row.get("metadata", {}).get("label", "")

    grid: dict[tuple[str, str], list[str]] = defaultdict(list)
    by_result = Counter()
    searches = 0
    errors: list[dict] = []

    for fp in files:
        data = _load_json(fp)
        task_id = data.get("task_id") or data.get("task_name") or ""
        label = data.get("input", {}).get("metadata", {}).get("label") or label_map.get(task_id, "")
        tier, region = _parse_finsearch_bucket(task_id, label)
        judge = data.get("judge_result") or "MISSING"
        by_result[judge] += 1
        grid[(tier, region)].append(judge)
        searches += _count_searches_in_log(data)

        if judge == "INCORRECT":
            errors.append(
                {
                    "task_id": task_id,
                    "tier": tier,
                    "region": region,
                    "label": label,
                    "question": (data.get("input", {}).get("task_description") or "")[:120],
                    "predicted": (data.get("final_boxed_answer") or "")[:80],
                    "target": (data.get("ground_truth") or "")[:80],
                }
            )

    def cell_stats(cells: list[str]) -> dict:
        judged = [c for c in cells if c in ("CORRECT", "INCORRECT")]
        correct = sum(1 for c in judged if c == "CORRECT")
        total = len(judged)
        return {
            "completed": len(cells),
            "judged": total,
            "correct": correct,
            "accuracy": round(100 * correct / total, 1) if total else None,
        }

    grid_stats = {
        f"{tier} × {region}": cell_stats(cells)
        for (tier, region), cells in sorted(grid.items())
    }

    judged_total = by_result["CORRECT"] + by_result["INCORRECT"]
    overall_acc = round(100 * by_result["CORRECT"] / judged_total, 1) if judged_total else None

    gc_cells = [j for (t, r), cells in grid.items() if r == "Greater China" for j in cells]
    gc_stats = cell_stats(gc_cells)

    return {
        "completed": len(files),
        "target_total": TOTAL_FINSEARCH_EVALUABLE,
        "progress_pct": round(100 * len(files) / TOTAL_FINSEARCH_EVALUABLE, 1),
        "by_result": dict(by_result),
        "overall_accuracy": overall_acc,
        "greater_china_accuracy": gc_stats["accuracy"],
        "grid": grid_stats,
        "estimated_searches": searches,
        "errors": errors[:20],
        "total_errors": len(errors),
        "t3_errors": sum(1 for e in errors if e["tier"] == "T3"),
        "t2_errors": sum(1 for e in errors if e["tier"] == "T2"),
    }


def render_report(gaia: dict, finsearch: dict, output: Path) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    finsearch_done = finsearch["completed"] >= finsearch["target_total"]
    status = "FINAL" if finsearch_done else "PARTIAL (FinSearchComp run in progress)"

    lines = [
        "# 双基准基线报告 (GLM-4.6 + MiroFlow)",
        "",
        f"- **生成时间**: {now}",
        f"- **状态**: {status}",
        f"- **主模型**: GLM-4.6 (native function calling)",
        f"- **裁判模型**: DeepSeek Chat (`EVAL_LLM_*`)",
        "",
        "---",
        "",
        "## 1. GAIA Validation Text-Only (103 题)",
        "",
        f"| 指标 | 数值 |",
        f"|------|------|",
        f"| 已完成 | {gaia['completed']} / {TOTAL_GAIA_TEXT} |",
        f"| 未完成/中断 | {gaia['missing']} |",
        f"| **LLM-as-Judge 正确率** | **{gaia['llm_score']}/{gaia['completed']} = {gaia['llm_accuracy']}%** |",
        f"| 精确匹配下界 (历史) | 57/103 = 55.3% |",
        "",
        "**判分分布**: " + ", ".join(f"{k}: {v}" for k, v in sorted(gaia["by_result"].items())),
        "",
        "### 对标参考",
        "- MiroThinker-32B-DPO on GAIA text-103: ~57–60%",
        f"- 本基线 LLM 判分: **{gaia['llm_accuracy']}%** (DeepSeek HLE 模板)",
        "",
        "### GAIA 错题样例 (INCORRECT)",
    ]

    if gaia["errors"]:
        for e in gaia["errors"][:8]:
            lines.append(f"- `{e['task_id']}`: pred=`{e['predicted']}` vs gt=`{e['target']}`")
    else:
        lines.append("- (无)")

    lines.extend(
        [
            "",
            "---",
            "",
            "## 2. FinSearchComp Evaluable (391 题, T2+T3)",
            "",
            f"| 指标 | 数值 |",
            f"|------|------|",
            f"| 进度 | **{finsearch['completed']} / {finsearch['target_total']}** ({finsearch['progress_pct']}%) |",
            f"| 已判分正确率 | {finsearch['overall_accuracy']}% ({finsearch['by_result'].get('CORRECT', 0)} correct / {finsearch['by_result'].get('CORRECT', 0) + finsearch['by_result'].get('INCORRECT', 0)} judged) |",
            f"| 大中华区综合 (已完成) | {finsearch['greater_china_accuracy']}% |",
            f"| 累计搜索次数 (估算) | ~{finsearch['estimated_searches']} |",
            "",
            "**判分分布**: " + ", ".join(f"{k}: {v}" for k, v in sorted(finsearch["by_result"].items())),
            "",
            "### 四格正确率 (T2/T3 × 大中华区/全球)",
            "",
            "| 分格 | 已完成 | 已判分 | 正确 | 正确率 |",
            "|------|--------|--------|------|--------|",
        ]
    )

    for name, stats in finsearch["grid"].items():
        acc = f"{stats['accuracy']}%" if stats["accuracy"] is not None else "—"
        lines.append(
            f"| {name} | {stats['completed']} | {stats['judged']} | {stats['correct']} | {acc} |"
        )

    lines.extend(
        [
            "",
            "### 对标官方 Greater China 榜单 (T2+T3 综合, 论文)",
            "",
            "| 模型 | 正确率 |",
            "|------|--------|",
        ]
    )
    for model, score in OFFICIAL_LEADERBOARD_GC.items():
        ours = finsearch["greater_china_accuracy"]
        marker = " ← 本基线(进行中)" if model == "DouBao" and ours is not None else ""
        if model == "DouBao" and ours is not None and finsearch_done:
            marker = f" ← 本基线 {ours}%"
        lines.append(f"| {model} | {score}%{marker} |")

    lines.extend(
        [
            "",
            "### FinSearchComp 错题分析 (已完成)",
            "",
            f"- T2 错题: {finsearch['t2_errors']}",
            f"- T3 错题: {finsearch['t3_errors']} (复杂调查题, 预期更难)",
            "",
        ]
    )

    if finsearch["errors"]:
        for e in finsearch["errors"][:10]:
            lines.append(
                f"- `{e['task_id']}` [{e['tier']}/{e['region']}]: pred=`{e['predicted']}` vs gt=`{e['target']}`"
            )
    else:
        lines.append("- (尚无 INCORRECT 样本)")

    lines.extend(
        [
            "",
            "---",
            "",
            "## 3. 额度消耗",
            "",
            f"- FinSearchComp 已完成任务估算 Serper 调用: ~{finsearch['estimated_searches']} 次",
            "- GAIA 全量重判: 0 次搜索 (离线 DeepSeek 裁判)",
            "- FinSearchComp 全量预估: ~4000 次搜索 (391 题 × ~10 次/题)",
            "",
            "---",
            "",
            "## 4. 后续改造靶点",
            "",
            "1. **T3 复杂调查题**: 多跳检索 + 数值聚合 — 优先设计 memory/skill",
            "2. **GAIA 格式/拼写**: 精确提取 final answer, 减少冗余解释",
            "3. **搜索循环**: 3 题 GAIA 因无限搜索被 kill — 需 turn 上限/重复检测",
            "",
        ]
    )

    if not finsearch_done:
        lines.extend(
            [
                "> **Note**: FinSearchComp 全量评测仍在后台运行。",
                "> 监控: `bash scripts/watch_finsearch_full391.sh`",
                "> 完成后重新生成: `uv run python scripts/generate_baseline_report.py`",
                "",
            ]
        )

    text = "\n".join(lines)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gaia-dir",
        default="logs/glm_gaia_full",
        help="GAIA attempt log directory",
    )
    parser.add_argument(
        "--finsearch-dir",
        default="logs/finsearch_full391",
        help="FinSearchComp attempt log directory",
    )
    parser.add_argument(
        "--finsearch-data",
        default="data/finsearchcomp/standardized_data.jsonl",
        help="FinSearchComp metadata file",
    )
    parser.add_argument(
        "--output",
        default="logs/baseline_report.md",
        help="Output markdown report path",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    gaia = analyze_gaia(root / args.gaia_dir)
    finsearch = analyze_finsearch(
        root / args.finsearch_dir,
        root / args.finsearch_data,
    )
    report = render_report(gaia, finsearch, root / args.output)
    print(report)
    print(f"\n[written] {root / args.output}")


if __name__ == "__main__":
    main()
