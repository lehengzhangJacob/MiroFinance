# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

"""Mem0-style memory facade: two-phase add (extract -> consolidate) + filtered search.

Pipeline per finished task (cf. mem0ai/mem0, arXiv:2504.19413):
  1. Extraction phase: one LLM call distills 0-2 atomic conditional lessons
     from the task trajectory + judge result. Strict JSON; unparseable output
     is retried once, then DISCARDED — raw LLM text is never stored.
  2. Update phase: each candidate is compared against the top-5 most similar
     existing memories; an LLM decides ADD / UPDATE (merge) / DELETE
     (contradicted) / NONE (duplicate). Every operation lands in the
     namespace's history JSONL.

Retrieval applies metadata filters:
  - before_month: only memories whose source market month is strictly earlier
    than the current task's month are visible (kills look-ahead leakage under
    any task ordering).
  - functional-stance quota: a lesson's direction pressure is derived from
    metadata (predicted_direction x judge_result), not keyword scanning — a
    counter-lesson from a wrong 跑输 prediction pushes bullish, so it counts
    against the bullish quota.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import Any, Optional

import requests
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from src.memory.prompts import (
    EXTRACTION_CORRECT_INSTRUCTIONS,
    EXTRACTION_INCORRECT_INSTRUCTIONS,
    EXTRACTION_PROMPT,
    MONTHLY_REFLECTION_PROMPT,
    UPDATE_PROMPT,
)
from src.memory.vector_store import MemoryRecord, VectorStore
from src.utils.env_loader import load_project_env

# Below this similarity the update phase is skipped and the candidate is
# ADDed directly: nothing in the bank is close enough to consolidate with.
_CONSOLIDATION_SIM_THRESHOLD = 0.45


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, requests.exceptions.HTTPError):
        status = exc.response.status_code if exc.response is not None else 0
        return status == 429 or status >= 500
    return isinstance(exc, (requests.exceptions.ConnectionError, requests.exceptions.Timeout))


def extract_direction(answer: str) -> str:
    """Pull the predicted direction (跑赢/跑输) out of a final answer."""
    text = answer or ""
    if "跑赢" in text and "跑输" not in text:
        return "跑赢"
    if "跑输" in text and "跑赢" not in text:
        return "跑输"
    m = re.search(r"\\boxed\{(跑赢|跑输)\}", text)
    return m.group(1) if m else ""


def functional_stance(predicted_direction: str, judge_result: str) -> str:
    """Direction a lesson pushes FUTURE predictions toward.

    Correct prediction  -> lesson reinforces that direction.
    Incorrect prediction -> counter-lesson pushes the opposite direction
    (e.g. "predicting 跑输 on weak momentum failed" nudges bullish).
    """
    if predicted_direction not in ("跑赢", "跑输"):
        return "neutral"
    reinforcing = judge_result == "CORRECT"
    bullish_pred = predicted_direction == "跑赢"
    return "bullish" if bullish_pred == reinforcing else "bearish"


def parse_task_month(task_id: str) -> str:
    """ashare task ids end with the entry date: ashare_300012_2025-04-01 -> 2025-04."""
    m = re.search(r"(\d{4})-(\d{2})-\d{2}$", task_id or "")
    return f"{m.group(1)}-{m.group(2)}" if m else ""


def summarize_trajectory(log_data: dict[str, Any], max_chars: int = 3000) -> str:
    parts: list[str] = []
    history = (log_data or {}).get("main_agent_message_history", {}).get("message_history", [])
    for msg in history[-12:]:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if isinstance(content, list):
            text = " ".join(
                block.get("text", "") for block in content if isinstance(block, dict)
            )
        else:
            text = str(content)
        text = re.sub(r"\s+", " ", text).strip()[:400]
        if text:
            parts.append(f"[{role}] {text}")
    return "\n".join(parts)[:max_chars]


class Mem0Memory:
    """Two-phase memory over a VectorStore, plus stance/temporal-filtered search."""

    def __init__(
        self,
        store: VectorStore,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        load_project_env()
        self.store = store
        self.model = model or os.getenv("REFLECTION_LLM_MODEL_NAME") or os.getenv(
            "DEEPSEEK_MODEL_NAME"
        ) or os.getenv("GLM_MODEL_NAME", "deepseek-chat")
        self.base_url = base_url or os.getenv("REFLECTION_LLM_BASE_URL") or os.getenv(
            "DEEPSEEK_BASE_URL"
        ) or os.getenv("GLM_BASE_URL", "https://api.deepseek.com/v1")
        self.api_key = api_key or os.getenv("REFLECTION_LLM_API_KEY") or os.getenv(
            "DEEPSEEK_API_KEY"
        ) or os.getenv("GLM_API_KEY", "")

    # ---------------------------------------------------------------- llm io

    @retry(
        retry=retry_if_exception(_is_retryable),
        wait=wait_exponential(multiplier=5, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _call_llm(self, prompt: str, max_tokens: int = 2048, json_mode: bool = True) -> str:
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        resp = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=90,
        )
        if resp.status_code == 400 and json_mode:
            # Endpoint may not support structured output; retry plain once.
            return self._call_llm(prompt, max_tokens=max_tokens, json_mode=False)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    @staticmethod
    def _parse_json(raw: str) -> Optional[dict[str, Any]]:
        """Strict JSON parse with fence stripping; None (never raw text) on failure."""
        text = (raw or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        try:
            data = json.loads(text)
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            pass
        # Salvage attempt: outermost braces (handles leading/trailing prose).
        start, end = text.find("{"), text.rfind("}")
        if 0 <= start < end:
            try:
                data = json.loads(text[start : end + 1])
                return data if isinstance(data, dict) else None
            except json.JSONDecodeError:
                return None
        return None

    def _llm_json(self, prompt: str, max_tokens: int = 2048) -> Optional[dict[str, Any]]:
        """One retry on unparseable output, then give up (discard, never store raw)."""
        for _ in range(2):
            try:
                raw = self._call_llm(prompt, max_tokens=max_tokens)
            except Exception as exc:
                print(f"    memory LLM call failed: {exc}")
                return None
            data = self._parse_json(raw)
            if data is not None:
                return data
        return None

    # -------------------------------------------------------------- add path

    def add(
        self,
        question: str,
        answer: str,
        judge_result: str,
        log_data: Optional[dict[str, Any]] = None,
        task_id: str = "",
        ts_code: str = "",
    ) -> list[str]:
        """Extraction phase -> update phase. Returns human-readable op summaries."""
        if not self.api_key:
            return []

        is_incorrect = judge_result == "INCORRECT"
        prompt = EXTRACTION_PROMPT.format(
            outcome_instructions=(
                EXTRACTION_INCORRECT_INSTRUCTIONS if is_incorrect else EXTRACTION_CORRECT_INSTRUCTIONS
            ),
            question=question[:800],
            answer=(answer or "N/A")[:400],
            judge_result=judge_result,
            trajectory=summarize_trajectory(log_data or {}) or "N/A",
        )
        data = self._llm_json(prompt)
        if data is None:
            print(f"    [{task_id}] extraction unparseable after retry — discarded")
            return []
        lessons = data.get("lessons", [])
        if not isinstance(lessons, list):
            return []
        # Symmetric cap: the old 2-correct/1-incorrect asymmetry minted lessons
        # from the majority-direction (bearish) wins at twice the rate and was
        # a mechanical amplifier of the bank's direction bias.
        max_lessons = 1

        predicted = extract_direction(answer)
        entry_month = parse_task_month(task_id)
        stance = functional_stance(predicted, judge_result)
        ops: list[str] = []

        for lesson in lessons[:max_lessons]:
            if not isinstance(lesson, dict):
                continue
            content = str(lesson.get("content", "")).strip()
            if len(content) < 20:  # degenerate/empty extraction
                continue
            tags = [str(t) for t in lesson.get("tags", []) if str(t).strip()]
            metadata = {
                "task_id": task_id,
                "ts_code": ts_code,
                "entry_month": entry_month,
                "judge_result": judge_result,
                "predicted_direction": predicted,
                "functional_stance": stance,
                "tags": tags,
                "source": "reflection",
                "source_tasks": [task_id],
            }
            op = self._consolidate(content, metadata, task_id)
            if op:
                ops.append(op)
        return ops

    def _consolidate(self, content: str, metadata: dict[str, Any], task_id: str) -> str:
        """Mem0 update phase for one candidate lesson."""
        try:
            candidate_emb = self.store.embed(content)
        except Exception as exc:
            print(f"    [{task_id}] embedding failed, lesson discarded: {exc}")
            return ""

        similar = self.store.search(query_embedding=candidate_emb, top_k=5)
        if not similar or similar[0][1] < _CONSOLIDATION_SIM_THRESHOLD:
            self.store.add(content, metadata=metadata, embedding=candidate_emb, source_task=task_id)
            return f"ADD: {content[:60]}"

        existing_block = "\n".join(
            f"- id={rec.id}\n  content={rec.content}" for rec, _ in similar
        )
        decision = self._llm_json(
            UPDATE_PROMPT.format(candidate=content, existing=existing_block),
            max_tokens=4096,  # reasoning models need headroom before content
        )
        if decision is None:
            # Consolidation undecidable -> conservative ADD (never lose a parsed lesson).
            self.store.add(content, metadata=metadata, embedding=candidate_emb, source_task=task_id)
            return f"ADD(fallback): {content[:60]}"

        action = str(decision.get("action", "ADD")).upper()
        target_id = str(decision.get("target_id", "") or "")
        new_content = str(decision.get("new_content", "") or "").strip()
        known_ids = {rec.id for rec, _ in similar}

        if action == "NONE":
            return f"NONE(dup): {content[:60]}"

        if action == "UPDATE" and target_id in known_ids and len(new_content) >= 20:
            target = next(rec for rec, _ in similar if rec.id == target_id)
            merged_meta = self._merge_metadata(target, metadata)
            self.store.update(target_id, new_content, metadata_patch=merged_meta, source_task=task_id)
            return f"UPDATE {target_id[:8]}: {new_content[:60]}"

        if action == "DELETE" and target_id in known_ids:
            self.store.delete(target_id, source_task=task_id, reason="contradicted by new evidence")
            self.store.add(content, metadata=metadata, embedding=candidate_emb, source_task=task_id)
            return f"DELETE {target_id[:8]} + ADD: {content[:60]}"

        # ADD, or malformed UPDATE/DELETE payload -> safe default.
        self.store.add(content, metadata=metadata, embedding=candidate_emb, source_task=task_id)
        return f"ADD: {content[:60]}"

    @staticmethod
    def _merge_metadata(target: MemoryRecord, candidate_meta: dict[str, Any]) -> dict[str, Any]:
        """Metadata for a merged lesson: visibility month is the LATER of the two
        sources (safe under the before_month filter); stances that disagree
        become 'mixed' (exempt from quota, like neutral)."""
        old_month = str(target.metadata.get("entry_month", "") or "")
        new_month = str(candidate_meta.get("entry_month", "") or "")
        months = [m for m in (old_month, new_month) if m]
        old_stance = target.metadata.get("functional_stance", "neutral")
        new_stance = candidate_meta.get("functional_stance", "neutral")
        source_tasks = list(
            dict.fromkeys(
                list(target.metadata.get("source_tasks", []))
                + list(candidate_meta.get("source_tasks", []))
            )
        )
        return {
            "entry_month": max(months) if months else "",
            "functional_stance": old_stance if old_stance == new_stance else "mixed",
            "source_tasks": source_tasks,
            "judge_result": "MERGED",
        }

    def add_monthly(self, month: str, features_table: str, n_stocks: int) -> list[str]:
        """Monthly cross-sectional reflection (v2 learning signal).

        One LLM call over the month's full decision-time feature table plus
        realized labels distills at most 2 patterns backed by n=16 evidence,
        replacing the noise-fitted n=1 per-task lessons. Candidates still run
        through the Mem0 consolidation phase.
        """
        if not self.api_key:
            return []
        # Reasoning models (deepseek-v4-pro) burn completion tokens on hidden
        # reasoning before emitting content; cross-sectional analysis needs a
        # far larger budget than simple extraction or the content comes back
        # empty at 2048.
        data = self._llm_json(
            MONTHLY_REFLECTION_PROMPT.format(month=month, n=n_stocks, table=features_table),
            max_tokens=8192,
        )
        if data is None:
            print(f"    [monthly {month}] reflection unparseable after retry — discarded")
            return []
        lessons = data.get("lessons", [])
        if not isinstance(lessons, list):
            return []

        ops: list[str] = []
        for lesson in lessons[:2]:
            if not isinstance(lesson, dict):
                continue
            content = str(lesson.get("content", "")).strip()
            if len(content) < 20:
                continue
            tags = [str(t) for t in lesson.get("tags", []) if str(t).strip()]
            metadata = {
                "task_id": f"monthly_{month}",
                "entry_month": month,
                "judge_result": "MONTHLY",
                "predicted_direction": "",
                # Cross-sectional conditionals are two-sided by construction;
                # exempt from the direction quota like neutral lessons.
                "functional_stance": "neutral",
                "tags": tags,
                "source": "monthly_reflection",
                "source_tasks": [f"monthly_{month}"],
            }
            op = self._consolidate(content, metadata, task_id=f"monthly_{month}")
            if op:
                ops.append(op)
        return ops

    def save_note(self, content: str, tags: list[str], as_of_month: str = "") -> MemoryRecord:
        """Agent-initiated note (MCP memory_save). Notes without a month are
        never visible under a before_month filter (fail-safe against leakage)."""
        return self.store.add(
            content,
            metadata={
                "source": "agent_note",
                "tags": tags,
                "entry_month": as_of_month,
                "functional_stance": "neutral",
            },
        )

    # ------------------------------------------------------- outcomes ledger

    @property
    def outcomes_path(self):
        return self.store.store_dir / f"{self.store.namespace}_outcomes.jsonl"

    def log_outcome(self, task_id: str, month: str, predicted: str, judge_result: str) -> None:
        """Append one judged prediction to the calibration ledger (idempotent per task)."""
        if judge_result not in ("CORRECT", "INCORRECT") or predicted not in ("跑赢", "跑输"):
            return
        with self.store._locked():
            seen: set[str] = set()
            if self.outcomes_path.exists():
                with open(self.outcomes_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            seen.add(json.loads(line).get("task_id", ""))
            if task_id in seen:
                return
            with open(self.outcomes_path, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "task_id": task_id,
                            "month": month,
                            "predicted": predicted,
                            "judge_result": judge_result,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    def calibration_block(self, before_month: str, min_samples: int = 16) -> str:
        """Self-calibration stats over past months' judged predictions.

        The agent's own bias is the one large-sample, measurable signal in past
        outcomes (unlike n=1 episodic lessons). Direction-neutral by design: it
        reports the agent's error profile, never any stock's future direction.
        """
        if not before_month or not self.outcomes_path.exists():
            return ""
        rows: list[dict[str, str]] = []
        with open(self.outcomes_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("month") and r["month"] < before_month:
                    rows.append(r)
        if len(rows) < min_samples:
            return ""

        n = len(rows)
        pred_counts = Counter(r["predicted"] for r in rows)
        hits = Counter()
        for r in rows:
            if r["judge_result"] == "CORRECT":
                hits[r["predicted"]] += 1
        # Realized label is recoverable from (prediction, judge): binary task.
        label_counts: Counter = Counter()
        for r in rows:
            if r["judge_result"] == "CORRECT":
                label_counts[r["predicted"]] += 1
            else:
                label_counts["跑赢" if r["predicted"] == "跑输" else "跑输"] += 1

        def pct(x: int, d: int) -> str:
            return f"{x / d * 100:.0f}%" if d else "-"

        lines = [
            f"### 自我校准统计（基于你此前 {n} 次已判题预测，仅含 {before_month} 之前的月份）",
            f"- 你的预测分布：跑输 {pred_counts['跑输']} 次（{pct(pred_counts['跑输'], n)}）、"
            f"跑赢 {pred_counts['跑赢']} 次（{pct(pred_counts['跑赢'], n)}）",
            f"- 分方向命中率：预测「跑输」命中 {pct(hits['跑输'], pred_counts['跑输'])}"
            f"（{hits['跑输']}/{pred_counts['跑输']}）、"
            f"预测「跑赢」命中 {pct(hits['跑赢'], pred_counts['跑赢'])}"
            f"（{hits['跑赢']}/{pred_counts['跑赢']}）",
            f"- 同期实际标签分布：跑输 {pct(label_counts['跑输'], n)}、跑赢 {pct(label_counts['跑赢'], n)}",
            "- 校准提示：若你的预测分布明显偏离实际标签分布、且占多数方向的命中率并不更高，"
            "说明存在系统性方向偏置。请仅基于当前任务自身证据独立判断，不要默认沿用多数方向。",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------ search path

    def search(
        self,
        query: str,
        top_k: int = 3,
        before_month: str = "",
        stance_balance: bool = True,
    ) -> list[tuple[MemoryRecord, float]]:
        def predicate(rec: MemoryRecord) -> bool:
            if not before_month:
                return True
            month = str(rec.metadata.get("entry_month", "") or "")
            # Records with unknown month are hidden under temporal filtering.
            return bool(month) and month < before_month

        try:
            wide = self.store.search(
                query=query, top_k=top_k * 4, predicate=predicate, min_score=0.05
            )
        except Exception as exc:
            print(f"    memory search failed (no injection): {exc}")
            return []

        if not stance_balance or len(wide) <= top_k:
            return wide[:top_k]

        # Tight quota: at most ~1/3 of slots per directional stance (1 bullish
        # + 1 bearish at top_k=3). ceil(top_k/2) still let 2-vs-1 tilts through,
        # which compounded over months in the mem0 v1 run.
        max_per_stance = max(1, top_k // 3)
        picked: list[tuple[MemoryRecord, float]] = []
        counts: Counter = Counter()
        deferred: list[tuple[MemoryRecord, float]] = []
        for rec, score in wide:
            if len(picked) >= top_k:
                break
            stance = rec.metadata.get("functional_stance", "neutral")
            if stance in ("bullish", "bearish") and counts[stance] >= max_per_stance:
                deferred.append((rec, score))
                continue
            picked.append((rec, score))
            counts[stance] += 1
        for rec, score in deferred:
            if len(picked) >= top_k:
                break
            picked.append((rec, score))
        return picked

    @staticmethod
    def format_results(results: list[tuple[MemoryRecord, float]]) -> str:
        if not results:
            return "No relevant memories found."
        lines = []
        for i, (rec, score) in enumerate(results, 1):
            tags = rec.metadata.get("tags", [])
            tag_str = ", ".join(tags) if tags else "none"
            month = rec.metadata.get("entry_month", "") or "?"
            lines.append(
                f"{i}. [score={score:.3f}|来源月={month}|tags={tag_str}] {rec.content}"
            )
        return "\n".join(lines)
