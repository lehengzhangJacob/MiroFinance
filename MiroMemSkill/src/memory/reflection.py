# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

"""Post-task reflection: distill strategy lessons into episodic memory."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Optional

import requests
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from src.memory.store import MemoryStore, stance_of
from src.utils.env_loader import load_project_env


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, requests.exceptions.HTTPError):
        status = exc.response.status_code if exc.response is not None else 0
        return status == 429 or status >= 500
    return isinstance(exc, (requests.exceptions.ConnectionError, requests.exceptions.Timeout))


_PROMPT_HEADER = """You are a meta-learning assistant for a financial research agent.

{outcome_instructions}

Hard requirements:
- Be CONCRETE: name the exact indicator / tool / threshold / step involved
  (e.g. "when 20d and 60d relative momentum disagree, trust the 20d window only if
  turnover is also rising"), never generic advice like "consider multiple factors".
- Phrase lessons as conditional tendencies, NOT guaranteed rules — a single task outcome
  is noisy evidence, especially for market-direction predictions.
- Do NOT restate the task protocol (point-in-time rules, boxed output format).
- Do NOT include the ground-truth answer or specific numeric result of this task.
- Do NOT repeat or trivially rephrase anything under "Already stored lessons".
- If this task teaches nothing genuinely new, return {{"lessons": []}}.

Task question (truncated):
{question}

Agent final answer:
{answer}

Judge result: {judge_result}

Trajectory summary (truncated):
{trajectory}

Already stored lessons (do NOT repeat these):
{existing}

Return ONLY valid JSON:
{{
  "lessons": [
    {{"content": "...", "tags": ["strategy", "..."]}}
  ]
}}
"""

_CORRECT_INSTRUCTIONS = """Given a completed task the agent answered CORRECTLY, distill AT MOST 2
reusable STRATEGY lessons (NOT the task's final answer). Focus on decision procedures: which
evidence to weight under what conditions, tool-usage tactics, calculation pitfalls,
output-format traps."""

_INCORRECT_INSTRUCTIONS = """The agent's reasoning below led to a WRONG prediction. Distill AT MOST 1
COUNTER-LESSON describing the failure mode: under what conditions the heuristic the agent relied on
breaks down, and what disconfirming evidence it should have checked.

CRITICAL: do NOT restate the agent's failed reasoning as a positive rule (e.g. if the agent
predicted 跑输 based on weak momentum and was wrong, the lesson must NOT be "weak momentum implies
跑输" — it must be "weak momentum alone is unreliable for predicting 跑输 when <specific
counter-condition observed here>"). The lesson must encode WHY this line of reasoning failed."""


class MemoryReflector:
    """Reflect on task outcome and write lessons to memory store."""

    def __init__(
        self,
        store: MemoryStore,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        load_project_env()
        self.store = store
        # Resolution order: explicit REFLECTION_LLM_* envs (set per-run, e.g.
        # moonshot-v1-32k for Kimi runs) > DeepSeek > GLM fallback.
        self.model = model or os.getenv("REFLECTION_LLM_MODEL_NAME") or os.getenv(
            "DEEPSEEK_MODEL_NAME"
        ) or os.getenv("GLM_MODEL_NAME", "deepseek-v4-pro")
        self.base_url = base_url or os.getenv("REFLECTION_LLM_BASE_URL") or os.getenv(
            "DEEPSEEK_BASE_URL"
        ) or os.getenv("GLM_BASE_URL", "https://api.deepseek.com/v1")
        self.api_key = api_key or os.getenv("REFLECTION_LLM_API_KEY") or os.getenv(
            "DEEPSEEK_API_KEY"
        ) or os.getenv("GLM_API_KEY", "")

    def _summarize_trajectory(self, log_data: dict[str, Any], max_chars: int = 3000) -> str:
        parts: list[str] = []
        history = log_data.get("main_agent_message_history", {}).get("message_history", [])
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
        summary = "\n".join(parts)
        return summary[:max_chars]

    @retry(
        retry=retry_if_exception(_is_retryable),
        wait=wait_exponential(multiplier=5, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _call_llm(self, prompt: str) -> str:
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        resp = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "max_tokens": 1024,
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    @staticmethod
    def _parse_lessons(raw: str) -> list[dict[str, Any]]:
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        try:
            data = json.loads(text)
            return data.get("lessons", [])
        except json.JSONDecodeError:
            return [{"content": text[:500], "tags": ["reflection"]}]

    @staticmethod
    def _extract_direction(answer: str) -> str:
        """Pull the predicted direction (跑赢/跑输) out of the final answer."""
        text = answer or ""
        if "跑赢" in text and "跑输" not in text:
            return "跑赢"
        if "跑输" in text and "跑赢" not in text:
            return "跑输"
        m = re.search(r"\\boxed\{(跑赢|跑输)\}", text)
        return m.group(1) if m else ""

    def reflect_and_store(
        self,
        question: str,
        answer: str,
        judge_result: str,
        log_data: Optional[dict[str, Any]] = None,
        task_id: str = "",
    ) -> list[str]:
        if not self.api_key:
            return []

        trajectory = self._summarize_trajectory(log_data or {})
        # Show the reflector what similar lessons already exist so it can
        # return novel ones (or an empty list) instead of re-deriving the
        # same generic advice on every task.
        try:
            similar = self.store.search(question[:300], top_k=3)
            existing_block = "\n".join(f"- {e.content[:200]}" for e, _ in similar) or "None"
        except Exception:
            existing_block = "None"

        # INCORRECT tasks must yield counter-lessons (why the heuristic
        # failed), never a restatement of the losing logic as a rule —
        # restated failures were the main driver of pool3's bearish collapse.
        is_incorrect = judge_result == "INCORRECT"
        outcome_instructions = _INCORRECT_INSTRUCTIONS if is_incorrect else _CORRECT_INSTRUCTIONS
        max_lessons = 1 if is_incorrect else 2

        prompt = _PROMPT_HEADER.format(
            outcome_instructions=outcome_instructions,
            question=question[:800],
            answer=(answer or "N/A")[:400],
            judge_result=judge_result,
            trajectory=trajectory or "N/A",
            existing=existing_block,
        )

        try:
            raw = self._call_llm(prompt)
            lessons = self._parse_lessons(raw)
        except Exception as exc:
            # Do NOT store error placeholders: they pollute retrieval.
            print(f"    Reflection LLM call failed after retries (skipped): {exc}")
            return []

        predicted_direction = self._extract_direction(answer)
        stored: list[str] = []
        for lesson in lessons[:max_lessons]:
            content = lesson.get("content", "").strip()
            if not content:
                continue
            if self.store.find_near_duplicate(content) is not None:
                continue  # write-side dedup: keep the memory bank non-redundant
            tags = lesson.get("tags", ["reflection", judge_result.lower()])
            self.store.add(
                content=content,
                kind="episodic",
                tags=tags,
                metadata={
                    "task_id": task_id,
                    "judge_result": judge_result,
                    "source": "reflection",
                    "predicted_direction": predicted_direction,
                    "stance": stance_of(content),
                },
                dedupe=False,  # already checked above
            )
            stored.append(content)
        return stored
