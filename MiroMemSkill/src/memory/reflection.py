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

from src.memory.store import MemoryStore
from src.utils.env_loader import load_project_env


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, requests.exceptions.HTTPError):
        status = exc.response.status_code if exc.response is not None else 0
        return status == 429 or status >= 500
    return isinstance(exc, (requests.exceptions.ConnectionError, requests.exceptions.Timeout))


REFLECTION_PROMPT = """You are a meta-learning assistant for a financial research agent.

Given a completed task, distill 1-2 reusable STRATEGY lessons (NOT the task's final answer).
Focus on: search strategy, source selection, calculation approach, common pitfalls, formatting rules.
Do NOT include the ground-truth answer or specific numeric result of this task.

Task question (truncated):
{question}

Agent final answer:
{answer}

Judge result: {judge_result}

Trajectory summary (truncated):
{trajectory}

Return ONLY valid JSON:
{{
  "lessons": [
    {{"content": "...", "tags": ["strategy", "..."]}}
  ]
}}
"""


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
        # Primary: DeepSeek (GLM balance exhausted 2026-07-07); GLM as fallback.
        self.model = model or os.getenv("DEEPSEEK_MODEL_NAME") or os.getenv(
            "GLM_MODEL_NAME", "deepseek-v4-pro"
        )
        self.base_url = base_url or os.getenv("DEEPSEEK_BASE_URL") or os.getenv(
            "GLM_BASE_URL", "https://api.deepseek.com/v1"
        )
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv(
            "GLM_API_KEY", ""
        )

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
        prompt = REFLECTION_PROMPT.format(
            question=question[:800],
            answer=(answer or "N/A")[:400],
            judge_result=judge_result,
            trajectory=trajectory or "N/A",
        )

        try:
            raw = self._call_llm(prompt)
            lessons = self._parse_lessons(raw)
        except Exception as exc:
            # Do NOT store error placeholders: they pollute retrieval.
            print(f"    Reflection LLM call failed after retries (skipped): {exc}")
            return []

        stored: list[str] = []
        for lesson in lessons[:2]:
            content = lesson.get("content", "").strip()
            if not content:
                continue
            tags = lesson.get("tags", ["reflection", judge_result.lower()])
            self.store.add(
                content=content,
                kind="episodic",
                tags=tags,
                metadata={
                    "task_id": task_id,
                    "judge_result": judge_result,
                    "source": "reflection",
                },
            )
            stored.append(content)
        return stored
