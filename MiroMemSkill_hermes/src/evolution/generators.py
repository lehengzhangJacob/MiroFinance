# SPDX-FileCopyrightText: 2026 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

"""Candidate generators: propose skill-body rewrites, nothing else.

A generator receives the baseline artifact plus structured settled-outcome
feedback and returns candidate BODY texts. It never runs arms, never scores,
never touches the registry or the production skill file. GEPA-style external
optimizers plug in behind the same protocol.
"""

from __future__ import annotations

import os
import re
import time
from typing import Protocol

import requests

from src.evolution.types import SkillArtifact

SKILL_OPEN = "<skill_body>"
SKILL_CLOSE = "</skill_body>"

SYSTEM_PROMPT = (
    "你是一名量化交易流程工程师。你的任务是根据已结算的回测事实反馈，改写一份"
    "A股月度组合构建 Skill 的正文，使其流程更能避免反馈中暴露的亏损模式。\n"
    "硬性约束：\n"
    "1. 只输出新的 Skill 正文 markdown，放在 "
    f"{SKILL_OPEN} 与 {SKILL_CLOSE} 标签之间；标签外不要有任何其他输出。\n"
    "2. 不得写入任何具体股票代码（如 600519.SH）、具体日期或对未来行情的断言。\n"
    "3. 不得引入反馈中不存在的新工具名；步骤必须使用现有工具。\n"
    "4. 保持步骤编号清晰、可执行；总长度不超过原正文的约 1.5 倍。\n"
    "5. 改动要针对反馈中的失败模式，而不是泛泛地追加空话。"
)

USER_TEMPLATE = """## 当前 Skill 正文

{body}

## 已结算的事实反馈

{feedback}

请输出改写后的完整 Skill 正文（只含正文，不含 frontmatter）。"""


class CandidateGenerator(Protocol):
    name: str

    def propose(
        self, baseline: SkillArtifact, feedback: str, n: int = 1
    ) -> list[str]:
        """Return up to ``n`` candidate body texts."""
        ...


class ReflectiveMutationGenerator:
    """GEPA-inspired reflective mutation via direct GLM chat completions."""

    name = "reflective_mutation_glm"

    def __init__(
        self,
        model: str = "glm-5.2",
        temperature: float = 0.9,
        max_tokens: int = 8000,
        timeout: int = 600,
        max_retries: int = 2,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.api_key = os.getenv("GLM_API_KEY", "")
        self.base_url = os.getenv(
            "GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"
        )
        if not self.api_key:
            raise RuntimeError("GLM_API_KEY not set; cannot propose candidates")

    # ------------------------------------------------------------ helpers

    def _chat(self, messages: list[dict], disable_thinking: bool) -> str:
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if disable_thinking:
            # Reasoning traces waste budget for pure text rewriting and have
            # produced empty visible bodies on some GLM endpoints.
            payload["thinking"] = {"type": "disabled"}
        resp = requests.post(
            f"{self.base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"] or ""

    @staticmethod
    def _extract_body(raw: str) -> str | None:
        match = re.search(
            re.escape(SKILL_OPEN) + r"(.*?)" + re.escape(SKILL_CLOSE),
            raw,
            re.DOTALL,
        )
        body = (match.group(1) if match else raw).strip()
        # Strip an accidental markdown fence wrapper.
        fence = re.fullmatch(r"```(?:markdown)?\s*\n(.*?)\n```", body, re.DOTALL)
        if fence:
            body = fence.group(1).strip()
        # A body that still carries frontmatter or tags is malformed.
        if not body or body.startswith("---") or SKILL_OPEN in body:
            return None
        return body

    def _mutate_once(self, baseline: SkillArtifact, feedback: str) -> str | None:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_TEMPLATE.format(
                    body=baseline.body, feedback=feedback
                ),
            },
        ]
        disable_thinking = True
        for attempt in range(self.max_retries + 1):
            try:
                raw = self._chat(messages, disable_thinking=disable_thinking)
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else 0
                if 400 <= status < 500 and disable_thinking:
                    # Endpoint may reject the thinking switch; retry without it.
                    disable_thinking = False
                    continue
                if attempt >= self.max_retries:
                    raise
                time.sleep(5 * (attempt + 1))
                continue
            except requests.RequestException:
                if attempt >= self.max_retries:
                    raise
                time.sleep(5 * (attempt + 1))
                continue
            body = self._extract_body(raw)
            if body:
                return body
        return None

    # ------------------------------------------------------------- public

    def propose(
        self, baseline: SkillArtifact, feedback: str, n: int = 1
    ) -> list[str]:
        bodies: list[str] = []
        seen: set[str] = {baseline.body.strip()}
        attempts = 0
        while len(bodies) < n and attempts < n * 3:
            attempts += 1
            body = self._mutate_once(baseline, feedback)
            if body and body.strip() not in seen:
                seen.add(body.strip())
                bodies.append(body)
        return bodies


class HermesGEPAGenerator:
    """Placeholder adapter for the external hermes-agent-self-evolution GEPA.

    Intentionally unimplemented: the current external checkout depends on
    uncommitted local patches and a proxy fitness. Once that repo is pinned,
    implement ``propose`` as a subprocess call that maps its evolved output
    back to a body text. The controller only depends on the protocol.
    """

    name = "hermes_gepa"

    def propose(
        self, baseline: SkillArtifact, feedback: str, n: int = 1
    ) -> list[str]:
        raise NotImplementedError(
            "HermesGEPAGenerator is a stub; use ReflectiveMutationGenerator"
        )
