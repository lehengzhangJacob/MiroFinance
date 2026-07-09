# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

"""Skill library: markdown files with YAML frontmatter."""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import requests
import yaml

from src.utils.env_loader import load_project_env


@dataclass
class Skill:
    name: str
    description: str
    triggers: list[str] = field(default_factory=list)
    body: str = ""
    path: str = ""
    embedding: Optional[list[float]] = None

    @property
    def search_text(self) -> str:
        return f"{self.name} {self.description} {' '.join(self.triggers)} {self.body[:500]}"


class SkillLibrary:
    """Load and match procedural skills from memory_bank/skills/*.md."""

    def __init__(self, skills_dir: str | Path, embedding_model: str = "embedding-3"):
        load_project_env()
        self.skills_dir = Path(skills_dir)
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_model = embedding_model
        self._skills: list[Skill] = []
        embedding_switch = os.getenv("MEMSKILL_EMBEDDING_ENABLED", "true").lower() != "false"
        self._api_key = os.getenv("GLM_API_KEY", "") if embedding_switch else ""
        self._base_url = os.getenv("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
        self.reload()

    def reload(self) -> None:
        self._skills = []
        # Flat skills (skills_dir/foo.md) plus packaged skills that ship
        # scripts/config alongside a SKILL.md (skills_dir/foo/SKILL.md).
        paths = sorted(self.skills_dir.glob("*.md")) + sorted(
            self.skills_dir.glob("*/SKILL.md")
        )
        for path in paths:
            skill = self._parse_skill_file(path)
            if skill:
                self._skills.append(skill)

    @staticmethod
    def _parse_skill_file(path: Path) -> Optional[Skill]:
        text = path.read_text(encoding="utf-8")
        match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", text, re.DOTALL)
        if not match:
            return None
        meta = yaml.safe_load(match.group(1)) or {}
        body = match.group(2).strip()
        triggers = meta.get("triggers", [])
        if isinstance(triggers, str):
            triggers = [triggers]
        return Skill(
            name=str(meta.get("name", path.stem)),
            description=str(meta.get("description", "")),
            triggers=[str(t) for t in triggers],
            body=body,
            path=str(path),
        )

    def list_skills(self) -> list[dict[str, str]]:
        return [
            {
                "name": s.name,
                "description": s.description,
                "triggers": ", ".join(s.triggers),
            }
            for s in self._skills
        ]

    def get_skill(self, name: str) -> Optional[Skill]:
        name_lower = name.lower()
        for skill in self._skills:
            path = Path(skill.path)
            # For packaged skills the stem is always "SKILL"; the directory
            # name is the meaningful identifier.
            file_key = path.parent.name if path.stem == "SKILL" else path.stem
            if skill.name.lower() == name_lower or file_key.lower() == name_lower:
                return skill
        return None

    def load_skill_text(self, name: str) -> str:
        skill = self.get_skill(name)
        if not skill:
            return f"Skill '{name}' not found. Available: {[s.name for s in self._skills]}"
        return (
            f"# Skill: {skill.name}\n\n"
            f"**Description**: {skill.description}\n\n"
            f"**Triggers**: {', '.join(skill.triggers)}\n\n"
            f"{skill.body}"
        )

    def _get_embedding(self, text: str) -> list[float]:
        url = f"{self._base_url.rstrip('/')}/embeddings"
        resp = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={"model": self.embedding_model, "input": text[:8000]},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def _keyword_score(self, query: str, skill: Skill) -> float:
        q = query.lower()
        score = 0.0
        for trigger in skill.triggers:
            if trigger.lower() in q:
                score += 2.0
        for token in re.findall(r"[\w\u4e00-\u9fff]+", skill.description.lower()):
            if len(token) > 2 and token in q:
                score += 0.5
        return score

    def match(self, query: str, top_k: int = 3) -> list[tuple[Skill, float]]:
        if not self._skills:
            return []

        scored: list[tuple[Skill, float]] = []
        if self._api_key:
            try:
                q_emb = self._get_embedding(query)
                for skill in self._skills:
                    if skill.embedding is None:
                        try:
                            skill.embedding = self._get_embedding(skill.search_text)
                        except Exception:
                            skill.embedding = None
                    emb_score = (
                        self._cosine(q_emb, skill.embedding) if skill.embedding else 0.0
                    )
                    kw_score = self._keyword_score(query, skill)
                    scored.append((skill, emb_score + kw_score * 0.1))
            except Exception:
                scored = [(s, self._keyword_score(query, s)) for s in self._skills]
        else:
            scored = [(s, self._keyword_score(query, s)) for s in self._skills]

        scored.sort(key=lambda x: x[1], reverse=True)
        return [(s, sc) for s, sc in scored[:top_k] if sc > 0]

    def format_matches(self, matches: list[tuple[Skill, float]]) -> str:
        if not matches:
            return "No matching skills found."
        lines = []
        for i, (skill, score) in enumerate(matches, 1):
            lines.append(
                f"{i}. **{skill.name}** (score={score:.3f}): {skill.description}"
            )
        return "\n".join(lines)
