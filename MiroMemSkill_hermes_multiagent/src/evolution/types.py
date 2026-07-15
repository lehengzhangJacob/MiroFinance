# SPDX-FileCopyrightText: 2026 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

"""Immutable value types shared by the skill-evolution control plane."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_skill_text(text: str) -> tuple[str, str]:
    """Split a skill markdown file into (raw_frontmatter, body).

    Raises ValueError when the file has no parseable frontmatter block,
    because such a file would be silently ignored by SkillLibrary.
    """
    match = FRONTMATTER_RE.match(text)
    if not match:
        raise ValueError("skill text has no YAML frontmatter block")
    return match.group(1), match.group(2).strip()


@dataclass(frozen=True)
class SkillArtifact:
    """A content-addressed snapshot of one skill markdown file."""

    name: str
    digest: str
    frontmatter: str
    body: str
    parent_digest: str | None = None

    @property
    def short_id(self) -> str:
        return self.digest[:12]

    @property
    def text(self) -> str:
        return f"---\n{self.frontmatter}\n---\n\n{self.body}\n"

    @classmethod
    def from_text(
        cls, text: str, parent_digest: str | None = None
    ) -> "SkillArtifact":
        frontmatter, body = split_skill_text(text)
        name_match = re.search(
            r"^name:\s*[\"']?([\w.-]+)[\"']?\s*$", frontmatter, re.MULTILINE
        )
        if not name_match:
            raise ValueError("skill frontmatter has no name field")
        canonical = f"---\n{frontmatter}\n---\n\n{body}\n"
        return cls(
            name=name_match.group(1),
            digest=sha256_text(canonical),
            frontmatter=frontmatter,
            body=body,
            parent_digest=parent_digest,
        )


@dataclass
class GateResult:
    passed: bool
    failures: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)
