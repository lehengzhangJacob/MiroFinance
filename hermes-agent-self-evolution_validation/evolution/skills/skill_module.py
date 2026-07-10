"""Wraps a SKILL.md file as a DSPy module for optimization.

The skill body is stored in the predictor signature *instructions* so GEPA
can mutate it directly (not only the static input field).
"""

import re
from pathlib import Path
from typing import Optional

import dspy


def load_skill(skill_path: Path) -> dict:
    raw = skill_path.read_text()

    frontmatter = ""
    body = raw
    if raw.strip().startswith("---"):
        parts = raw.split("---", 2)
        if len(parts) >= 3:
            frontmatter = parts[1].strip()
            body = parts[2].strip()

    name = ""
    description = ""
    for line in frontmatter.split("\n"):
        if line.strip().startswith("name:"):
            name = line.split(":", 1)[1].strip().strip("'\"")
        elif line.strip().startswith("description:"):
            description = line.split(":", 1)[1].strip().strip("'\"")

    return {
        "path": skill_path,
        "raw": raw,
        "frontmatter": frontmatter,
        "body": body,
        "name": name,
        "description": description,
    }


def find_skill(skill_name: str, hermes_agent_path: Path) -> Optional[Path]:
    skills_dir = hermes_agent_path / "skills"
    if not skills_dir.exists():
        return None

    for skill_md in skills_dir.rglob("SKILL.md"):
        if skill_md.parent.name == skill_name:
            return skill_md

    for skill_md in skills_dir.rglob("SKILL.md"):
        try:
            content = skill_md.read_text()[:500]
            if f"name: {skill_name}" in content or f'name: "{skill_name}"' in content:
                return skill_md
        except Exception:
            continue

    return None


def extract_skill_instructions(module: "SkillModule") -> str:
    """Read the current optimizable skill text from the compiled module."""
    try:
        return module.predictor.predict.signature.instructions or module.skill_text
    except Exception:
        return module.skill_text


class SkillModule(dspy.Module):
    """Skill body lives in signature instructions for GEPA mutation."""

    def __init__(self, skill_text: str):
        super().__init__()
        self.skill_text = skill_text
        signature = dspy.Signature(
            "task_input -> output",
            instructions=skill_text,
        )
        self.predictor = dspy.ChainOfThought(signature)

    def forward(self, task_input: str) -> dspy.Prediction:
        result = self.predictor(task_input=task_input)
        # Keep skill_text in sync with instructions (post-GEPA compile)
        self.skill_text = extract_skill_instructions(self)
        return dspy.Prediction(
            output=result.output,
            skill_text=self.skill_text,
        )


def reassemble_skill(frontmatter: str, evolved_body: str) -> str:
    return f"---\n{frontmatter}\n---\n\n{evolved_body}\n"
