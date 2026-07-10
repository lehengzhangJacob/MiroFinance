"""Wrap Python source code as a DSPy module for GEPA optimization."""

import tempfile
from pathlib import Path
from typing import Optional

import dspy

from evolution.core.fitness import code_pytest_fitness


def load_code_file(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8")
    return {"path": path, "raw": raw}


def find_code_target(name: str, hermes_agent_path: Path) -> Optional[Path]:
    candidate = hermes_agent_path / "code" / f"{name}.py"
    if candidate.exists():
        return candidate
    return None


class CodeModule(dspy.Module):
    """Code module — forward runs pytest against the candidate source."""

    def __init__(self, code_text: str, test_path: Path, repo_root: Path):
        super().__init__()
        self.code_text = code_text
        self.test_path = test_path
        self.repo_root = repo_root

    def forward(self, task_input: str = "") -> dspy.Prediction:
        code_dir = self.repo_root / "hermes-agent" / "code"
        code_dir.mkdir(parents=True, exist_ok=True)
        target = code_dir / "text_helpers.py"
        target.write_text(self.code_text, encoding="utf-8")
        score, feedback = code_pytest_fitness(target, self.test_path, self.repo_root)
        return dspy.Prediction(
            output=self.code_text,
            pytest_score=score,
            pytest_feedback=feedback,
        )
