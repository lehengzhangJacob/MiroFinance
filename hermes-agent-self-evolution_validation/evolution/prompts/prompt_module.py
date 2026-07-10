"""Wrap a prompt section as a DSPy module for GEPA optimization."""

from pathlib import Path
from typing import Optional

import dspy


def load_prompt_section(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8")
    return {"path": path, "raw": raw, "body": raw}


def find_prompt_section(name: str, hermes_agent_path: Path) -> Optional[Path]:
    candidate = hermes_agent_path / "prompts" / f"{name}.md"
    if candidate.exists():
        return candidate
    return None


class PromptSectionModule(dspy.Module):
    class RespondWithGuidelines(dspy.Signature):
        """Respond to a coding scenario following the system guidelines."""
        guidelines: str = dspy.InputField(desc="System prompt section with agent rules")
        task_input: str = dspy.InputField(desc="Scenario or user request")
        output: str = dspy.OutputField(desc="How the agent should behave/respond")

    def __init__(self, prompt_text: str):
        super().__init__()
        self.prompt_text = prompt_text
        self.predictor = dspy.ChainOfThought(self.RespondWithGuidelines)

    def forward(self, task_input: str) -> dspy.Prediction:
        result = self.predictor(guidelines=self.prompt_text, task_input=task_input)
        return dspy.Prediction(output=result.output, prompt_text=self.prompt_text)
