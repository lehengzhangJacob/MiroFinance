"""Wrap tool descriptions as a DSPy module for GEPA optimization."""

import json
from pathlib import Path
from typing import Optional

import dspy


def load_tool_registry(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data


def format_tool_catalog(tools: list[dict]) -> str:
    lines = ["Available tools:"]
    for t in tools:
        lines.append(f"- {t['name']}: {t['description']}")
    return "\n".join(lines)


def parse_tool_catalog(text: str) -> list[dict]:
    tools = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("- ") and ": " in line:
            name_part, desc = line[2:].split(": ", 1)
            tools.append({"name": name_part.strip(), "description": desc.strip()})
    return tools


def find_tool_registry(tool_set: str, hermes_agent_path: Path) -> Optional[Path]:
    candidate = hermes_agent_path / "tools" / f"{tool_set}_tools.json"
    if candidate.exists():
        return candidate
    return None


class ToolDescModule(dspy.Module):
    class SelectTool(dspy.Signature):
        """Choose the single best tool for the user's request based on tool descriptions.
        Output ONLY the tool name, nothing else."""
        tool_catalog: str = dspy.InputField(desc="Tool names and descriptions")
        task_input: str = dspy.InputField(desc="User request")
        selected_tool: str = dspy.OutputField(desc="Exact tool name e.g. read_git_diff")

    def __init__(self, catalog_text: str):
        super().__init__()
        self.catalog_text = catalog_text
        self.predictor = dspy.ChainOfThought(self.SelectTool)

    def forward(self, task_input: str) -> dspy.Prediction:
        result = self.predictor(tool_catalog=self.catalog_text, task_input=task_input)
        return dspy.Prediction(
            selected_tool=result.selected_tool,
            output=result.selected_tool,
        )
