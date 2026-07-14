# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import dotenv

_MIROFLOW_DIR = Path(__file__).resolve().parents[2]
_AGENT_DIR = _MIROFLOW_DIR.parent


def load_project_env() -> None:
    """Load LLM keys from agent/llm_key, then tool/proxy settings from MiroFlow/.env."""
    dotenv.load_dotenv(_AGENT_DIR / "llm_key")
    dotenv.load_dotenv(_MIROFLOW_DIR / ".env")
