# SPDX-FileCopyrightText: 2026 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BASELINE_TEXT = """---
name: demo_skill
description: 演示技能
version: "1.0"
triggers:
  - 全A股
---

## 决策流程

1. 核对 as_of 与约束。
2. 多路筛选建立候选集。
3. 交叉验证后构建组合并确定 CASH。
"""


@pytest.fixture
def baseline_text() -> str:
    return BASELINE_TEXT


@pytest.fixture
def skill_repo(tmp_path: Path) -> Path:
    """A miniature fork root with one production skill file."""
    skills = tmp_path / "memory_bank" / "skills_ashare"
    skills.mkdir(parents=True)
    (skills / "demo_skill.md").write_text(BASELINE_TEXT, encoding="utf-8")
    return tmp_path


def make_tasks(months: list[str]) -> list[dict]:
    tasks = []
    for i, as_of in enumerate(months):
        tasks.append(
            {
                "task_id": f"ashare_open_trader_{as_of}",
                "task_question": f"当前日期为 {as_of}",
                "metadata": {
                    "as_of": as_of,
                    "entry_date": as_of.replace("-", ""),
                    "exit_date": as_of.replace("-", ""),
                    "index_return": 0.01 * ((-1) ** i),
                    "universe": "all_ashare",
                },
            }
        )
    return tasks
