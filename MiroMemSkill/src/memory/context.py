# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

"""Build pre-task memory/skill context block for orchestrator injection."""

from __future__ import annotations

from omegaconf import DictConfig

from src.memory.skills import SkillLibrary
from src.memory.store import MemoryStore


def get_memory_components(cfg: DictConfig) -> tuple[MemoryStore | None, SkillLibrary | None]:
    if not cfg.get("memory") or not cfg.memory.get("enabled", False):
        return None, None

    store_dir = cfg.memory.get("store_dir", "memory_bank")
    namespace = cfg.memory.get("namespace", "default")
    skills_dir = cfg.memory.get("skills_dir", f"{store_dir}/skills")

    store = MemoryStore(store_dir=store_dir, namespace=namespace)
    skill_lib = SkillLibrary(skills_dir=skills_dir) if cfg.memory.get("skill_enabled", True) else None
    return store, skill_lib


def build_memory_context_block(
    task_description: str,
    store: MemoryStore,
    skill_lib: SkillLibrary | None,
    inject_top_k: int = 3,
    skill_top_k: int = 2,
    memory_enabled: bool = True,
    skill_enabled: bool = True,
) -> str:
    sections: list[str] = []

    if memory_enabled:
        results = store.search(task_description, top_k=inject_top_k)
        if results:
            sections.append("### Relevant Past Experiences\n" + store.format_search_results(results))

    if skill_enabled and skill_lib:
        matches = skill_lib.match(task_description, top_k=skill_top_k)
        if matches:
            sections.append("### Recommended Skills\n" + skill_lib.format_matches(matches))
            top_skill = matches[0][0]
            sections.append(
                "### Top Skill Preview\n"
                + skill_lib.load_skill_text(top_skill.name)[:1500]
            )

    if not sections:
        return ""

    return (
        "\n\n## Relevant Experience & Skills (from memory bank)\n\n"
        "Use the following retrieved experiences and skills to guide your approach. "
        "You may also call memory_search / skill_load tools during reasoning for more detail.\n\n"
        + "\n\n".join(sections)
    )
