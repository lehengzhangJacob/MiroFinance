# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

"""Build pre-task memory/skill context block for orchestrator injection."""

from __future__ import annotations

from omegaconf import DictConfig

from src.memory.skills import SkillLibrary
from src.memory.store import MemoryStore

# Benchmark task prompts share large boilerplate sections (data-usage rules,
# output format). Cutting at these markers keys retrieval on the task-specific
# head (stock, industry, date) instead of template text shared by every task.
_QUERY_CUT_MARKERS = ["数据使用规则", "输出要求", "Data usage rules"]


def compact_task_query(task_description: str, max_chars: int = 400) -> str:
    text = task_description
    for marker in _QUERY_CUT_MARKERS:
        idx = text.find(marker)
        if idx > 0:
            text = text[:idx]
            break
    return text.strip()[:max_chars]


def get_memory_components(cfg: DictConfig) -> tuple[MemoryStore | None, SkillLibrary | None]:
    if not cfg.get("memory") or not cfg.memory.get("enabled", False):
        return None, None

    store_dir = cfg.memory.get("store_dir", "memory_bank")
    namespace = cfg.memory.get("namespace", "default")
    skills_dir = cfg.memory.get("skills_dir", f"{store_dir}/skills")

    store = MemoryStore(
        store_dir=store_dir,
        namespace=namespace,
        dedupe_threshold=float(cfg.memory.get("dedupe_threshold", 0.8)),
    )
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
    skill_preview_min_score: float = 0.0,
) -> str:
    sections: list[str] = []
    query = compact_task_query(task_description)

    if memory_enabled:
        # Stance-balanced retrieval: never let the injected experiences stack
        # one direction, or they act as a prior and the agent's predictions
        # collapse toward the bank's majority stance.
        results = store.search_balanced(query, top_k=inject_top_k)
        if results:
            sections.append(
                "### Relevant Past Experiences\n"
                "(Conditional heuristics from past tasks — NOT direction priors. "
                "Evaluate the current task's own evidence first; your final call must "
                "be supported by the data you retrieve now, not by these notes.)\n"
                + store.format_search_results(results)
            )

    if skill_enabled and skill_lib:
        matches = skill_lib.match(query, top_k=skill_top_k)
        if matches:
            sections.append("### Recommended Skills\n" + skill_lib.format_matches(matches))
            top_skill, top_score = matches[0]
            # Only paste the full skill body when the match is confident
            # (>= one trigger hit under keyword scoring); otherwise the agent
            # can still pull it explicitly via skill_load.
            if top_score >= skill_preview_min_score:
                sections.append(
                    "### Top Skill Preview\n"
                    + skill_lib.load_skill_text(top_skill.name)[:1500]
                )
        # Keyword matching misses skills whose triggers don't appear in the
        # task text (e.g. qlib/tushare utility skills), leaving them invisible
        # and unused. Always list the rest so the agent can skill_load them.
        matched_names = {s.name for s, _ in matches}
        others = [s for s in skill_lib.list_skills() if s["name"] not in matched_names]
        if others:
            sections.append(
                "### Other Available Skills (load with skill_load(name))\n"
                + "\n".join(f"- **{s['name']}**: {s['description']}" for s in others)
            )

    if not sections:
        return ""

    return (
        "\n\n## Relevant Experience & Skills (from memory bank)\n\n"
        "Use the following retrieved experiences and skills to guide your approach. "
        "You may also call memory_search / skill_load tools during reasoning for more detail.\n\n"
        + "\n\n".join(sections)
    )
