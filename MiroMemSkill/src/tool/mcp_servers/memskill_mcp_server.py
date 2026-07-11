# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

"""MCP server exposing memory and skill tools for agent reasoning."""

import os

from fastmcp import FastMCP

from src.logging.logger import setup_mcp_logging
from src.memory.memory import Mem0Memory
from src.memory.skills import SkillLibrary
from src.memory.vector_store import VectorStore

setup_mcp_logging(tool_name=os.path.basename(__file__))
mcp = FastMCP("memskill-mcp-server")

_STORE_DIR = os.environ.get("MEMSKILL_STORE_DIR", "memory_bank")
_NAMESPACE = os.environ.get("MEMSKILL_NAMESPACE", "default")
_SKILLS_DIR = os.environ.get("MEMSKILL_SKILLS_DIR", f"{_STORE_DIR}/skills")
_ALLOW_SAVE = os.environ.get("MEMSKILL_ALLOW_SAVE", "true").lower() in {
    "1", "true", "yes", "on"
}

_memory: Mem0Memory | None = None
_skills: SkillLibrary | None = None


def _get_memory() -> Mem0Memory:
    global _memory
    if _memory is None:
        _memory = Mem0Memory(store=VectorStore(store_dir=_STORE_DIR, namespace=_NAMESPACE))
    return _memory


def _get_skills() -> SkillLibrary:
    global _skills
    if _skills is None:
        _skills = SkillLibrary(skills_dir=_SKILLS_DIR)
    return _skills


@mcp.tool()
def memory_search(
    query: str,
    top_k: int = 5,
    before_month: str = "",
    before_date: str = "",
) -> str:
    """Search stored past experiences and notes for relevant lessons.

    Args:
        query: Natural language search query.
        top_k: Maximum number of results to return (default 5).
        before_month: Point-in-time guard, format YYYY-MM. For prediction tasks
            ALWAYS pass the task's current month so only lessons learned from
            strictly earlier market months are returned (avoids look-ahead).
        before_date: Preferred exact point-in-time guard, format YYYYMMDD. Pass
            the task's as-of date so 20-day labels are visible only after their
            actual exit date.
    """
    try:
        memory = _get_memory()
        results = memory.search(
            query,
            top_k=top_k,
            before_month=before_month,
            before_date=before_date,
        )
        return memory.format_results(results)
    except Exception as exc:
        return f"Memory search unavailable: {exc}"


@mcp.tool()
def memory_save(content: str, tags: str = "", as_of_month: str = "") -> str:
    """Save a reusable note or heuristic to memory for future tasks.

    Args:
        content: The note content (strategy/heuristic/source — not full answers).
        tags: Comma-separated tags, e.g. "momentum,valuation".
        as_of_month: The market month (YYYY-MM) this note is based on. Required
            for the note to be retrievable in point-in-time filtered searches.
    """
    try:
        if not _ALLOW_SAVE:
            return "Memory save disabled: this namespace accepts validated rolling rules only."
        memory = _get_memory()
        tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
        rec = memory.save_note(content, tags=tag_list, as_of_month=as_of_month)
        return f"Saved memory id={rec.id} tags={tag_list} as_of_month={as_of_month or 'unset'}"
    except Exception as exc:
        return f"Memory save failed: {exc}"


@mcp.tool()
def skill_list() -> str:
    """List all available procedural skills in the skill library."""
    skills = _get_skills().list_skills()
    if not skills:
        return "No skills available."
    lines = []
    for s in skills:
        lines.append(f"- **{s['name']}**: {s['description']} (triggers: {s['triggers']})")
    return "\n".join(lines)


@mcp.tool()
def skill_load(name: str) -> str:
    """Load the full procedural steps for a named skill.

    Args:
        name: Skill name as returned by skill_list.
    """
    return _get_skills().load_skill_text(name)


if __name__ == "__main__":
    mcp.run()
