# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

"""MCP server exposing memory and skill tools for agent reasoning."""

import os
import sys
from pathlib import Path

from fastmcp import FastMCP

from src.logging.logger import setup_mcp_logging
from src.memory.skills import SkillLibrary
from src.memory.store import MemoryStore

setup_mcp_logging(tool_name=os.path.basename(__file__))
mcp = FastMCP("memskill-mcp-server")

_STORE_DIR = os.environ.get("MEMSKILL_STORE_DIR", "memory_bank")
_NAMESPACE = os.environ.get("MEMSKILL_NAMESPACE", "default")
_SKILLS_DIR = os.environ.get("MEMSKILL_SKILLS_DIR", f"{_STORE_DIR}/skills")

_store: MemoryStore | None = None
_skills: SkillLibrary | None = None


def _get_store() -> MemoryStore:
    global _store
    if _store is None:
        _store = MemoryStore(store_dir=_STORE_DIR, namespace=_NAMESPACE)
    return _store


def _get_skills() -> SkillLibrary:
    global _skills
    if _skills is None:
        _skills = SkillLibrary(skills_dir=_SKILLS_DIR)
    return _skills


@mcp.tool()
def memory_search(query: str, top_k: int = 5) -> str:
    """Search episodic and semantic memory for relevant past experiences and notes.

    Args:
        query: Natural language search query.
        top_k: Maximum number of results to return (default 5).
    """
    store = _get_store()
    results = store.search(query, top_k=top_k)
    return store.format_search_results(results)


@mcp.tool()
def memory_save(content: str, tags: str = "", kind: str = "semantic") -> str:
    """Save a note or fact to memory for future retrieval.

    Args:
        content: The note content to store (strategy, source URL, fact — not full answers).
        tags: Comma-separated tags, e.g. "finance,source,china".
        kind: Memory type: "semantic" (facts/notes) or "episodic" (experiences).
    """
    store = _get_store()
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
    if kind not in ("semantic", "episodic"):
        kind = "semantic"
    entry = store.add(content=content, kind=kind, tags=tag_list)
    return f"Saved memory entry id={entry.id} kind={entry.kind} tags={tag_list}"


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
