# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

"""Build pre-task memory/skill context block for orchestrator injection."""

from __future__ import annotations

import re
from functools import lru_cache

from omegaconf import DictConfig

from src.memory.memory import Mem0Memory
from src.memory.rank_reflection import build_rank_factor_block
from src.memory.skills import SkillLibrary
from src.memory.store_factory import create_memory_store

# Benchmark task prompts share large boilerplate sections (data-usage rules,
# output format). Cutting at these markers keys retrieval on the task-specific
# head (stock, industry, date) instead of template text shared by every task.
_QUERY_CUT_MARKERS = ["数据使用规则", "输出要求", "Data usage rules"]

# Point-in-time anchor in ashare task prompts: 当前日期为 2025-04-01（收盘后）
_AS_OF_RE = re.compile(r"当前日期为\s*(\d{4})-(\d{2})-(\d{2})")


def compact_task_query(task_description: str, max_chars: int = 400) -> str:
    text = task_description
    for marker in _QUERY_CUT_MARKERS:
        idx = text.find(marker)
        if idx > 0:
            text = text[:idx]
            break
    return text.strip()[:max_chars]


def task_before_month(task_description: str) -> str:
    """Extract the task's market month (YYYY-MM) for temporal memory filtering."""
    m = _AS_OF_RE.search(task_description)
    return f"{m.group(1)}-{m.group(2)}" if m else ""


def task_as_of_date(task_description: str) -> str:
    """Extract the exact task decision date as YYYYMMDD."""
    m = _AS_OF_RE.search(task_description)
    return f"{m.group(1)}{m.group(2)}{m.group(3)}" if m else ""


def get_memory_components(cfg: DictConfig) -> tuple[Mem0Memory | None, SkillLibrary | None]:
    if not cfg.get("memory") or not cfg.memory.get("enabled", False):
        return None, None

    store_dir = cfg.memory.get("store_dir", "memory_bank")
    namespace = cfg.memory.get("namespace", "default")
    skills_dir = cfg.memory.get("skills_dir", f"{store_dir}/skills")

    return _cached_memory_components(
        store_dir=store_dir,
        namespace=namespace,
        backend=cfg.memory.get("backend", "mem0_qdrant"),
        embedding_model=cfg.memory.get("embedding_model", "embedding-3"),
        embedding_dims=int(cfg.memory.get("embedding_dims", 2048)),
        qdrant_host=cfg.memory.get("qdrant_host"),
        qdrant_port=cfg.memory.get("qdrant_port"),
        collection_name=cfg.memory.get("qdrant_collection"),
        history_db_path=cfg.memory.get("history_db_path"),
        skills_dir=skills_dir,
        skill_enabled=bool(cfg.memory.get("skill_enabled", True)),
    )


@lru_cache(maxsize=32)
def _cached_memory_components(
    *,
    store_dir: str,
    namespace: str,
    backend: str,
    embedding_model: str,
    embedding_dims: int,
    qdrant_host: str | None,
    qdrant_port: int | str | None,
    collection_name: str | None,
    history_db_path: str | None,
    skills_dir: str,
    skill_enabled: bool,
) -> tuple[Mem0Memory, SkillLibrary | None]:
    store = create_memory_store(
        store_dir=store_dir,
        namespace=namespace,
        backend=backend,
        embedding_model=embedding_model,
        embedding_dims=embedding_dims,
        qdrant_host=qdrant_host,
        qdrant_port=int(qdrant_port) if qdrant_port is not None else None,
        collection_name=collection_name,
        history_db_path=history_db_path,
    )
    memory = Mem0Memory(store=store)
    skill_lib = SkillLibrary(skills_dir=skills_dir) if skill_enabled else None
    return memory, skill_lib


def build_memory_context_block(
    task_description: str,
    store: Mem0Memory,
    skill_lib: SkillLibrary | None,
    inject_top_k: int = 3,
    skill_top_k: int = 2,
    memory_enabled: bool = True,
    skill_enabled: bool = True,
    skill_preview_min_score: float = 0.0,
    skill_preview_max_chars: int = 1500,
    list_other_skills: bool = True,
    calibration_enabled: bool = False,
    calibration_mode: str = "",
    calibration_min_samples: int = 16,
    rolling_status_enabled: bool = False,
    rolling_min_samples: int = 64,
    rolling_min_months: int = 6,
    rank_factor_enabled: bool = False,
    rank_factor_min_months: int = 3,
    rank_factor_fdr_q: float = 0.10,
    rank_factor_status_enabled: bool = False,
    trader_episode_enabled: bool = False,
    trader_episode_max: int = 3,
) -> str:
    sections: list[str] = []
    query = compact_task_query(task_description)
    before_month = task_before_month(task_description)
    before_date = task_as_of_date(task_description)

    mode = (calibration_mode or ("legacy" if calibration_enabled else "off")).lower()
    if mode != "off" and before_month:
        try:
            if mode == "reliability":
                calib = store.reliability_block(
                    before_month,
                    min_samples=calibration_min_samples,
                    before_date=before_date,
                )
            elif mode == "legacy":
                calib = store.calibration_block(
                    before_month,
                    min_samples=calibration_min_samples,
                    before_date=before_date,
                )
            else:
                raise ValueError(f"Unknown calibration_mode: {calibration_mode!r}")
        except Exception:
            calib = ""
        if calib:
            sections.append(calib)

    if rolling_status_enabled and before_date:
        try:
            snapshot = store.rolling_snapshot(before_date)
        except Exception:
            snapshot = {"samples": 0, "months": 0, "rules": 0}
        if snapshot["rules"]:
            status = (
                f"已有 {snapshot['samples']} 条、{snapshot['months']} 个月的已到期样本，"
                f"其中 {snapshot['rules']} 条条件规则通过时间验证；仅在当前特征满足条件时使用。"
            )
        elif (
            snapshot["samples"] < rolling_min_samples
            or snapshot["months"] < rolling_min_months
        ):
            status = (
                f"当前只有 {snapshot['samples']} 条、{snapshot['months']} 个月的已到期样本，"
                f"尚未达到 {rolling_min_samples} 条且 {rolling_min_months} 个月的启用门槛。"
            )
        else:
            status = (
                f"已有 {snapshot['samples']} 条、{snapshot['months']} 个月的已到期样本，"
                "但没有任何条件规则同时通过训练支持度、最近月份时间验证、跨月一致性和FDR检验。"
            )
        sections.append(
            "### 大样本记忆状态\n"
            + status
            + " 不得自行从历史样本编造方向规则；请以当前任务的Qlib、行情、估值和基本面证据独立判断。"
        )

    if rank_factor_enabled and before_date:
        try:
            rank_block = build_rank_factor_block(
                store,
                before_date,
                min_months=rank_factor_min_months,
                fdr_q=rank_factor_fdr_q,
                show_status_when_empty=rank_factor_status_enabled,
            )
        except Exception:
            rank_block = ""
        if rank_block:
            sections.append(rank_block)

    if trader_episode_enabled and before_date:
        try:
            episode_block = store.trader_episode_block(
                before_date,
                max_episodes=trader_episode_max,
            )
        except Exception:
            episode_block = ""
        if episode_block:
            sections.append(episode_block)

    memory_search_available = False
    if memory_enabled:
        # Temporal filter: only lessons from strictly earlier market months are
        # visible, so shuffled/concurrent execution cannot leak future lessons.
        # Stance quota uses metadata-derived functional stance, so the injected
        # block never becomes a direction prior.
        memory_search_available = store.has_vector_memories(
            before_month=before_month,
            before_date=before_date,
        )
        results = (
            store.search(
                query,
                top_k=inject_top_k,
                before_month=before_month,
                before_date=before_date,
                stance_balance=True,
            )
            if memory_search_available
            else []
        )
        if results:
            sections.append(
                "### Relevant Past Experiences\n"
                "(Conditional heuristics from past tasks — NOT direction priors. "
                "Evaluate the current task's own evidence first; your final call must "
                "be supported by the data you retrieve now, not by these notes.)\n"
                + store.format_results(results)
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
                skill_text = skill_lib.load_skill_text(top_skill.name)
                # skill_preview_max_chars <= 0 injects the full body. Skill
                # evolution arms need this: candidate edits beyond a fixed
                # truncation point would otherwise be invisible to the agent.
                if skill_preview_max_chars > 0:
                    skill_text = skill_text[:skill_preview_max_chars]
                sections.append("### Top Skill Preview\n" + skill_text)
        # Optionally list skills whose triggers do not appear in the task text.
        matched_names = {s.name for s, _ in matches}
        others = [s for s in skill_lib.list_skills() if s["name"] not in matched_names]
        if list_other_skills and others:
            sections.append(
                "### Other Available Skills (load with skill_load(name))\n"
                + "\n".join(f"- **{s['name']}**: {s['description']}" for s in others)
            )

    if not sections:
        return ""

    available_tools = []
    if memory_search_available:
        available_tools.append("memory_search")
    if skill_enabled and skill_lib:
        available_tools.append("skill_load")
    tool_hint = (
        "You may also call "
        + " / ".join(available_tools)
        + " during reasoning for more detail.\n\n"
        if available_tools
        else ""
    )
    return (
        "\n\n## Relevant Experience & Skills (from memory bank)\n\n"
        "Use the following retrieved experiences and skills to guide your approach. "
        + tool_hint
        + "\n\n".join(sections)
    )
