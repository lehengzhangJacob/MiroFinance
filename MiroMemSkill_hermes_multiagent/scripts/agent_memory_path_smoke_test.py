#!/usr/bin/env python3
"""Verify passive agent injection and MCP search share one Mem0 collection."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

NAMESPACE = "__agent_memory_path_smoke__"
SENTINEL = "相对动量必须同时得到估值分位确认"
TASK = (
    "当前日期为 2024-08-01（收盘后）。请预测测试股票未来20个交易日"
    "相对沪深300是跑赢还是跑输。"
)


async def _mcp_search() -> str:
    env = dict(os.environ)
    env.update(
        {
            "MEMSKILL_STORE_DIR": str(ROOT / "memory_bank"),
            "MEMSKILL_NAMESPACE": NAMESPACE,
            "MEMSKILL_SKILLS_DIR": str(ROOT / "memory_bank" / "skills_ashare"),
            "MEMSKILL_MEMORY_BACKEND": "mem0_qdrant",
            "MEMSKILL_ALLOW_SAVE": "false",
            "MEM0_QDRANT_HOST": os.getenv("MEM0_QDRANT_HOST", "127.0.0.1"),
            "MEM0_QDRANT_PORT": os.getenv("MEM0_QDRANT_PORT", "6333"),
            "MEM0_QDRANT_COLLECTION": os.getenv(
                "MEM0_QDRANT_COLLECTION", "miromemskill"
            ),
            "MEM0_HISTORY_DB_PATH": str(ROOT / "memory_bank" / "mem0_history.db"),
            "MEM0_EMBEDDING_MODEL": "embedding-3",
            "MEM0_EMBEDDING_DIMS": "2048",
        }
    )
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "src.tool.mcp_servers.memskill_mcp_server"],
        env=env,
        cwd=str(ROOT),
    )
    async with stdio_client(parameters) as (read, write):
        async with ClientSession(
            read,
            write,
            sampling_callback=None,
        ) as session:
            await session.initialize()
            result = await session.call_tool(
                "memory_search",
                arguments={
                    "query": SENTINEL,
                    "top_k": 3,
                    "before_month": "2024-08",
                    "before_date": "20240801",
                },
            )
            return result.content[-1].text if result.content else ""


def main() -> int:
    if not os.getenv("GLM_API_KEY"):
        raise RuntimeError("GLM_API_KEY is required")

    from src.memory.context import build_memory_context_block
    from src.memory.memory import Mem0Memory
    from src.memory.official_mem0_store import OfficialMem0Store

    store = OfficialMem0Store(ROOT / "memory_bank", namespace=NAMESPACE)
    store.reset_namespace()
    try:
        store.add(
            f"条件规则：{SENTINEL}，否则降低信号置信度。",
            metadata={
                "source": "rolling_statistical",
                "rule_id": "agent_path",
                "entry_month": "2024-07",
                "available_after": "20240729",
                "functional_stance": "neutral",
                "q_value": 0.05,
                "validation_lift": 0.15,
                "validation_support": 12,
            },
        )
        memory = Mem0Memory(store)
        passive_block = build_memory_context_block(
            TASK,
            memory,
            skill_lib=None,
            memory_enabled=True,
            skill_enabled=False,
        )
        if SENTINEL not in passive_block:
            raise AssertionError("passive agent context did not read the sentinel")

        mcp_result = asyncio.run(_mcp_search())
        if SENTINEL not in mcp_result:
            raise AssertionError(
                f"MCP memory_search did not read the sentinel: {mcp_result}"
            )
        print(
            "Agent memory-path smoke passed: passive injection and MCP search "
            "read the same Qdrant namespace."
        )
        return 0
    finally:
        store.reset_namespace()


if __name__ == "__main__":
    raise SystemExit(main())
