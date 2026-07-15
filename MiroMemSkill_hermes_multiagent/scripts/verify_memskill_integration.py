#!/usr/bin/env python3
"""Verify the three memory/skill linkage paths in benchmark logs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def verify_log(log_path: Path) -> dict:
    data = json.loads(log_path.read_text(encoding="utf-8"))
    sys_prompt = data.get("main_agent_message_history", {}).get("system_prompt", "")
    user_text = ""
    for msg in data.get("main_agent_message_history", {}).get("message_history", []):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, list):
                user_text = content[0].get("text", "")
            else:
                user_text = str(content)
            break

    steps = [s.get("step_name", "") for s in data.get("step_logs", [])]
    tool_calls = json.dumps(data).lower()

    return {
        "task_id": data.get("task_id"),
        "passive_injection": "Relevant Experience & Skills" in user_text,
        "mcp_tools_registered": all(
            t in sys_prompt for t in ("memory_search", "skill_load", "tool-memskill")
        ),
        "active_tool_use": any(
            x in tool_calls for x in ("memory_search", "skill_load", "memory_save")
        ),
        "memory_injection_step": "memory_injection" in steps,
        "judge_result": data.get("judge_result", ""),
        "status": data.get("status", ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", default="logs/memskill_smoke5", nargs="?")
    args = parser.parse_args()
    log_dir = Path(args.log_dir)
    logs = sorted(log_dir.glob("*_attempt_1.json"))
    if not logs:
        print(f"No logs in {log_dir}")
        sys.exit(1)

    for lp in logs:
        r = verify_log(lp)
        print(f"\n=== {r['task_id']} ({r['status']}) ===")
        print(f"  [1] Passive injection:     {'OK' if r['passive_injection'] else 'MISSING'}")
        print(f"  [2] MCP tools registered:  {'OK' if r['mcp_tools_registered'] else 'MISSING'}")
        print(f"  [3] Active tool use:       {'OK' if r['active_tool_use'] else 'pending/none'}")
        print(f"  memory_injection step:     {'OK' if r['memory_injection_step'] else 'MISSING'}")
        print(f"  judge: {r['judge_result'] or 'pending'}")


if __name__ == "__main__":
    main()
