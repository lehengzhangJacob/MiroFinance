# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

import pathlib
import sys
import importlib
from mcp import StdioServerParameters
from omegaconf import DictConfig, OmegaConf

from src.logging.logger import bootstrap_logger
from src.utils.env_loader import load_project_env
from config.agent_prompts.base_agent_prompt import BaseAgentPrompt

import os

LOGGER_LEVEL = os.getenv("LOGGER_LEVEL", "INFO")
logger = bootstrap_logger(level=LOGGER_LEVEL)


def _resolved_tool_env(tool_cfg: DictConfig, cfg: DictConfig) -> dict[str, str]:
    env = {
        str(key): str(value)
        for key, value in tool_cfg.get("env", {}).items()
    }
    if tool_cfg.get("name") != "tool-memskill":
        return env

    memory_cfg = cfg.get("memory", {})
    # Hydra is the source of truth for the namespace and backend. This prevents
    # passive injection and the MCP process from silently addressing different
    # user_id scopes in the shared Mem0 collection.
    env["MEMSKILL_NAMESPACE"] = str(
        memory_cfg.get("namespace", env.get("MEMSKILL_NAMESPACE", "default"))
    )
    env["MEMSKILL_STORE_DIR"] = str(
        memory_cfg.get("store_dir", env.get("MEMSKILL_STORE_DIR", "memory_bank"))
    )
    env["MEMSKILL_SKILLS_DIR"] = str(
        memory_cfg.get(
            "skills_dir",
            env.get("MEMSKILL_SKILLS_DIR", "memory_bank/skills"),
        )
    )
    env["MEMSKILL_MEMORY_BACKEND"] = str(
        memory_cfg.get("backend", "mem0_qdrant")
    )
    env["MEM0_QDRANT_HOST"] = str(
        memory_cfg.get(
            "qdrant_host",
            os.getenv("MEM0_QDRANT_HOST", "127.0.0.1"),
        )
    )
    env["MEM0_QDRANT_PORT"] = str(
        memory_cfg.get("qdrant_port", os.getenv("MEM0_QDRANT_PORT", "6333"))
    )
    env["MEM0_QDRANT_COLLECTION"] = str(
        memory_cfg.get(
            "qdrant_collection",
            os.getenv("MEM0_QDRANT_COLLECTION", "miromemskill"),
        )
    )
    env["MEM0_HISTORY_DB_PATH"] = str(
        memory_cfg.get(
            "history_db_path",
            os.getenv("MEM0_HISTORY_DB_PATH", "memory_bank/mem0_history.db"),
        )
    )
    env["MEM0_EMBEDDING_MODEL"] = str(
        memory_cfg.get("embedding_model", "embedding-3")
    )
    env["MEM0_EMBEDDING_DIMS"] = str(
        memory_cfg.get("embedding_dims", 2048)
    )
    for name in (
        "REFLECTION_LLM_API_KEY",
        "REFLECTION_LLM_BASE_URL",
        "REFLECTION_LLM_MODEL_NAME",
        "MEM0_TELEMETRY",
    ):
        value = os.getenv(name)
        if value is not None:
            env.setdefault(name, value)
    return env


# MCP server configuration generation function
def create_mcp_server_parameters(
    cfg: DictConfig, agent_cfg: DictConfig, logs_dir: str | None = None
):
    """Define and return MCP server configuration list"""
    load_project_env()
    configs = []

    if agent_cfg.get("tool_config", None) is not None:
        for tool in agent_cfg["tool_config"]:
            try:
                config_path = (
                    pathlib.Path(__file__).parent.parent.parent
                    / "config"
                    / "tool"
                    / f"{tool}.yaml"
                )
                tool_cfg = OmegaConf.load(config_path)
                tool_env = _resolved_tool_env(tool_cfg, cfg)
                configs.append(
                    {
                        "name": tool_cfg.get("name", tool),
                        "params": StdioServerParameters(
                            command=sys.executable
                            if tool_cfg["tool_command"] == "python"
                            else tool_cfg["tool_command"],
                            args=tool_cfg.get("args", []),
                            env=tool_env,
                        ),
                    }
                )
            except Exception as e:
                logger.error(
                    f"[ERROR] Error creating MCP server parameters for tool {tool}: {e}"
                )
                continue

    blacklist = set()
    for black_list_item in agent_cfg.get("tool_blacklist", []):
        blacklist.add((black_list_item[0], black_list_item[1]))
    return configs, blacklist


def _load_agent_prompt_class(prompt_class_name: str) -> BaseAgentPrompt:
    # Dynamically import the class from the config.agent_prompts module
    if not isinstance(prompt_class_name, str) or not prompt_class_name.isidentifier():
        raise ValueError(f"Invalid prompt class name: {prompt_class_name}")

    try:
        # Import the module dynamically
        agent_prompts_module = importlib.import_module("config.agent_prompts")
        # Get the class from the module
        PromptClass = getattr(agent_prompts_module, prompt_class_name)
    except (ModuleNotFoundError, AttributeError) as e:
        raise ImportError(
            f"Could not import class '{prompt_class_name}' from 'config.agent_prompts': {e}"
        )
    return PromptClass()


def expose_sub_agents_as_tools(sub_agents_cfg: DictConfig):
    """Expose sub-agents as tools"""
    sub_agents_server_params = []
    for sub_agent in sub_agents_cfg.keys():
        if not sub_agent.startswith("agent-"):
            raise ValueError(
                f"Sub-agent name must start with 'agent-': {sub_agent}. Please check the sub-agent name in the agent's config file."
            )
        try:
            sub_agent_prompt_instance = _load_agent_prompt_class(
                sub_agents_cfg[sub_agent].prompt_class
            )
            sub_agent_tool_definition = sub_agent_prompt_instance.expose_agent_as_tool(
                subagent_name=sub_agent
            )
            sub_agents_server_params.append(sub_agent_tool_definition)
        except Exception as e:
            raise ValueError(f"Failed to expose sub-agent {sub_agent} as a tool: {e}")
    return sub_agents_server_params
