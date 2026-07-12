# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import datetime
import os
import sys
import time
import uuid
from typing import Any, Optional
import importlib
from config.agent_prompts.base_agent_prompt import BaseAgentPrompt

from omegaconf import DictConfig


from src.llm.provider_client_base import LLMProviderClientBase
from src.llm.providers.claude_openrouter_client import ContextLimitError
from src.logging.logger import bootstrap_logger
from src.logging.task_tracer import TaskTracer
from src.tool.manager import ToolManager
from src.utils.io_utils import OutputFormatter, process_input
from src.utils.tool_utils import expose_sub_agents_as_tools
from src.utils.summary_utils import (
    extract_hints,
    extract_gaia_final_answer,
    extract_browsecomp_zh_final_answer,
)
from src.memory.context import (
    build_memory_context_block,
    get_memory_components,
    task_as_of_date,
)
from src.memory.feature_evidence import build_feature_evidence_block
from src.utils.ashare_trader import (
    canonicalize_core_satellite_answer,
    validate_portfolio_answer,
)

LOGGER_LEVEL = os.getenv("LOGGER_LEVEL", "INFO")
logger = bootstrap_logger(level=LOGGER_LEVEL)
MAX_TRADER_PORTFOLIO_REPAIRS = 2
REQUIRED_TRADER_ANCHOR_TOOLS = frozenset(
    {"ashare_momentum_baseline", "ashare_market_breadth"}
)


def _list_tools(sub_agent_tool_managers: dict[str, ToolManager]):
    # Use a dictionary to store the cached result
    cache = None

    async def wrapped():
        nonlocal cache
        if cache is None:
            # Only fetch tool definitions if not already cached
            # Handle empty sub_agent_tool_managers (single agent mode)
            if not sub_agent_tool_managers:
                result = {}
            else:
                result = {
                    name: await tool_manager.get_all_tool_definitions()
                    for name, tool_manager in sub_agent_tool_managers.items()
                }
            cache = result
        return cache

    return wrapped


def _generate_message_id() -> str:
    """Generate random message ID using common LLM format"""
    # Use 8-character random hex string, similar to OpenAI API format, avoid cross-conversation cache hits
    return f"msg_{uuid.uuid4().hex[:8]}"


def _normalize_ashare_date(value: Any) -> str:
    return str(value or "").strip().replace("-", "").replace("/", "")


def _is_rate_limit_error(error: BaseException) -> bool:
    """Recognize OpenAI 429 errors even when wrapped by tenacity RetryError."""
    pending: list[Any] = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if (
            type(current).__name__ == "RateLimitError"
            or getattr(current, "status_code", None) == 429
            or "rate limit" in str(current).lower()
            or "error code: 429" in str(current).lower()
        ):
            return True
        last_attempt = getattr(current, "last_attempt", None)
        if last_attempt is not None:
            try:
                nested = last_attempt.exception()
            except Exception:
                nested = None
            if nested is not None:
                pending.append(nested)
        for nested in (
            getattr(current, "__cause__", None),
            getattr(current, "__context__", None),
        ):
            if nested is not None:
                pending.append(nested)
    return False


def _ashare_trader_anchor_enabled(
    metadata: dict[str, Any] | None,
) -> bool:
    if not metadata or metadata.get("task_type") != "portfolio_allocation":
        return False
    policy = metadata.get("anchor_policy")
    return bool(isinstance(policy, dict) and policy.get("enabled", False))


def _ashare_trader_core_satellite_enabled(
    metadata: dict[str, Any] | None,
) -> bool:
    if not _ashare_trader_anchor_enabled(metadata):
        return False
    policy = metadata.get("anchor_policy") if metadata else None
    return bool(
        isinstance(policy, dict)
        and str(policy.get("mode", "legacy")).strip() == "core_satellite"
    )


def _ashare_trader_mandatory_tool_error(
    metadata: dict[str, Any] | None,
    tools_seen: set[str] | frozenset[str],
) -> str:
    if not _ashare_trader_anchor_enabled(metadata):
        return ""
    missing = sorted(REQUIRED_TRADER_ANCHOR_TOOLS - set(tools_seen))
    if not missing:
        return ""
    return "mandatory hard-anchor tools not called successfully: " + ",".join(missing)


def _replace_or_append_final_boxed(text: str, canonical_boxed: str) -> str:
    """Replace the last balanced boxed value, or append the canonical one."""
    raw = str(text or "").rstrip()
    marker = r"\boxed{"
    search_from = 0
    last_span: tuple[int, int] | None = None
    while search_from < len(raw):
        start = raw.find(marker, search_from)
        if start < 0:
            break
        cursor = start + len(marker)
        depth = 1
        while cursor < len(raw) and depth:
            if raw[cursor] == "{":
                depth += 1
            elif raw[cursor] == "}":
                depth -= 1
            cursor += 1
        if depth == 0:
            last_span = (start, cursor)
            search_from = cursor
        else:
            search_from = start + len(marker)
    if last_span is not None:
        start, end = last_span
        return raw[:start] + canonical_boxed + raw[end:]
    return f"{raw}\n{canonical_boxed}" if raw else canonical_boxed


def _canonicalize_core_satellite_response(
    answer: str,
    metadata: dict[str, Any] | None,
) -> tuple[str, Any]:
    """Canonicalize one response and persist its auditable selection facts."""
    canonical = canonicalize_core_satellite_answer(answer, metadata)
    if canonical is None:
        raise ValueError("core-satellite canonicalization was not active")
    if metadata is None:
        raise ValueError("core-satellite canonicalization requires task metadata")
    candidate_rows = metadata.get("satellite_candidates")
    if (
        not isinstance(candidate_rows, list)
        or not candidate_rows
        or not isinstance(candidate_rows[0], dict)
    ):
        raise ValueError(
            "core-satellite canonicalization requires structured satellite_candidates"
        )
    candidate_codes = {
        str(row.get("ts_code", "")).strip().upper()
        for row in candidate_rows
        if isinstance(row, dict)
    }
    required_candidate_codes = set(canonical.selected_codes) | set(
        canonical.deterministic_codes
    )
    if not required_candidate_codes.issubset(candidate_codes):
        raise ValueError(
            "core-satellite selection codes are missing from satellite_candidates"
        )
    signal_date = str(
        metadata.get("satellite_signal_date")
        or candidate_rows[0].get("signal_date")
        or ""
    ).strip()
    target = str(
        metadata.get("satellite_target")
        or candidate_rows[0].get("target")
        or ""
    ).strip()
    if not signal_date or not target:
        raise ValueError(
            "core-satellite candidates require signal date and target metadata"
        )
    selection = {
        "selected_codes": list(canonical.selected_codes),
        "deterministic_codes": list(canonical.deterministic_codes),
        "selection_fallback": bool(canonical.selection_fallback),
        "source": canonical.source,
        "diagnostic": canonical.diagnostic,
        "candidate_signal_date": signal_date,
        "target": target,
    }
    metadata["satellite_selection"] = selection
    return (
        _replace_or_append_final_boxed(answer, canonical.canonical_boxed_answer),
        canonical,
    )


def _validate_ashare_trader_tool_call(
    tool_name: str,
    arguments: Any,
    metadata: dict[str, Any] | None,
) -> None:
    """Reject pool escapes and lookahead calls in unified-trader tasks."""
    if not metadata or metadata.get("task_type") != "portfolio_allocation":
        return
    if not isinstance(arguments, dict):
        raise ValueError("ashare-trader tool arguments must be an object")

    pool = {str(code).upper() for code in metadata.get("stock_pool", [])}
    ts_code = str(arguments.get("ts_code", "")).upper()
    if ts_code and ts_code not in pool:
        raise ValueError(
            f"{tool_name} rejected stock outside the task pool: {ts_code}; "
            "use only one of the 16 codes in the original task"
        )

    cutoff = _normalize_ashare_date(
        metadata.get("entry_date") or metadata.get("as_of")
    )
    requested_as_of = _normalize_ashare_date(arguments.get("as_of"))
    if (
        cutoff
        and requested_as_of
        and len(cutoff) == 8
        and len(requested_as_of) == 8
        and requested_as_of > cutoff
    ):
        raise ValueError(
            f"{tool_name} rejected lookahead as_of={requested_as_of}; "
            f"the task cutoff is {cutoff}"
        )
    if (
        _ashare_trader_anchor_enabled(metadata)
        and tool_name in REQUIRED_TRADER_ANCHOR_TOOLS
    ):
        if requested_as_of != cutoff:
            raise ValueError(
                f"{tool_name} must use the exact task cutoff as_of={cutoff}"
            )
        if tool_name == "ashare_momentum_baseline":
            try:
                window = int(arguments.get("window", 20))
                top_k = int(arguments.get("top_k", 4))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "ashare_momentum_baseline requires window=20 and top_k=4"
                ) from exc
            if window != 20 or top_k != 4:
                raise ValueError(
                    "ashare_momentum_baseline requires window=20 and top_k=4"
                )


def _ashare_trader_portfolio_error(
    answer: str,
    metadata: dict[str, Any] | None,
) -> str:
    if not metadata or metadata.get("task_type") != "portfolio_allocation":
        return ""
    validation = validate_portfolio_answer(answer, metadata)
    return "" if validation.ok else validation.error


def _ashare_trader_terminal_error(
    answer: str,
    metadata: dict[str, Any] | None,
    tools_seen: set[str] | frozenset[str],
) -> str:
    errors: list[str] = []
    mandatory_error = _ashare_trader_mandatory_tool_error(metadata, tools_seen)
    if mandatory_error:
        errors.append(mandatory_error)
    portfolio_error = _ashare_trader_portfolio_error(answer, metadata)
    if portfolio_error:
        errors.append(portfolio_error)
    return "; ".join(errors)


def _filter_memskill_tool_definitions(
    tool_definitions: list[dict[str, Any]],
    *,
    expose_memory_search: bool,
    expose_memory_save: bool,
) -> list[dict[str, Any]]:
    """Hide unavailable memory tools from the prompt without mutating the input."""
    filtered: list[dict[str, Any]] = []
    for server in tool_definitions:
        copied = dict(server)
        tools = []
        for tool in server.get("tools", []):
            name = tool.get("name") if isinstance(tool, dict) else None
            if name == "memory_search" and not expose_memory_search:
                continue
            if name == "memory_save" and not expose_memory_save:
                continue
            tools.append(tool)
        copied["tools"] = tools
        filtered.append(copied)
    return filtered


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


class Orchestrator:
    def __init__(
        self,
        main_agent_tool_manager: ToolManager,
        sub_agent_tool_managers: dict[str, ToolManager],
        llm_client: LLMProviderClientBase,
        output_formatter: OutputFormatter,
        cfg: DictConfig,
        task_log: TaskTracer,
        sub_agent_llm_client: Optional[LLMProviderClientBase] = None,
    ):
        self.main_agent_tool_manager = main_agent_tool_manager
        self.sub_agent_tool_managers = sub_agent_tool_managers
        self.llm_client = llm_client
        self.sub_agent_llm_client = (
            sub_agent_llm_client or llm_client
        )  # Use client from main agent if not provided
        self.output_formatter = output_formatter
        self.cfg = cfg
        self.task_log = task_log
        # call this once, then use cache value
        self._list_sub_agent_tools = _list_tools(sub_agent_tool_managers)

        self.chinese_context = (
            self.cfg.main_agent.chinese_context.lower().strip() == "true"
        )

        # Handle add_message_id configuration, support string to bool conversion
        add_message_id_val = self.cfg.main_agent.get("add_message_id", False)
        if isinstance(add_message_id_val, str):
            self.add_message_id: bool = add_message_id_val.lower().strip() == "true"
        else:
            self.add_message_id: bool = bool(add_message_id_val)
        logger.info(
            f"add_message_id config value: {add_message_id_val} (type: {type(add_message_id_val)}) -> parsed as: {self.add_message_id}"
        )

        # Pass task_log to llm_client
        if self.llm_client and task_log:
            self.llm_client.task_log = task_log
        if (
            self.sub_agent_llm_client
            and task_log
            and self.sub_agent_llm_client != self.llm_client
        ):
            self.sub_agent_llm_client.task_log = task_log

    async def _handle_llm_call_with_logging(
        self,
        system_prompt,
        message_history,
        tool_definitions,
        step_id: int,
        purpose: str = "LLM call",
        keep_tool_result: int = -1,
        agent_type: str = "main",
    ) -> tuple[str | None, bool, Any | None]:
        """Unified LLM call and logging handling
        Returns:
            tuple[Optional[str], bool, Optional[object]]: (response_text, should_break, tool_calls_info)
        """

        # Select correct LLM client based on agent_type
        current_llm_client = (
            self.llm_client if agent_type == "main" else self.sub_agent_llm_client
        )

        # Add message ID to user messages (if configured and message doesn't have ID yet)
        if self.add_message_id:
            for message in message_history:
                if message.get("role") == "user":
                    content = message.get("content")
                    if isinstance(content, list):
                        # content is list format (Anthropic style)
                        for content_item in content:
                            if content_item.get("type") == "text":
                                text = content_item["text"]
                                # Check if message ID already exists
                                if not text.startswith("[msg_"):
                                    message_id = _generate_message_id()
                                    content_item["text"] = f"[{message_id}] {text}"
                    elif isinstance(content, str):
                        # content is string format (simple format)
                        if not content.startswith("[msg_"):
                            message_id = _generate_message_id()
                            message["content"] = f"[{message_id}] {content}"

        # Save message history before LLM call
        if self.task_log:
            if agent_type == "main":
                self.task_log.main_agent_message_history = {
                    "system_prompt": system_prompt,
                    "message_history": message_history,
                }
            elif self.task_log.current_sub_agent_session_id:
                self.task_log.sub_agent_message_history_sessions[
                    self.task_log.current_sub_agent_session_id
                ] = {"system_prompt": system_prompt, "message_history": message_history}
            self.task_log.save()

        try:
            response = await current_llm_client.create_message(
                system_prompt=system_prompt,
                message_history=message_history,
                tool_definitions=tool_definitions,
                keep_tool_result=self.cfg.main_agent.keep_tool_result,
                step_id=step_id,
                task_log=self.task_log,
                agent_type=agent_type,
            )

            if response:
                # Use client's response processing method
                assistant_response_text, should_break = (
                    current_llm_client.process_llm_response(
                        response, message_history, agent_type
                    )
                )

                # Save message history after LLM response processing
                if self.task_log:
                    if agent_type == "main":
                        self.task_log.main_agent_message_history = {
                            "system_prompt": system_prompt,
                            "message_history": message_history,
                        }
                    elif self.task_log.current_sub_agent_session_id:
                        self.task_log.sub_agent_message_history_sessions[
                            self.task_log.current_sub_agent_session_id
                        ] = {
                            "system_prompt": system_prompt,
                            "message_history": message_history,
                        }
                    self.task_log.save()

                # Use client's tool call information extraction method
                tool_calls_info = current_llm_client.extract_tool_calls_info(
                    response, assistant_response_text
                )

                if assistant_response_text:
                    self.task_log.log_step(
                        f"{purpose.lower().replace(' ', '_')}_success",
                        f"{purpose} completed successfully",
                    )
                    return assistant_response_text, should_break, tool_calls_info
                else:
                    self.task_log.log_step(
                        f"{purpose.lower().replace(' ', '_')}_failed",
                        f"{purpose} returned no valid response",
                        "failed",
                    )
                    return None, True, None
            else:
                self.task_log.log_step(
                    f"{purpose.lower().replace(' ', '_')}_failed",
                    f"{purpose} returned no valid response",
                    "failed",
                )
                return None, True, None

        except asyncio.TimeoutError:
            logger.debug(f"⚠️ {purpose} timed out")
            self.task_log.log_step(
                f"{purpose.lower().replace(' ', '_')}_timeout",
                f"{purpose} timed out",
                "failed",
            )
            return None, True, None

        except ContextLimitError as e:
            logger.debug(f"⚠️ {purpose} context limit exceeded: {e}")
            self.task_log.log_step(
                f"{purpose.lower().replace(' ', '_')}_context_limit",
                f"{purpose} context limit exceeded: {str(e)}",
                "warning",
            )
            # For context limit exceeded, return special identifier for upper layer handling
            return None, True, "context_limit"

        except Exception as e:
            logger.debug(f"⚠️ {purpose} call failed: {e}")
            failure_kind = "rate_limit" if _is_rate_limit_error(e) else None
            self.task_log.log_step(
                f"{purpose.lower().replace(' ', '_')}_error",
                f"{purpose} failed: {str(e)}",
                "failed",
            )
            return None, True, failure_kind

    async def _handle_summary_with_context_limit_retry(
        self,
        system_prompt,
        agent_prompt_instance,
        message_history,
        tool_definitions,
        purpose,
        task_description,
        task_failed,
        agent_type="main",
        task_guidence="",
    ):
        """
        Handle context limit retry logic when processing summary

        Returns:
            str: final_answer_text - LLM generated summary text, error message on failure

        Only a confirmed context-limit error may remove dialogue.  Network and
        429 failures keep the evidence intact and stop after bounded backoff.
        """
        retry_count = 0

        while True:
            # Generate summary prompt
            summary_prompt = agent_prompt_instance.generate_summarize_prompt(
                task_description + task_guidence,
                task_failed=task_failed,
                chinese_context=self.chinese_context,
            )

            # Handle merging of message history and summary prompt
            current_llm_client = (
                self.llm_client if agent_type == "main" else self.sub_agent_llm_client
            )
            summary_prompt = current_llm_client.handle_max_turns_reached_summary_prompt(
                message_history, summary_prompt
            )

            # Directly add summary prompt to message history
            message_history.append(
                {"role": "user", "content": [{"type": "text", "text": summary_prompt}]}
            )

            tool_calls_info = None
            for network_retry_count in range(5):
                (
                    response_text,
                    _,
                    tool_calls_info,
                ) = await self._handle_llm_call_with_logging(
                    system_prompt,
                    message_history,
                    tool_definitions,
                    999,
                    purpose,
                    agent_type=agent_type,
                )
                if response_text or tool_calls_info == "context_limit":
                    break
                delay = (
                    min(300, 60 * (2**network_retry_count))
                    if tool_calls_info == "rate_limit"
                    else 60
                )
                if network_retry_count == 4:
                    logger.error("LLM summary retries exhausted after attempt 5/5")
                    self.task_log.log_step(
                        f"{agent_type}_summary_retry_exhausted",
                        "LLM summary retries exhausted after attempt 5/5; "
                        "conversation context remains intact",
                        "failed",
                    )
                    break
                logger.error(
                    "LLM summary process call failed, attempt "
                    f"{network_retry_count+1}/5, retrying after {delay} seconds..."
                )
                self.task_log.log_step(
                    f"{agent_type}_summary_retry",
                    "LLM summary process call failed, attempt "
                    f"{network_retry_count+1}/5, retrying after {delay} seconds...",
                    "warning",
                )
                await asyncio.sleep(delay)

            if response_text:
                # Call successful: return generated summary text
                return response_text

            if tool_calls_info != "context_limit":
                if message_history and message_history[-1]["role"] == "user":
                    message_history.pop()
                failure = (
                    "rate limit"
                    if tool_calls_info == "rate_limit"
                    else "network/API failure"
                )
                self.task_log.log_step(
                    f"{agent_type}_summary_network_failed",
                    f"Summary stopped after bounded {failure} retries; context kept intact",
                    "failed",
                )
                return (
                    "[ERROR] Unable to generate final summary after bounded "
                    f"{failure} retries. Conversation context was preserved."
                )

            # Confirmed context limit: remove one dialogue pair and retry.
            retry_count += 1
            logger.debug(
                f"LLM call failed (context_limit), attempt {retry_count} retry, removing recent assistant-user dialogue"
            )
            # First remove the just-added summary prompt
            if message_history and message_history[-1]["role"] == "user":
                message_history.pop()
            # Remove the most recent assistant message (tool call request)
            if message_history and message_history[-1]["role"] == "assistant":
                message_history.pop()
            # Once assistant-user dialogue needs to be removed, task fails (information is lost)
            task_failed = True
            # If there are no more dialogues to remove
            if len(message_history) <= 2:  # Only initial system-user messages remain
                logger.warning(
                    "Removed all removable dialogues, but still unable to generate summary"
                )
                break
            self.task_log.log_step(
                f"{agent_type}_summary_context_retry",
                f"Removed assistant-user pair, retry {retry_count}, task marked as failed",
                "warning",
            )

        # If still fails after removing all dialogues
        logger.error(
            "Summary failed after several attempts (removing all possible messages)"
        )
        self.task_log.log_step(
            f"{agent_type}_summary_failed",
            "Summary failed after several attempts (removing all possible messages)",
            "failed",
        )
        return "[ERROR] Unable to generate final summary due to context limit or network issues. You should try again."

    async def run_sub_agent(
        self, sub_agent_name, task_description, keep_tool_result: int = -1
    ):
        """
        Run sub agent
        """
        logger.debug(f"\n=== Starting Sub Agent {sub_agent_name} ===")
        task_description += "\n\nPlease provide the answer and detailed supporting information of the subtask given to you."
        logger.debug(f"Subtask: {task_description}")

        # Start new sub-agent session
        self.task_log.start_sub_agent_session(sub_agent_name, task_description)

        # Reset sub-agent usage stats for independent tracking
        self.sub_agent_llm_client.reset_usage_stats()

        # Simplified initial user content (no file attachments)
        initial_user_content = [{"type": "text", "text": task_description}]
        message_history = [{"role": "user", "content": initial_user_content}]

        # Get sub-agent tool definitions
        tool_definitions = await self._list_sub_agent_tools()
        tool_definitions = tool_definitions.get(sub_agent_name, [])
        self.task_log.log_step(
            f"get_sub_{sub_agent_name}_tool_definitions", f"{tool_definitions}"
        )

        if not tool_definitions:
            logger.debug(
                "Warning: Failed to get any tool definitions. LLM may not be able to use tools."
            )
            self.task_log.log_step(
                f"{sub_agent_name}_no_tools",
                f"No tool definitions available for {sub_agent_name}",
                "warning",
            )

        # Generate sub-agent system prompt
        if not self.cfg.sub_agents or sub_agent_name not in self.cfg.sub_agents:
            raise ValueError(f"Sub-agent {sub_agent_name} not found in configuration")
        sub_agent_prompt_instance = _load_agent_prompt_class(
            self.cfg.sub_agents[sub_agent_name].prompt_class
        )
        system_prompt = sub_agent_prompt_instance.generate_system_prompt_with_mcp_tools(
            mcp_servers=tool_definitions,
            chinese_context=self.chinese_context,
        )

        # Limit sub-agent turns
        max_turns = self.cfg.sub_agents[sub_agent_name].max_turns
        if max_turns < 0:
            max_turns = sys.maxsize
        turn_count = 0
        all_tool_results_content_with_id = []
        task_failed = False  # Track whether task failed

        while turn_count < max_turns:
            turn_count += 1
            logger.debug(f"\n--- Sub Agent {sub_agent_name} Turn {turn_count} ---")
            self.task_log.save()

            # Use unified LLM call handling
            (
                assistant_response_text,
                should_break,
                tool_calls,
            ) = await self._handle_llm_call_with_logging(
                system_prompt,
                message_history,
                tool_definitions,
                turn_count,
                f"Sub agent {sub_agent_name} turn {turn_count}",
                keep_tool_result=keep_tool_result,
                agent_type=sub_agent_name,
            )

            # Handle LLM response
            if assistant_response_text:
                if should_break:
                    self.task_log.log_step(
                        "sub_agent_early_termination",
                        f"Sub agent {sub_agent_name} terminated early on turn {turn_count}",
                    )
                    break
            else:
                # LLM call failed, mark task as failed and end current turn
                if tool_calls == "context_limit":
                    # Context limit exceeded situation
                    self.task_log.log_step(
                        "sub_agent_context_limit_reached",
                        f"Sub agent {sub_agent_name} context limit reached, jumping to summary",
                        "warning",
                    )
                else:
                    # Other LLM call failure situations
                    self.task_log.log_step(
                        "sub_agent_llm_call_failed",
                        "LLM call failed",
                        "failed",
                    )
                task_failed = True  # Mark task as failed
                break

            # Use tool calls parsed from LLM response
            if (
                tool_calls is None
                or len(tool_calls) < 2
                or (len(tool_calls[0]) == 0 and len(tool_calls[1]) == 0)
            ):
                logger.debug(
                    f"Sub Agent {sub_agent_name} did not request tool use, ending task."
                )
                self.task_log.log_step(
                    "sub_agent_no_tool_calls",
                    f"No tool calls found in sub agent {sub_agent_name}, ending on turn {turn_count}",
                )
                break

            # Execute tool calls
            tool_calls_data = []
            all_tool_results_content_with_id = []

            # Get maximum tool calls per turn from configuration
            max_tool_calls = self.cfg.sub_agents[sub_agent_name].max_tool_calls_per_turn
            tool_calls_exceeded = (
                len(tool_calls) > 0 and len(tool_calls[0]) > max_tool_calls
            )
            if tool_calls_exceeded:
                logger.warning(
                    f"[ERROR] Sub agent single turn tool call count too high ({len(tool_calls[0])} calls), only processing first {max_tool_calls}"
                )

            for call in tool_calls[0][:max_tool_calls]:
                # This place can be used to inject arguments of tools
                server_name = call["server_name"]
                tool_name = call["tool_name"]
                arguments = call["arguments"]
                call_id = call["id"]

                self.task_log.log_step(
                    "sub_agent_tool_call_start",
                    f"Executing {tool_name} on {server_name}",
                )

                call_start_time = time.time()
                try:
                    tool_result = await self.sub_agent_tool_managers[
                        sub_agent_name
                    ].execute_tool_call(server_name, tool_name, arguments)

                    call_end_time = time.time()
                    call_duration_ms = int((call_end_time - call_start_time) * 1000)

                    self.task_log.log_step(
                        "sub_agent_tool_call_success",
                        f"Tool {tool_name} executed successfully in {call_duration_ms}ms",
                    )

                    tool_calls_data.append(
                        {
                            "server_name": server_name,
                            "tool_name": tool_name,
                            "arguments": arguments,
                            "result": tool_result,
                            "duration_ms": call_duration_ms,
                            "call_time": datetime.datetime.now(),
                        }
                    )

                except Exception as e:
                    call_end_time = time.time()
                    call_duration_ms = int((call_end_time - call_start_time) * 1000)

                    # Handle empty error messages, especially for TimeoutError
                    error_msg = str(e) or (
                        "[ERROR]: Tool execution timeout"
                        if isinstance(e, TimeoutError)
                        else f"Tool execution failed: {type(e).__name__}"
                    )

                    tool_calls_data.append(
                        {
                            "server_name": server_name,
                            "tool_name": tool_name,
                            "arguments": arguments,
                            "error": error_msg,
                            "duration_ms": call_duration_ms,
                            "call_time": datetime.datetime.now(),
                        }
                    )
                    tool_result = {
                        "error": f"Tool call failed: {error_msg}",
                        "server_name": server_name,
                        "tool_name": tool_name,
                    }

                tool_result_for_llm = self.output_formatter.format_tool_result_for_user(
                    tool_result
                )
                logger.debug(f"Tool result: {tool_result}")

                all_tool_results_content_with_id.append((call_id, tool_result_for_llm))

            if len(tool_calls) > 1 and len(tool_calls[1]) > 0:
                tool_result = {
                    "result": f"Your tool call format was incorrect, and the tool invocation failed, error_message: {tool_calls[1][0]['error']}; please review it carefully and try calling again.",
                    "server_name": "re-think",
                    "tool_name": "re-think",
                }
                tool_calls_data.append(
                    {
                        "server_name": "",
                        "tool_name": "",
                        "arguments": "",
                        "result": tool_result,
                        "duration_ms": 0,
                        "call_time": datetime.datetime.now(),
                    }
                )
                tool_result_for_llm = self.output_formatter.format_tool_result_for_user(
                    tool_result
                )
                all_tool_results_content_with_id.append(("FAILED", tool_result_for_llm))

            message_history = self.sub_agent_llm_client.update_message_history(
                message_history, all_tool_results_content_with_id, tool_calls_exceeded
            )

        # Continue execution
        logger.debug(
            f"\n=== Sub Agent {sub_agent_name} Completed ({turn_count} turns) ==="
        )

        # Record browser agent loop end
        if turn_count >= max_turns:
            if (
                not task_failed
            ):  # If not yet marked as failed and due to turn limit exceeded
                task_failed = True
            self.task_log.log_step(
                "sub_agent_max_turns_reached",
                f"Sub agent {sub_agent_name} reached maximum turns ({max_turns})",
                "warning",
            )

        else:
            self.task_log.log_step(
                "sub_agent_loop_completed",
                f"Sub agent {sub_agent_name} loop completed after {turn_count} turns",
            )

        # Final summary - following main agent process
        self.task_log.log_step(
            "sub_agent_final_summary",
            f"Generating sub agent {sub_agent_name} final summary",
        )

        # Use context limit retry logic to generate final summary
        final_answer_text = await self._handle_summary_with_context_limit_retry(
            system_prompt,
            sub_agent_prompt_instance,
            message_history,
            tool_definitions,
            f"Sub agent {sub_agent_name} final summary",
            task_description,
            task_failed,
            agent_type=sub_agent_name,
        )

        if final_answer_text:
            self.task_log.log_step(
                "sub_agent_final_answer",
                f"Sub agent {sub_agent_name} final answer generated successfully",
            )

        else:
            final_answer_text = (
                f"No final answer generated by sub agent {sub_agent_name}."
            )
            self.task_log.log_step(
                "sub_agent_final_answer",
                f"Failed to generate sub agent {sub_agent_name} final answer",
                "failed",
            )

        logger.debug(f"Sub Agent {sub_agent_name} Final Answer: {final_answer_text}")

        self.task_log.sub_agent_message_history_sessions[
            self.task_log.current_sub_agent_session_id
        ] = {"system_prompt": system_prompt, "message_history": message_history}  # type: ignore
        self.task_log.save()

        # Record sub-agent cumulative usage
        usage_log = self.sub_agent_llm_client.get_usage_log()
        self.task_log.log_step(
            "usage_calculation",
            usage_log,
            metadata={"session_id": self.task_log.current_sub_agent_session_id},
        )

        self.task_log.end_sub_agent_session(sub_agent_name)
        self.task_log.log_step(
            "sub_agent_completed", f"Sub agent {sub_agent_name} completed", "info"
        )

        # Return final answer instead of dialogue log, so main agent can use directly
        return final_answer_text

    async def run_main_agent(
        self,
        task_description,
        task_file_name=None,
        task_id="default_task",
        metadata: dict[str, Any] | None = None,
    ):
        """
        Execute the main end-to-end task.
        """
        keep_tool_result = int(self.cfg.main_agent.keep_tool_result)

        logger.debug(f"\n{'=' * 20} Starting Task: {task_id} {'=' * 20}")
        logger.debug(f"Task Description: {task_description}")
        if task_file_name:
            logger.debug(f"Associated File: {task_file_name}")

        # Reset main agent usage stats for independent tracking
        self.llm_client.reset_usage_stats()

        # 1. Process input
        initial_user_content, task_description = process_input(
            task_description, task_file_name
        )

        task_guidence = ""
        if self.cfg.main_agent.get("generic_task_guidance_enabled", True):
            task_guidence = """

Your task is to comprehensively address the question by actively collecting detailed information from the web, and generating a thorough, transparent report. Your goal is NOT to rush a single definitive answer or conclusion, but rather to gather complete information and present ALL plausible candidate answers you find, accompanied by clearly documented supporting evidence, reasoning steps, uncertainties, and explicit intermediate findings.

User does not intend to set traps or create confusion on purpose. Handle the task using the most common, reasonable, and straightforward interpretation, and do not overthink or focus on rare or far-fetched interpretations.

Important considerations:
- Collect comprehensive information from reliable sources to understand all aspects of the question.
- Present every possible candidate answer identified during your information gathering, regardless of uncertainty, ambiguity, or incomplete verification. Avoid premature conclusions or omission of any discovered possibility.
- Explicitly document detailed facts, evidence, and reasoning steps supporting each candidate answer, carefully preserving intermediate analysis results.
- Clearly flag and retain any uncertainties, conflicting interpretations, or alternative understandings identified during information gathering. Do not arbitrarily discard or resolve these issues on your own.
- If the question's explicit instructions (e.g., numeric precision, formatting, specific requirements) appear inconsistent, unclear, erroneous, or potentially mismatched with general guidelines or provided examples, explicitly record and clearly present all plausible interpretations and corresponding candidate answers.  

Recognize that the original task description might itself contain mistakes, imprecision, inaccuracies, or conflicts introduced unintentionally by the user due to carelessness, misunderstanding, or limited expertise. Do NOT try to second-guess or "correct" these instructions internally; instead, transparently present findings according to every plausible interpretation.

Your objective is maximum completeness, transparency, and detailed documentation to empower the user to judge and select their preferred answer independently. Even if uncertain, explicitly documenting the existence of possible answers significantly enhances the user's experience, ensuring no plausible solution is irreversibly omitted due to early misunderstanding or premature filtering.
"""

        # Add Chinese-specific guidance if enabled
        if task_guidence and self.chinese_context:
            task_guidence += """

## 中文任务处理指导

如果任务涉及中文语境，请遵循以下指导：

- **信息收集策略**：使用中文关键词进行网络搜索，优先浏览中文网页，以获取更准确和全面的中文资源
- **思考过程**：所有分析、推理、判断等思考过程都应使用中文表达，保持语义的一致性
- **候选答案收集**：对于中文问题，收集所有可能的中文答案选项，包括不同的表达方式和格式
- **证据文档化**：保持中文资源的原始格式，避免不必要的翻译或改写，确保信息的准确性
- **不确定性标注**：使用中文清晰地标记任何不确定性、冲突信息或需要进一步验证的内容
- **结果组织**：以中文组织和呈现最终报告，使用恰当的中文术语和表达习惯
- **过程透明化**：所有步骤描述、状态更新、中间结果等都应使用中文，确保用户理解
"""

        if task_guidence:
            initial_user_content[0]["text"] += task_guidence

        hint_notes = ""  # Initialize hint_notes
        if self.cfg.main_agent.input_process.hint_generation:
            # Execute hint generation
            try:
                hint_content = await extract_hints(
                    task_description,
                    self.cfg.main_agent.openai_api_key,
                    self.chinese_context,
                    self.add_message_id,
                    self.cfg.main_agent.input_process.get(
                        "hint_llm_base_url", "https://api.openai.com/v1"
                    ),
                )
                hint_notes = (
                    "\n\nBefore you begin, please review the following preliminary notes highlighting subtle or easily misunderstood points in the question, which might help you avoid common pitfalls during your analysis (for reference only; these may not be exhaustive):\n\n"
                    + hint_content
                )

                # Update initial user content
                original_text = initial_user_content[0]["text"]
                initial_user_content[0]["text"] = original_text + hint_notes
            except Exception as e:
                logger.error(f"Hint generation failed after retries: {str(e)}")
                self.task_log.log_step(
                    step_name="hint_generation",
                    message=f"[ERROR] Hint generation failed: {str(e)}",
                    status="failed",
                )
                hint_notes = ""  # Continue execution but without hints

        # Memory/skill pre-task injection (passive retrieval)
        memory_store = None
        if self.cfg.get("memory") and self.cfg.memory.get("enabled", False):
            try:
                memory_store, skill_lib = get_memory_components(self.cfg)
                if memory_store:
                    memory_block = build_memory_context_block(
                        task_description=task_description,
                        store=memory_store,
                        skill_lib=skill_lib,
                        inject_top_k=int(self.cfg.memory.get("inject_top_k", 3)),
                        skill_top_k=int(self.cfg.memory.get("skill_top_k", 2)),
                        # inject_enabled=false gives a skill-only ablation arm
                        # (memory components loaded, retrieval not injected).
                        memory_enabled=bool(self.cfg.memory.get("inject_enabled", True)),
                        skill_enabled=bool(
                            self.cfg.memory.get("skill_enabled", True) and skill_lib
                        ),
                        skill_preview_min_score=float(
                            self.cfg.memory.get("skill_preview_min_score", 0.0)
                        ),
                        list_other_skills=bool(
                            self.cfg.memory.get("skill_list_others", True)
                        ),
                        calibration_enabled=bool(
                            self.cfg.memory.get("calibration_enabled", False)
                        ),
                        calibration_mode=str(
                            self.cfg.memory.get("calibration_mode", "")
                        ),
                        calibration_min_samples=int(
                            self.cfg.memory.get("calibration_min_samples", 16)
                        ),
                        rolling_status_enabled=(
                            str(self.cfg.memory.get("reflection_mode", "")) == "rolling"
                        ),
                        rolling_min_samples=int(
                            self.cfg.memory.get("rolling_min_samples", 64)
                        ),
                        rolling_min_months=int(
                            self.cfg.memory.get("rolling_min_history_months", 6)
                        ),
                        rank_factor_enabled=(
                            str(self.cfg.memory.get("reflection_mode", ""))
                            in ("rank_factor", "trader")
                        ),
                        rank_factor_min_months=int(
                            self.cfg.memory.get("rank_factor_min_months", 3)
                        ),
                        rank_factor_fdr_q=float(
                            self.cfg.memory.get("rank_factor_fdr_q", 0.10)
                        ),
                        rank_factor_status_enabled=bool(
                            self.cfg.memory.get("rank_factor_status_enabled", False)
                        ),
                        trader_episode_enabled=(
                            str(self.cfg.memory.get("reflection_mode", ""))
                            == "trader"
                        ),
                        trader_episode_max=int(
                            self.cfg.memory.get("trader_episode_max", 3)
                        ),
                    )
                    if memory_block:
                        original_text = initial_user_content[0]["text"]
                        initial_user_content[0]["text"] = original_text + memory_block
                        self.task_log.log_step(
                            "memory_injection",
                            "Injected retrieved memory/skill context into initial prompt",
                            "info",
                        )
            except Exception as e:
                logger.error(f"Memory injection failed: {e}")
                self.task_log.log_step(
                    step_name="memory_injection",
                    message=f"[ERROR] Memory injection failed: {str(e)}",
                    status="failed",
                )

        logger.info("Initial user input content: %s", initial_user_content)
        message_history = [{"role": "user", "content": initial_user_content}]

        # 2. Get tool definitions
        tool_definitions = await self.main_agent_tool_manager.get_all_tool_definitions()
        if self.cfg.get("memory") and self.cfg.memory.get("enabled", False):
            search_mode = str(
                self.cfg.memory.get("expose_memory_search", "always")
            ).lower()
            expose_search = search_mode == "always"
            if search_mode == "auto" and memory_store:
                expose_search = memory_store.has_vector_memories(
                    before_month="",
                    before_date=task_as_of_date(task_description),
                )
            expose_save = bool(self.cfg.memory.get("expose_memory_save", True))
            tool_definitions = _filter_memskill_tool_definitions(
                tool_definitions,
                expose_memory_search=expose_search,
                expose_memory_save=expose_save,
            )
        if self.cfg.sub_agents is not None and self.cfg.sub_agents:
            tool_definitions += expose_sub_agents_as_tools(self.cfg.sub_agents)
        if not tool_definitions:
            logger.debug(
                "Warning: No tool definitions found. LLM cannot use any tools."
            )

        self.task_log.log_step("get_main_tool_definitions", f"{tool_definitions}")

        # 3. Generate system prompt
        main_agent_prompt_instance = _load_agent_prompt_class(
            self.cfg.main_agent.prompt_class
        )
        system_prompt = (
            main_agent_prompt_instance.generate_system_prompt_with_mcp_tools(
                mcp_servers=tool_definitions,
                chinese_context=self.chinese_context,
            )
        )

        # 4. Main loop: LLM <-> Tools
        max_turns = self.cfg.main_agent.max_turns
        if max_turns < 0:
            max_turns = sys.maxsize
        turn_count = 0
        trader_repair_count = 0
        task_failed = False  # Track whether task failed
        ashare_tools_seen: set[str] = set()
        feature_evidence_attempted = False
        assistant_response_text = ""
        core_satellite_intent_text = ""

        def canonicalize_core_satellite_terminal(answer: str) -> str:
            nonlocal core_satellite_intent_text
            core_satellite_intent_text = answer
            canonical_text, canonical = _canonicalize_core_satellite_response(
                answer,
                metadata,
            )
            selection = dict((metadata or {}).get("satellite_selection", {}))
            self.task_log.log_step(
                "trader_core_satellite_canonicalized",
                canonical.diagnostic,
                "success",
                metadata=selection,
            )
            return canonical_text

        while turn_count < max_turns:
            turn_count += 1
            logger.debug(f"\n--- Main Agent Turn {turn_count} ---")
            self.task_log.save()

            # Use unified LLM call handling
            (
                assistant_response_text,
                should_break,
                tool_calls,
            ) = await self._handle_llm_call_with_logging(
                system_prompt,
                message_history,
                tool_definitions,
                turn_count,
                f"Main agent turn {turn_count}",
                keep_tool_result=keep_tool_result,
                agent_type="main",
            )

            # Handle LLM response
            if assistant_response_text:
                if should_break:
                    if (
                        _ashare_trader_core_satellite_enabled(metadata)
                        and not _ashare_trader_mandatory_tool_error(
                            metadata,
                            ashare_tools_seen,
                        )
                    ):
                        assistant_response_text = (
                            canonicalize_core_satellite_terminal(
                                assistant_response_text
                            )
                        )
                    break
            else:
                # LLM call failed, mark task as failed and end current turn
                if tool_calls == "context_limit":
                    # Context limit exceeded situation
                    self.task_log.log_step(
                        "main_agent_context_limit_reached",
                        "Main agent context limit reached, jumping to summary",
                        "warning",
                    )
                else:
                    # Other LLM call failure situations
                    self.task_log.log_step(
                        step_name="main_agent",
                        message="LLM call failed",
                        status="failed",
                    )
                task_failed = True  # Mark task as failed
                break

            if (
                tool_calls is None
                or len(tool_calls) < 2
                or (len(tool_calls[0]) == 0 and len(tool_calls[1]) == 0)
            ):
                if _ashare_trader_core_satellite_enabled(metadata):
                    portfolio_error = _ashare_trader_mandatory_tool_error(
                        metadata,
                        ashare_tools_seen,
                    )
                    if not portfolio_error:
                        assistant_response_text = (
                            canonicalize_core_satellite_terminal(
                                assistant_response_text or ""
                            )
                        )
                        logger.debug(
                            "System canonicalized the core-satellite allocation."
                        )
                        break
                else:
                    portfolio_error = _ashare_trader_terminal_error(
                        assistant_response_text or "",
                        metadata,
                        ashare_tools_seen,
                    )
                if (
                    portfolio_error
                    and trader_repair_count < MAX_TRADER_PORTFOLIO_REPAIRS
                ):
                    trader_repair_count += 1
                    repair_prompt = (
                        "你的终局组合未通过确定性 A 股硬约束复核："
                        f"{portfolio_error}。请保留已有点时分析，只完成错误要求的"
                        "必要工具调用或仓位修正，不扩展其他搜索；随后重新给出正文"
                        "和最后一行 boxed 组合。"
                    )
                    message_history.append(
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": repair_prompt}],
                        }
                    )
                    self.task_log.log_step(
                        "trader_portfolio_repair",
                        "Repair "
                        f"{trader_repair_count}/{MAX_TRADER_PORTFOLIO_REPAIRS} "
                        f"requested: {portfolio_error}",
                        "warning",
                    )
                    continue
                # No tool calls, consider as final answer
                logger.debug("LLM did not request tool use, process ends.")
                break  # Exit loop

            # 7. Execute tool calls (in sequence)
            tool_calls_data = []
            all_tool_results_content_with_id = []

            # Get maximum tool calls per turn from configuration
            max_tool_calls = self.cfg.main_agent.max_tool_calls_per_turn
            tool_calls_exceeded = (
                len(tool_calls) > 0 and len(tool_calls[0]) > max_tool_calls
            )
            if tool_calls_exceeded:
                logger.warning(
                    f"[ERROR] Single turn tool call count too high ({len(tool_calls[0])} calls), only processing first {max_tool_calls}"
                )

            for call in tool_calls[0][:max_tool_calls]:
                server_name = call["server_name"]
                tool_name = call["tool_name"]
                arguments = call["arguments"]
                call_id = call["id"]

                call_start_time = time.time()
                try:
                    _validate_ashare_trader_tool_call(
                        tool_name,
                        arguments,
                        metadata,
                    )
                    if server_name.startswith("agent-"):
                        sub_agent_result = await self.run_sub_agent(
                            server_name, str(arguments), keep_tool_result
                        )
                        tool_result = {
                            "server_name": server_name,
                            "tool_name": tool_name,
                            "result": sub_agent_result,
                        }
                    else:
                        tool_result = (
                            await self.main_agent_tool_manager.execute_tool_call(
                                server_name=server_name,
                                tool_name=tool_name,
                                arguments=arguments,
                            )
                        )
                        if (
                            server_name == "tool-ashare"
                            and not (
                                isinstance(tool_result, dict)
                                and tool_result.get("error")
                            )
                        ):
                            ashare_tools_seen.add(tool_name)

                    call_end_time = time.time()
                    call_duration_ms = int((call_end_time - call_start_time) * 1000)

                    tool_calls_data.append(
                        {
                            "server_name": server_name,
                            "tool_name": tool_name,
                            "arguments": arguments,
                            "result": tool_result,
                            "duration_ms": call_duration_ms,
                            "call_time": datetime.datetime.now(),
                        }
                    )

                except Exception as e:
                    call_end_time = time.time()
                    call_duration_ms = int((call_end_time - call_start_time) * 1000)

                    # Handle empty error messages, especially for TimeoutError
                    error_msg = str(e) or (
                        "[ERROR]: Tool execution timeout"
                        if isinstance(e, TimeoutError)
                        else f"Tool execution failed: {type(e).__name__}"
                    )

                    tool_calls_data.append(
                        {
                            "server_name": server_name,
                            "tool_name": tool_name,
                            "arguments": arguments,
                            "error": error_msg,
                            "duration_ms": call_duration_ms,
                            "call_time": datetime.datetime.now(),
                        }
                    )
                    tool_result = {
                        "server_name": server_name,
                        "tool_name": tool_name,
                        "error": error_msg,
                    }

                # Format result for LLM feedback (more concise)
                tool_result_for_llm = self.output_formatter.format_tool_result_for_user(
                    tool_result
                )
                # all_tool_results_content.extend(tool_result_for_llm)  # Collect all tool results
                all_tool_results_content_with_id.append((call_id, tool_result_for_llm))

            if len(tool_calls) > 1 and len(tool_calls[1]) > 0:
                tool_result = {
                    "result": f"Your tool call format was incorrect, and the tool invocation failed, error_message: {tool_calls[1][0]['error']}; please review it carefully and try calling again.",
                    "server_name": "re-think",
                    "tool_name": "re-think",
                }
                tool_calls_data.append(
                    {
                        "server_name": "",
                        "tool_name": "",
                        "arguments": "",
                        "result": tool_result,
                        "duration_ms": 0,
                        "call_time": datetime.datetime.now(),
                    }
                )
                tool_result_for_llm = self.output_formatter.format_tool_result_for_user(
                    tool_result
                )
                all_tool_results_content_with_id.append(("FAILED", tool_result_for_llm))

            # Update message history with tool calls data (llm client specific)
            message_history = self.llm_client.update_message_history(
                message_history, all_tool_results_content_with_id, tool_calls_exceeded
            )

            if (
                not feature_evidence_attempted
                and memory_store
                and self.cfg.get("memory")
                and self.cfg.memory.get("feature_evidence_enabled", False)
                and metadata
            ):
                required_tools = set(
                    self.cfg.memory.get(
                        "feature_evidence_required_tools",
                        [
                            "ashare_price_history",
                            "ashare_index_history",
                            "ashare_valuation",
                            "ashare_ml_signal",
                        ],
                    )
                )
                if required_tools.issubset(ashare_tools_seen):
                    feature_evidence_attempted = True
                    try:
                        data_dir = os.path.join(
                            str(self.cfg.get("data_dir", "data")), "ashare"
                        )
                        evidence_block, evidence_audit = build_feature_evidence_block(
                            memory_store,
                            entry_date=str(
                                metadata.get("entry_date")
                                or task_as_of_date(task_description)
                            ),
                            ts_code=str(metadata.get("ts_code", "")),
                            stock_name=str(metadata.get("stock_name", "")),
                            data_dir=data_dir,
                            k=int(self.cfg.memory.get("feature_evidence_k", 12)),
                            min_neighbors=int(
                                self.cfg.memory.get("feature_evidence_min_neighbors", 8)
                            ),
                            min_common_features=int(
                                self.cfg.memory.get(
                                    "feature_evidence_min_common_features", 3
                                )
                            ),
                        )
                        min_pool = int(
                            self.cfg.memory.get("feature_evidence_min_pool", 32)
                        )
                        if evidence_audit.get("eligible_samples", 0) < min_pool:
                            evidence_block = ""
                            evidence_audit["withheld_reason"] = (
                                f"eligible sample pool below {min_pool}"
                            )
                        if evidence_block:
                            message_history.append(
                                {"role": "user", "content": evidence_block}
                            )
                            self.task_log.log_step(
                                "feature_evidence_injection",
                                "Injected point-in-time feature-conditioned evidence",
                                "info",
                                metadata=evidence_audit,
                            )
                        else:
                            self.task_log.log_step(
                                "feature_evidence_withheld",
                                "No statistically reliable feature-conditioned evidence",
                                "info",
                                metadata=evidence_audit,
                            )
                    except Exception as e:
                        logger.error(f"Feature evidence injection failed: {e}")
                        self.task_log.log_step(
                            "feature_evidence_injection",
                            f"[ERROR] Feature evidence injection failed: {e}",
                            "failed",
                        )

        # Record main loop end
        if turn_count >= max_turns:
            if (
                not task_failed
            ):  # If not yet marked as failed and due to turn limit exceeded
                task_failed = True
            self.task_log.log_step(
                "max_turns_reached",
                f"Reached maximum turns ({max_turns})",
                "warning",
            )

        else:
            self.task_log.log_step(
                "main_loop_completed", f"Main loop completed after {turn_count} turns"
            )

        # Final summary
        self.task_log.log_step("final_summary", "Generating final summary")

        reuse_terminal_response = bool(
            self.cfg.main_agent.output_process.get("reuse_terminal_response", False)
        )
        terminal_has_boxed_answer = (
            bool(assistant_response_text)
            and "\\boxed{" in assistant_response_text.replace(" ", "")
            and not _ashare_trader_terminal_error(
                assistant_response_text,
                metadata,
                ashare_tools_seen,
            )
        )
        if reuse_terminal_response and not task_failed and terminal_has_boxed_answer:
            final_answer_text = assistant_response_text
            self.task_log.log_step(
                "final_summary_reused",
                "Reused terminal valid boxed response without another LLM call",
            )
        else:
            # Use context limit retry logic to generate final summary
            final_answer_text = await self._handle_summary_with_context_limit_retry(
                system_prompt,
                main_agent_prompt_instance,
                message_history,
                tool_definitions,
                "Final summary generation",
                task_description,
                task_failed,
                agent_type="main",
                task_guidence=task_guidence,
            )

        # Handle response result
        if final_answer_text:
            self.task_log.log_step(
                "final_answer", "Final answer extracted successfully"
            )

            # Log the final answer
            self.task_log.log_step(
                "final_answer_content", f"Final answer content: {final_answer_text}"
            )

            # Use LLM to extract final answer
            extracted_answer = ""
            if self.cfg.main_agent.output_process.final_answer_extraction:
                # Execute final answer extraction
                try:
                    # For browsecomp-zh, we use another Chinese prompt to extract the final answer
                    if "browsecomp-zh" in self.cfg.benchmark.name:
                        extracted_answer = await extract_browsecomp_zh_final_answer(
                            task_description,
                            final_answer_text,
                            self.cfg.main_agent.openai_api_key,
                            self.cfg.main_agent.output_process.get(
                                "final_answer_llm_base_url", "https://api.openai.com/v1"
                            ),
                        )

                        # Disguise LLM extracted answer as assistant returned result and add to message history
                        assistant_extracted_message = {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"LLM extracted final answer:\n{extracted_answer}",
                                }
                            ],
                        }
                        message_history.append(assistant_extracted_message)

                        # LLM answer as final result
                        final_answer_text = extracted_answer
                    else:
                        extracted_answer = await extract_gaia_final_answer(
                            task_description,
                            final_answer_text,
                            self.cfg.main_agent.openai_api_key,
                            self.chinese_context,
                            self.cfg.main_agent.output_process.get(
                                "final_answer_llm_base_url", "https://api.openai.com/v1"
                            ),
                        )

                        # Disguise LLM extracted answer as assistant returned result and add to message history
                        assistant_extracted_message = {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"LLM extracted final answer:\n{extracted_answer}",
                                }
                            ],
                        }
                        message_history.append(assistant_extracted_message)

                        # Concatenate original summary and LLM answer as final result
                        final_answer_text = f"{final_answer_text}\n\nLLM Extracted Answer:\n{extracted_answer}"

                except Exception as e:
                    logger.error(
                        f"Final answer extraction failed after retries: {str(e)}"
                    )
                    self.task_log.log_step(
                        step_name="final_answer_extraction",
                        message=f"[ERROR] Final answer extraction failed: {str(e)}",
                        status="failed",
                    )
                    # Continue using original final_answer_text

            else:
                # to process when final_answer_extraction is false
                # leave it here to be more clear
                final_answer_text = final_answer_text

        else:
            final_answer_text = "No final answer generated."
            self.task_log.log_step(
                "final_answer", "Failed to extract final answer", "failed"
            )

        logger.debug(f"LLM Final Answer: {final_answer_text}")

        # Save final message history (including LLM processing results)
        self.task_log.main_agent_message_history = {
            "system_prompt": system_prompt,
            "message_history": message_history,
        }
        self.task_log.save()

        # Format and return final output
        self.task_log.log_step("format_output", "Formatting final output")
        final_summary, final_boxed_answer = (
            self.output_formatter.format_final_summary_and_log(
                final_answer_text, self.llm_client
            )
        )
        mandatory_tool_error = _ashare_trader_mandatory_tool_error(
            metadata,
            ashare_tools_seen,
        )
        if (
            _ashare_trader_core_satellite_enabled(metadata)
            and not mandatory_tool_error
        ):
            canonicalization_source = (
                core_satellite_intent_text or final_answer_text
            )
            _, canonical = _canonicalize_core_satellite_response(
                canonicalization_source,
                metadata,
            )
            final_answer_text = _replace_or_append_final_boxed(
                final_answer_text,
                canonical.canonical_boxed_answer,
            )
            final_summary, _ = self.output_formatter.format_final_summary_and_log(
                final_answer_text,
                self.llm_client,
            )
            final_boxed_answer = canonical.canonical_boxed_answer
            self.task_log.log_step(
                "trader_core_satellite_finalized",
                "Forced the final boxed answer to the canonical allocation",
                "success",
                metadata=dict((metadata or {}).get("satellite_selection", {})),
            )
        portfolio_error = _ashare_trader_terminal_error(
            final_boxed_answer,
            metadata,
            ashare_tools_seen,
        )
        if portfolio_error:
            self.task_log.log_step(
                "invalid_trader_portfolio",
                f"Rejected final portfolio: {portfolio_error}",
                "failed",
            )
            final_summary += (
                "\n\n"
                f"[INVALID ASHARE-TRADER PORTFOLIO] {portfolio_error}"
            )
            final_boxed_answer = ""

        logger.debug(f"\n{'=' * 20} Task {task_id} Finished {'=' * 20}")
        self.task_log.log_step(
            "task_completed", f"Main agent task {task_id} completed successfully"
        )

        # Record main agent cumulative usage
        usage_log = self.llm_client.get_usage_log()
        self.task_log.log_step(
            "usage_calculation",
            usage_log,
            metadata={"session_id": "main_agent"},
        )

        if "browsecomp-zh" in self.cfg.benchmark.name:
            return final_summary, final_summary
        else:
            return final_summary, final_boxed_answer
