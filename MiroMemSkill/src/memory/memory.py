# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

"""Domain memory facade over official Mem0 plus statistical reflection.

Production uses ``mem0ai.Memory`` for extraction, consolidation, vector CRUD,
and SQLite history. The legacy JSONL path remains available only for isolated
tests and historical compatibility.

Retrieval applies metadata filters:
  - before_date: labels become visible only on/after their actual exit date;
    before_month remains a conservative fallback for legacy records.
  - functional-stance quota: a lesson's direction pressure is derived from
    metadata (predicted_direction x judge_result), not keyword scanning — a
    counter-lesson from a wrong 跑输 prediction pushes bullish, so it counts
    against the bullish quota.

The A-share v3 path does not ask an LLM to discover rules.  It keeps an
idempotent feature/outcome ledger and synchronizes only expanding-window rules
that pass temporal validation in ``rolling_reflection``.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from datetime import date
from typing import Any, Mapping, Optional, Sequence

import requests
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from src.memory.prompts import (
    EXTRACTION_CORRECT_INSTRUCTIONS,
    EXTRACTION_INCORRECT_INSTRUCTIONS,
    EXTRACTION_PROMPT,
    MONTHLY_REFLECTION_PROMPT,
    UPDATE_PROMPT,
)
from src.memory.rolling_reflection import (
    RollingRuleConfig,
    ValidatedRule,
    mine_rolling_rules,
    normalize_date,
    normalize_month,
)
from src.memory.vector_store import MemoryRecord
from src.utils.env_loader import load_project_env

# Below this similarity the update phase is skipped and the candidate is
# ADDed directly: nothing in the bank is close enough to consolidate with.
_CONSOLIDATION_SIM_THRESHOLD = 0.45


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, requests.exceptions.HTTPError):
        status = exc.response.status_code if exc.response is not None else 0
        return status == 429 or status >= 500
    return isinstance(exc, (requests.exceptions.ConnectionError, requests.exceptions.Timeout))


def extract_direction(answer: str) -> str:
    """Pull the predicted direction (跑赢/跑输) out of a final answer."""
    text = answer or ""
    if "跑赢" in text and "跑输" not in text:
        return "跑赢"
    if "跑输" in text and "跑赢" not in text:
        return "跑输"
    m = re.search(r"\\boxed\{(跑赢|跑输)\}", text)
    return m.group(1) if m else ""


def month_available_after(month: str) -> str:
    """First calendar day after YYYY-MM, used for month-scoped notes."""
    normalized = normalize_month(month)
    if not normalized:
        return ""
    year, month_number = (int(part) for part in normalized.split("-"))
    if month_number == 12:
        return date(year + 1, 1, 1).strftime("%Y%m%d")
    return date(year, month_number + 1, 1).strftime("%Y%m%d")


def functional_stance(predicted_direction: str, judge_result: str) -> str:
    """Direction a lesson pushes FUTURE predictions toward.

    Correct prediction  -> lesson reinforces that direction.
    Incorrect prediction -> counter-lesson pushes the opposite direction
    (e.g. "predicting 跑输 on weak momentum failed" nudges bullish).
    """
    if predicted_direction not in ("跑赢", "跑输"):
        return "neutral"
    reinforcing = judge_result == "CORRECT"
    bullish_pred = predicted_direction == "跑赢"
    return "bullish" if bullish_pred == reinforcing else "bearish"


def parse_task_month(task_id: str) -> str:
    """ashare task ids end with the entry date: ashare_300012_2025-04-01 -> 2025-04."""
    m = re.search(r"(\d{4})-(\d{2})-\d{2}$", task_id or "")
    return f"{m.group(1)}-{m.group(2)}" if m else ""


def summarize_trajectory(log_data: dict[str, Any], max_chars: int = 3000) -> str:
    parts: list[str] = []
    history = (log_data or {}).get("main_agent_message_history", {}).get("message_history", [])
    for msg in history[-12:]:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if isinstance(content, list):
            text = " ".join(
                block.get("text", "") for block in content if isinstance(block, dict)
            )
        else:
            text = str(content)
        text = re.sub(r"\s+", " ", text).strip()[:400]
        if text:
            parts.append(f"[{role}] {text}")
    return "\n".join(parts)[:max_chars]


def _fact_number(value: Any, digits: int = 10) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, digits) if math.isfinite(number) else None


def _fact_codes(value: Any) -> list[str]:
    if isinstance(value, str):
        raw_codes: Sequence[Any] = [value]
    elif isinstance(value, Mapping):
        raw_codes = list(value)
    elif isinstance(value, Sequence):
        raw_codes = value
    else:
        return []
    seen: set[str] = set()
    result: list[str] = []
    for raw_code in raw_codes:
        code = str(raw_code or "").strip().upper()
        if code and code not in seen:
            seen.add(code)
            result.append(code)
    return result


def _first_field(raw: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in raw:
            return raw.get(key)
    return None


def _fact_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        if normalized in {
            "1",
            "true",
            "yes",
            "used",
            "fallback",
            "deterministic",
            "deterministic_fallback",
            "model_top",
        }:
            return True
        if normalized in {
            "0",
            "false",
            "no",
            "unused",
            "none",
            "explicit",
            "model_selected",
        }:
            return False
    return None


def _candidate_fact_rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, Mapping):
        if any(key in value for key in ("code", "ts_code", "symbol")):
            raw_rows: list[Any] = [value]
        else:
            raw_rows = []
            for code, item in value.items():
                if isinstance(item, Mapping):
                    row = dict(item)
                    row.setdefault("code", code)
                else:
                    row = {"code": code, "model_score": item}
                raw_rows.append(row)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        raw_rows = list(value)
    else:
        return []

    rows: list[dict[str, Any]] = []
    for position, item in enumerate(raw_rows, 1):
        if isinstance(item, str):
            raw = {"code": item}
        elif isinstance(item, Mapping):
            raw = item
        else:
            continue
        code = str(
            _first_field(raw, "code", "ts_code", "symbol") or ""
        ).strip().upper()
        realized_stock_return = _fact_number(
            _first_field(
                raw,
                "realized_stock_return",
                "stock_return",
                "realized_return",
            )
        )
        realized_excess = _fact_number(
            _first_field(
                raw,
                "realized_excess_vs_csi300",
                "realized_excess_return",
                "excess_return",
            )
        )
        if not code or (realized_stock_return is None and realized_excess is None):
            continue
        candidate_rank_raw = _fact_number(
            _first_field(raw, "candidate_rank", "candidate_order", "eligible_rank")
        )
        candidate_rank = (
            int(candidate_rank_raw)
            if candidate_rank_raw is not None
            and candidate_rank_raw > 0
            and candidate_rank_raw.is_integer()
            else position
        )
        row = {
            "code": code,
            "candidate_rank": candidate_rank,
        }
        if realized_stock_return is not None:
            row["realized_stock_return"] = realized_stock_return
        if realized_excess is not None:
            row["realized_excess_vs_csi300"] = realized_excess
        model_rank = _fact_number(
            _first_field(raw, "model_rank", "rank", "prediction_rank", "ml_rank")
        )
        if model_rank is not None and model_rank > 0:
            row["model_rank"] = (
                int(model_rank) if model_rank.is_integer() else model_rank
            )
        model_score = _fact_number(
            _first_field(
                raw,
                "model_score",
                "score",
                "prediction_score",
                "predicted_excess_return",
                "ml_score",
            ),
            digits=12,
        )
        if model_score is not None:
            row["model_score"] = model_score
        selected = _fact_bool(
            _first_field(raw, "selected", "actual_selected")
        )
        deterministic_selected = _fact_bool(raw.get("deterministic_selected"))
        if selected is not None:
            row["selected"] = selected
        if deterministic_selected is not None:
            row["deterministic_selected"] = deterministic_selected
        for target, aliases in {
            "actual_weight": ("actual_weight",),
            "deterministic_weight": (
                "deterministic_weight",
                "model_weight",
                "counterfactual_weight",
            ),
            "weighted_return_contribution": (
                "weighted_return_contribution",
                "actual_weighted_return_contribution",
            ),
            "weighted_excess_contribution": (
                "weighted_excess_contribution",
                "actual_weighted_excess_contribution",
                "actual_weighted_contribution",
            ),
            "net_contribution": (
                "net_contribution",
                "actual_net_contribution",
            ),
            "deterministic_weighted_return_contribution": (
                "deterministic_weighted_return_contribution",
                "model_weighted_return_contribution",
            ),
            "deterministic_weighted_excess_contribution": (
                "deterministic_weighted_excess_contribution",
                "model_weighted_contribution",
            ),
            "deterministic_net_contribution": (
                "deterministic_net_contribution",
                "model_net_contribution",
            ),
        }.items():
            number = _fact_number(_first_field(raw, *aliases))
            if number is not None:
                row[target] = number
        rows.append(row)
    return rows


def _normalize_satellite_attribution(
    value: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Whitelist structured facts; never retain free-form model reasoning."""
    raw = dict(value or {})
    if not raw:
        return {}
    mode = str(raw.get("policy_mode") or raw.get("mode") or "").strip().lower()
    if mode and mode != "core_satellite":
        return {}

    actual_codes = _fact_codes(
        _first_field(
            raw,
            "selected_codes",
            "actual_satellite_codes",
            "actual_selected_satellites",
            "actual_satellites",
            "selected_satellites",
        )
    )
    deterministic_codes = _fact_codes(
        _first_field(
            raw,
            "deterministic_codes",
            "deterministic_model_satellite_codes",
            "deterministic_satellite_codes",
            "expected_satellite_codes",
            "model_top_satellites",
        )
    )
    if not actual_codes or not deterministic_codes:
        return {}

    raw_sleeve = _first_field(raw, "system_sleeve", "sleeve")
    sleeve = raw_sleeve if isinstance(raw_sleeve, Mapping) else {}
    regime = str(
        raw.get("regime")
        or raw.get("market_regime")
        or sleeve.get("regime")
        or ""
    ).strip().lower()
    if regime not in {"risk_on", "neutral", "defensive"}:
        return {}
    core_codes = _fact_codes(
        _first_field(raw, "top4_codes", "core_codes", "top4")
        or _first_field(sleeve, "core_codes", "top4")
    )
    core_weight = _fact_number(
        _first_field(raw, "core_weight", "core_total_weight")
        if _first_field(raw, "core_weight", "core_total_weight") is not None
        else _first_field(sleeve, "core_total_weight", "core_weight")
    )
    satellite_weight = _fact_number(
        _first_field(raw, "satellite_weight", "satellite_total_weight")
        if _first_field(raw, "satellite_weight", "satellite_total_weight")
        is not None
        else _first_field(
            sleeve,
            "satellite_total_weight",
            "satellite_weight",
        )
    )
    cash_weight = _fact_number(
        raw.get("cash_weight")
        if raw.get("cash_weight") is not None
        else _first_field(sleeve, "cash_weight", "cash")
    )
    if core_codes and (len(core_codes) != 4 or len(set(core_codes)) != 4):
        return {}
    if (
        core_weight is not None
        and satellite_weight is not None
        and cash_weight is not None
        and abs(core_weight + satellite_weight + cash_weight - 1.0) > 1e-6
    ):
        return {}

    candidates = _candidate_fact_rows(
        _first_field(
            raw,
            "candidate_facts",
            "candidate_attribution",
            "candidates",
            "satellite_candidates",
        )
    )
    candidate_codes = {row["code"] for row in candidates}
    selected_union = set(actual_codes) | set(deterministic_codes)
    if not candidates or not selected_union.issubset(candidate_codes):
        return {}
    actual_set = set(actual_codes)
    deterministic_set = set(deterministic_codes)
    for row in candidates:
        row["selected"] = row["code"] in actual_set
        row["deterministic_selected"] = row["code"] in deterministic_set
    candidates = [
        row for row in candidates if row["code"] in selected_union
    ]
    candidates.sort(key=lambda row: (row["candidate_rank"], row["code"]))

    number_aliases = {
        "satellite_gross_return": ("satellite_gross_return",),
        "satellite_excess_return": ("satellite_excess_return",),
        "satellite_net_return": ("satellite_net_return",),
        "satellite_weighted_return_contribution": (
            "satellite_weighted_return_contribution",
            "actual_satellite_weighted_return_contribution",
        ),
        "satellite_weighted_excess_contribution": (
            "satellite_weighted_excess_contribution",
            "actual_satellite_weighted_excess_contribution",
        ),
        "satellite_net_contribution": (
            "satellite_net_contribution",
            "actual_satellite_net_contribution",
        ),
        "core_weighted_return_contribution": (
            "core_weighted_return_contribution",
        ),
        "core_net_contribution": (
            "core_net_contribution",
            "actual_core_net_contribution",
        ),
        "deterministic_satellite_gross_return": (
            "deterministic_satellite_gross_return",
        ),
        "deterministic_satellite_excess_return": (
            "deterministic_satellite_excess_return",
        ),
        "deterministic_satellite_net_return": (
            "deterministic_satellite_net_return",
        ),
        "deterministic_satellite_weighted_return_contribution": (
            "deterministic_satellite_weighted_return_contribution",
        ),
        "deterministic_satellite_weighted_excess_contribution": (
            "deterministic_satellite_weighted_excess_contribution",
        ),
        "deterministic_satellite_net_contribution": (
            "deterministic_satellite_net_contribution",
        ),
        "actual_net_return": ("actual_net_return",),
        "deterministic_counterfactual_net_return": (
            "deterministic_counterfactual_net_return",
            "model_satellite_counterfactual_net_return",
            "deterministic_net_return",
            "counterfactual_net_return",
        ),
        "deterministic_counterfactual_total_cost": (
            "deterministic_counterfactual_total_cost",
            "model_satellite_counterfactual_total_cost",
        ),
        "actual_minus_deterministic_net_return": (
            "actual_minus_deterministic_net_return",
        ),
        "actual_minus_deterministic_pnl": (
            "actual_minus_deterministic_pnl",
        ),
        "pure_momentum_net_return": ("pure_momentum_net_return",),
        "actual_minus_pure_momentum_net_return": (
            "actual_minus_pure_momentum_net_return",
            "actual_minus_momentum_net_return",
        ),
        "actual_minus_pure_momentum_pnl": (
            "actual_minus_pure_momentum_pnl",
            "actual_minus_momentum_pnl",
        ),
    }
    numbers: dict[str, float] = {}
    for target, aliases in number_aliases.items():
        number = _fact_number(
            _first_field(raw, *aliases),
            digits=6
            if target.endswith("_pnl") or target.endswith("_total_cost")
            else 10,
        )
        if number is not None:
            numbers[target] = number

    required = {
        "satellite_net_contribution",
        "actual_net_return",
        "deterministic_counterfactual_net_return",
        "pure_momentum_net_return",
    }
    if not required.issubset(numbers):
        return {}
    numbers.setdefault(
        "core_net_contribution",
        numbers["actual_net_return"] - numbers["satellite_net_contribution"],
    )
    numbers.setdefault(
        "actual_minus_deterministic_net_return",
        numbers["actual_net_return"]
        - numbers["deterministic_counterfactual_net_return"],
    )
    numbers.setdefault(
        "actual_minus_pure_momentum_net_return",
        numbers["actual_net_return"] - numbers["pure_momentum_net_return"],
    )

    fallback = _fact_bool(
        _first_field(
            raw,
            "selection_fallback",
            "fallback_used",
            "deterministic_fallback_used",
        )
    )
    selection_source = str(
        _first_field(raw, "selection_source", "source") or ""
    ).strip().lower()
    if selection_source not in {"agent", "deterministic_fallback"}:
        selection_source = "unknown"
    if fallback is None and selection_source != "unknown":
        fallback = selection_source == "deterministic_fallback"
    candidate_signal_date = str(raw.get("candidate_signal_date") or "").strip()[:32]
    target_value = raw.get("target")
    target = (
        str(target_value).strip()[:120]
        if isinstance(target_value, (str, int, float, bool))
        else ""
    )

    normalized: dict[str, Any] = {
        "policy_mode": "core_satellite",
        "regime": regime,
        "top4_codes": core_codes,
        "selected_codes": actual_codes,
        "deterministic_codes": deterministic_codes,
        "selection_differs": set(actual_codes) != set(deterministic_codes),
        "selection_fallback": fallback,
        "selection_source": selection_source,
        "candidate_signal_date": candidate_signal_date,
        "target": target,
        "candidate_facts": candidates,
        **numbers,
    }
    if core_weight is not None:
        normalized["core_weight"] = core_weight
    if satellite_weight is not None:
        normalized["satellite_weight"] = satellite_weight
    if cash_weight is not None:
        normalized["cash_weight"] = cash_weight
    return normalized


def _attribution_version(metadata: Mapping[str, Any]) -> int:
    try:
        return int(metadata.get("attribution_version", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _satellite_attribution_version(metadata: Mapping[str, Any]) -> int:
    try:
        version = int(metadata.get("satellite_attribution_version", 0) or 0)
    except (TypeError, ValueError):
        version = 0
    if version > 0:
        return version
    # Compatibility with short-lived rows that overloaded attribution_version=2.
    if (
        _attribution_version(metadata) >= 2
        and isinstance(metadata.get("satellite_attribution"), Mapping)
    ):
        return 1
    return 0


def _stored_satellite_attribution(
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    nested = metadata.get("satellite_attribution")
    if isinstance(nested, Mapping):
        return _normalize_satellite_attribution(nested)
    # Forward compatibility for early v2 rows that stored the same whitelist
    # at the metadata root.
    return _normalize_satellite_attribution(metadata)


class Mem0Memory:
    """Domain facade over official Mem0, with a legacy test-store fallback."""

    def __init__(
        self,
        store: Any,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        load_project_env()
        self.store = store
        self.model = model or os.getenv("REFLECTION_LLM_MODEL_NAME") or os.getenv(
            "DEEPSEEK_MODEL_NAME"
        ) or os.getenv("GLM_MODEL_NAME", "deepseek-chat")
        self.base_url = base_url or os.getenv("REFLECTION_LLM_BASE_URL") or os.getenv(
            "DEEPSEEK_BASE_URL"
        ) or os.getenv("GLM_BASE_URL", "https://api.deepseek.com/v1")
        self.api_key = api_key or os.getenv("REFLECTION_LLM_API_KEY") or os.getenv(
            "DEEPSEEK_API_KEY"
        ) or os.getenv("GLM_API_KEY", "")

    # ---------------------------------------------------------------- llm io

    @retry(
        retry=retry_if_exception(_is_retryable),
        wait=wait_exponential(multiplier=5, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _call_llm(self, prompt: str, max_tokens: int = 2048, json_mode: bool = True) -> str:
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        resp = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=90,
        )
        if resp.status_code == 400 and json_mode:
            # Endpoint may not support structured output; retry plain once.
            return self._call_llm(prompt, max_tokens=max_tokens, json_mode=False)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    @staticmethod
    def _parse_json(raw: str) -> Optional[dict[str, Any]]:
        """Strict JSON parse with fence stripping; None (never raw text) on failure."""
        text = (raw or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        try:
            data = json.loads(text)
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            pass
        # Salvage attempt: outermost braces (handles leading/trailing prose).
        start, end = text.find("{"), text.rfind("}")
        if 0 <= start < end:
            try:
                data = json.loads(text[start : end + 1])
                return data if isinstance(data, dict) else None
            except json.JSONDecodeError:
                return None
        return None

    def _llm_json(self, prompt: str, max_tokens: int = 2048) -> Optional[dict[str, Any]]:
        """One retry on unparseable output, then give up (discard, never store raw)."""
        for _ in range(2):
            try:
                raw = self._call_llm(prompt, max_tokens=max_tokens)
            except Exception as exc:
                print(f"    memory LLM call failed: {exc}")
                return None
            data = self._parse_json(raw)
            if data is not None:
                return data
        return None

    # -------------------------------------------------------------- add path

    def add(
        self,
        question: str,
        answer: str,
        judge_result: str,
        log_data: Optional[dict[str, Any]] = None,
        task_id: str = "",
        ts_code: str = "",
        available_after: str = "",
    ) -> list[str]:
        """Extraction phase -> update phase. Returns human-readable op summaries."""
        predicted = extract_direction(answer)
        entry_month = parse_task_month(task_id)
        stance = functional_stance(predicted, judge_result)
        metadata = {
            "task_id": task_id,
            "ts_code": ts_code,
            "entry_month": entry_month,
            "available_after": available_after,
            "judge_result": judge_result,
            "predicted_direction": predicted,
            "functional_stance": stance,
            "source": "reflection",
            "source_tasks": [task_id],
        }

        if getattr(self.store, "is_official_mem0", False):
            trajectory = summarize_trajectory(log_data or {}) or "N/A"
            messages = [
                {
                    "role": "user",
                    "content": (
                        f"Task:\n{question[:1200]}\n\n"
                        f"Tool trajectory:\n{trajectory[:4000]}"
                    ),
                },
                {"role": "assistant", "content": (answer or "N/A")[:1200]},
                {
                    "role": "user",
                    "content": (
                        f"Evaluation: {judge_result}. Extract only a reusable "
                        "conditional lesson; do not store the final direction."
                    ),
                },
            ]
            results = self.store.add_inferred(messages, metadata=metadata)
            return [
                f"{str(item.get('event', 'MEM0')).upper()}: "
                f"{str(item.get('memory', ''))[:60]}"
                for item in results
            ]

        if not self.api_key:
            return []

        is_incorrect = judge_result == "INCORRECT"
        prompt = EXTRACTION_PROMPT.format(
            outcome_instructions=(
                EXTRACTION_INCORRECT_INSTRUCTIONS if is_incorrect else EXTRACTION_CORRECT_INSTRUCTIONS
            ),
            question=question[:800],
            answer=(answer or "N/A")[:400],
            judge_result=judge_result,
            trajectory=summarize_trajectory(log_data or {}) or "N/A",
        )
        data = self._llm_json(prompt)
        if data is None:
            print(f"    [{task_id}] extraction unparseable after retry — discarded")
            return []
        lessons = data.get("lessons", [])
        if not isinstance(lessons, list):
            return []
        # Symmetric cap: the old 2-correct/1-incorrect asymmetry minted lessons
        # from the majority-direction (bearish) wins at twice the rate and was
        # a mechanical amplifier of the bank's direction bias.
        max_lessons = 1

        ops: list[str] = []

        for lesson in lessons[:max_lessons]:
            if not isinstance(lesson, dict):
                continue
            content = str(lesson.get("content", "")).strip()
            if len(content) < 20:  # degenerate/empty extraction
                continue
            tags = [str(t) for t in lesson.get("tags", []) if str(t).strip()]
            lesson_metadata = {
                "task_id": task_id,
                "ts_code": ts_code,
                "entry_month": entry_month,
                "available_after": available_after,
                "judge_result": judge_result,
                "predicted_direction": predicted,
                "functional_stance": stance,
                "tags": tags,
                "source": "reflection",
                "source_tasks": [task_id],
            }
            op = self._consolidate(content, lesson_metadata, task_id)
            if op:
                ops.append(op)
        return ops

    def _consolidate(self, content: str, metadata: dict[str, Any], task_id: str) -> str:
        """Mem0 update phase for one candidate lesson."""
        try:
            candidate_emb = self.store.embed(content)
        except Exception as exc:
            print(f"    [{task_id}] embedding failed, lesson discarded: {exc}")
            return ""

        similar = self.store.search(query_embedding=candidate_emb, top_k=5)
        if not similar or similar[0][1] < _CONSOLIDATION_SIM_THRESHOLD:
            self.store.add(content, metadata=metadata, embedding=candidate_emb, source_task=task_id)
            return f"ADD: {content[:60]}"

        existing_block = "\n".join(
            f"- id={rec.id}\n  content={rec.content}" for rec, _ in similar
        )
        decision = self._llm_json(
            UPDATE_PROMPT.format(candidate=content, existing=existing_block),
            max_tokens=4096,  # reasoning models need headroom before content
        )
        if decision is None:
            # Consolidation undecidable -> conservative ADD (never lose a parsed lesson).
            self.store.add(content, metadata=metadata, embedding=candidate_emb, source_task=task_id)
            return f"ADD(fallback): {content[:60]}"

        action = str(decision.get("action", "ADD")).upper()
        target_id = str(decision.get("target_id", "") or "")
        new_content = str(decision.get("new_content", "") or "").strip()
        known_ids = {rec.id for rec, _ in similar}

        if action == "NONE":
            return f"NONE(dup): {content[:60]}"

        if action == "UPDATE" and target_id in known_ids and len(new_content) >= 20:
            target = next(rec for rec, _ in similar if rec.id == target_id)
            merged_meta = self._merge_metadata(target, metadata)
            self.store.update(target_id, new_content, metadata_patch=merged_meta, source_task=task_id)
            return f"UPDATE {target_id[:8]}: {new_content[:60]}"

        if action == "DELETE" and target_id in known_ids:
            self.store.delete(target_id, source_task=task_id, reason="contradicted by new evidence")
            self.store.add(content, metadata=metadata, embedding=candidate_emb, source_task=task_id)
            return f"DELETE {target_id[:8]} + ADD: {content[:60]}"

        # ADD, or malformed UPDATE/DELETE payload -> safe default.
        self.store.add(content, metadata=metadata, embedding=candidate_emb, source_task=task_id)
        return f"ADD: {content[:60]}"

    @staticmethod
    def _merge_metadata(target: MemoryRecord, candidate_meta: dict[str, Any]) -> dict[str, Any]:
        """Metadata for a merged lesson: visibility month is the LATER of the two
        sources (safe under the before_month filter); stances that disagree
        become 'mixed' (exempt from quota, like neutral)."""
        old_month = str(target.metadata.get("entry_month", "") or "")
        new_month = str(candidate_meta.get("entry_month", "") or "")
        months = [m for m in (old_month, new_month) if m]
        old_stance = target.metadata.get("functional_stance", "neutral")
        new_stance = candidate_meta.get("functional_stance", "neutral")
        source_tasks = list(
            dict.fromkeys(
                list(target.metadata.get("source_tasks", []))
                + list(candidate_meta.get("source_tasks", []))
            )
        )
        merged = {
            "entry_month": max(months) if months else "",
            "functional_stance": old_stance if old_stance == new_stance else "mixed",
            "source_tasks": source_tasks,
            "judge_result": "MERGED",
        }
        availability = [
            normalize_date(value)
            for value in (
                target.metadata.get("available_after"),
                candidate_meta.get("available_after"),
            )
            if normalize_date(value)
        ]
        if availability:
            merged["available_after"] = max(availability)
        return merged

    def add_monthly(
        self,
        month: str,
        features_table: str,
        n_stocks: int,
        available_after: str = "",
    ) -> list[str]:
        """Monthly cross-sectional reflection (v2 learning signal).

        One LLM call over the month's full decision-time feature table plus
        realized labels distills at most 2 patterns backed by n=16 evidence,
        replacing the noise-fitted n=1 per-task lessons. Production delegates
        extraction and consolidation to official Mem0.
        """
        monthly_metadata = {
            "task_id": f"monthly_{month}",
            "entry_month": month,
            "available_after": available_after,
            "judge_result": "MONTHLY",
            "predicted_direction": "",
            "functional_stance": "neutral",
            "source": "monthly_reflection",
            "source_tasks": [f"monthly_{month}"],
        }
        if getattr(self.store, "is_official_mem0", False):
            results = self.store.add_inferred(
                (
                    f"Month {month} contains {n_stocks} settled A-share "
                    "point-in-time predictions. Extract at most one reusable "
                    "cross-sectional conditional lesson from this table. Do "
                    "not store unconditional direction counts.\n\n"
                    f"{features_table}"
                ),
                metadata=monthly_metadata,
            )
            return [
                f"{str(item.get('event', 'MEM0')).upper()}: "
                f"{str(item.get('memory', ''))[:60]}"
                for item in results
            ]

        if not self.api_key:
            return []
        # Reasoning models (deepseek-v4-pro) burn completion tokens on hidden
        # reasoning before emitting content; cross-sectional analysis needs a
        # far larger budget than simple extraction or the content comes back
        # empty at 2048.
        data = self._llm_json(
            MONTHLY_REFLECTION_PROMPT.format(month=month, n=n_stocks, table=features_table),
            max_tokens=8192,
        )
        if data is None:
            print(f"    [monthly {month}] reflection unparseable after retry — discarded")
            return []
        lessons = data.get("lessons", [])
        if not isinstance(lessons, list):
            return []

        ops: list[str] = []
        for lesson in lessons[:2]:
            if not isinstance(lesson, dict):
                continue
            content = str(lesson.get("content", "")).strip()
            if len(content) < 20:
                continue
            tags = [str(t) for t in lesson.get("tags", []) if str(t).strip()]
            metadata = {
                "task_id": f"monthly_{month}",
                "entry_month": month,
                "available_after": available_after,
                "judge_result": "MONTHLY",
                "predicted_direction": "",
                # Cross-sectional conditionals are two-sided by construction;
                # exempt from the direction quota like neutral lessons.
                "functional_stance": "neutral",
                "tags": tags,
                "source": "monthly_reflection",
                "source_tasks": [f"monthly_{month}"],
            }
            op = self._consolidate(content, metadata, task_id=f"monthly_{month}")
            if op:
                ops.append(op)
        return ops

    # --------------------------------------------- unified trader episodes

    def add_trader_episode(
        self,
        *,
        task_id: str,
        month: str,
        available_after: str,
        weights: Mapping[str, float],
        cash: float,
        gross_return: float,
        net_return: float,
        index_return: float,
        active_return: float,
        total_cost: float,
        contributions: Mapping[str, float],
        episode_kind: str = "",
        anchor_attribution: Mapping[str, Any] | None = None,
        satellite_attribution: Mapping[str, Any] | None = None,
        reasoning: str = "",
        parse_ok: bool = True,
    ) -> str:
        """Idempotently store one factual portfolio episode in Mem0.

        The episode is written with ``infer=False`` through the store adapter:
        portfolio weights and realized P&L must not be rewritten by an LLM.
        Visibility is still gated by the exact liquidation date.
        """
        entry_month = normalize_month(month or parse_task_month(task_id))
        availability = normalize_date(available_after)
        if not task_id or not entry_month or not availability:
            raise ValueError("trader episode requires task_id, month, and exit date")

        positions = sorted(
            (
                (str(code), float(weight))
                for code, weight in weights.items()
                if float(weight) > 1e-9
            ),
            key=lambda item: (-item[1], item[0]),
        )
        ranked_contributions = sorted(
            (
                (str(code), float(value))
                for code, value in contributions.items()
            ),
            key=lambda item: (-item[1], item[0]),
        )
        allocation_text = (
            ", ".join(f"{code}={weight:.1%}" for code, weight in positions)
            if positions
            else "无股票仓位"
        )
        best = ", ".join(
            f"{code}={value:+.2%}" for code, value in ranked_contributions[:2]
        ) or "无"
        worst = ", ".join(
            f"{code}={value:+.2%}" for code, value in ranked_contributions[-2:]
        ) or "无"
        # Kept as a compatibility-only argument for older callers.  Free-form
        # model reasoning is deliberately never persisted or re-injected.
        _ = reasoning
        satellite = _normalize_satellite_attribution(satellite_attribution)
        content_lines = [
            f"交易月份 {entry_month}（任务 {task_id}，{availability} 后可见）。",
            f"仓位：{allocation_text}；CASH={float(cash):.1%}。",
            (
                f"实现收益：毛收益 {float(gross_return):+.2%}，"
                f"扣费后 {float(net_return):+.2%}，沪深300 {float(index_return):+.2%}，"
                f"主动收益 {float(active_return):+.2%}，"
                f"按100万元名义交易费 ¥{float(total_cost):.2f}。"
            ),
            f"股票净贡献较大：{best}；较弱：{worst}。",
            (
                "输出格式合法。"
                if parse_ok
                else "原始输出不合法，本月按全现金回退执行。"
            ),
        ]
        attribution = dict(anchor_attribution or {})
        if attribution:
            anchor_positions = sorted(
                (
                    (str(code), float(weight))
                    for code, weight in dict(
                        attribution.get("anchor_weights", {})
                    ).items()
                    if float(weight) > 1e-9
                ),
                key=lambda item: (-item[1], item[0]),
            )
            anchor_text = ", ".join(
                f"{code}={weight:.1%}" for code, weight in anchor_positions
            )
            deviation = float(attribution.get("deviation_active_return", 0.0))
            dropped = ",".join(attribution.get("dropped_from_anchor", [])) or "无"
            added = ",".join(attribution.get("added_vs_anchor", [])) or "无"
            delta_contributions = sorted(
                (
                    (str(code), float(value))
                    for code, value in dict(
                        attribution.get("weight_delta_contributions", {})
                    ).items()
                ),
                key=lambda item: (item[1], item[0]),
            )
            weakest_delta = ", ".join(
                f"{code}={value:+.2%}" for code, value in delta_contributions[:2]
            ) or "无"
            content_lines.extend(
                [
                    (
                        f"动量锚点：{anchor_text}；"
                        f"CASH={float(attribution.get('anchor_cash', 0.0)):.1%}。"
                    ),
                    (
                        "锚点扣费后收益 "
                        f"{float(attribution.get('anchor_net_return', 0.0)):+.2%}，"
                        f"实际相对锚点偏离损益 {deviation:+.2%}，"
                        f"成本差 ¥{float(attribution.get('cost_delta', 0.0)):+.2f}。"
                    ),
                    (
                        f"重合 {int(attribution.get('overlap_count', 0))}/4；"
                        f"降配/剔除={dropped}；新增={added}；"
                        f"现金差={float(attribution.get('cash_delta', 0.0)):+.1%}；"
                        f"市场状态={str(attribution.get('market_regime', '')) or 'unknown'}。"
                    ),
                    f"偏离拖累最大的权重差贡献：{weakest_delta}。",
                ]
            )
            if deviation < -0.001:
                content_lines.append(
                    f"结构化教训：偏离动量锚点造成 {abs(deviation):.2%} 损失；"
                    "下次除非满足硬锚双风险门槛，否则不重复该偏离。"
                )
            elif deviation > 0.001:
                content_lines.append(
                    f"结构化教训：有证据的偏离贡献 {deviation:.2%}；"
                    "只复用已满足的风险类别，不复用当时叙事。"
                )
            else:
                content_lines.append(
                    "结构化教训：本次偏离相对锚点影响接近零；保持硬锚优先。"
                )
        if satellite:
            actual_satellites = ",".join(satellite["selected_codes"])
            deterministic_satellites = ",".join(
                satellite["deterministic_codes"]
            )
            fallback = satellite.get("selection_fallback")
            fallback_text = (
                "是" if fallback is True else "否" if fallback is False else "未知"
            )
            source_text = {
                "agent": "agent",
                "deterministic_fallback": "deterministic_fallback",
            }.get(str(satellite.get("selection_source", "")), "unknown")
            content_lines.extend(
                [
                    (
                        "核心-卫星固定归因："
                        f"实际卫星={actual_satellites}；"
                        f"确定性top卫星={deterministic_satellites}；"
                        f"来源={source_text}；确定性回退={fallback_text}；"
                        f"市场状态={satellite['regime']}。"
                    ),
                    (
                        "固定袖套核心/卫星/CASH="
                        f"{float(satellite.get('core_weight', 0.0)):.1%}/"
                        f"{float(satellite.get('satellite_weight', 0.0)):.1%}/"
                        f"{float(satellite.get('cash_weight', 0.0)):.1%}；"
                        f"卫星净收益={float(satellite.get('satellite_net_return', 0.0)):+.2%}，"
                        "卫星/核心净贡献="
                        f"{float(satellite['satellite_net_contribution']):+.2%}/"
                        f"{float(satellite.get('core_net_contribution', 0.0)):+.2%}。"
                    ),
                ]
            )
            signal_context = []
            if satellite.get("candidate_signal_date"):
                signal_context.append(
                    f"候选信号日={satellite['candidate_signal_date']}"
                )
            if satellite.get("target"):
                signal_context.append(f"目标={satellite['target']}")
            if signal_context:
                content_lines.append("；".join(signal_context) + "。")
            for candidate in satellite["candidate_facts"]:
                rank_text = f"候选排名={candidate['candidate_rank']}"
                if "model_rank" in candidate:
                    rank_text += f"，模型排名={candidate['model_rank']}"
                score_text = (
                    f"，模型分数={candidate['model_score']:+.6f}"
                    if "model_score" in candidate
                    else ""
                )
                realized_return = candidate.get("realized_stock_return")
                realized_excess = candidate.get("realized_excess_vs_csi300")
                return_text = (
                    f"个股收益 {float(realized_return):+.2%}"
                    if realized_return is not None
                    else "个股收益 未记录"
                )
                excess_text = (
                    f"相对沪深300 {float(realized_excess):+.2%}"
                    if realized_excess is not None
                    else "相对沪深300 未记录"
                )
                content_lines.append(
                    f"卫星候选 {candidate['code']}：{rank_text}{score_text}，"
                    f"{return_text}，{excess_text}；"
                    "实际权重/加权收益/净贡献="
                    f"{candidate.get('actual_weight', 0.0):.1%}/"
                    f"{candidate.get('weighted_return_contribution', 0.0):+.2%}/"
                    f"{candidate.get('net_contribution', 0.0):+.2%}；"
                    "确定性权重/加权收益/净贡献="
                    f"{candidate.get('deterministic_weight', 0.0):.1%}/"
                    f"{candidate.get('deterministic_weighted_return_contribution', 0.0):+.2%}/"
                    f"{candidate.get('deterministic_net_contribution', 0.0):+.2%}。"
                )
            content_lines.extend(
                [
                    (
                        "确定性卫星反事实扣费后收益 "
                        f"{satellite['deterministic_counterfactual_net_return']:+.2%}；"
                        "实际减确定性卫星 "
                        f"{satellite['actual_minus_deterministic_net_return']:+.2%}"
                        f"（¥{float(satellite.get('actual_minus_deterministic_pnl', 0.0)):+.2f}）。"
                    ),
                    (
                        f"纯动量top4扣费后 {satellite['pure_momentum_net_return']:+.2%}；"
                        "实际减纯动量top4 "
                        f"{satellite['actual_minus_pure_momentum_net_return']:+.2%}"
                        f"（¥{float(satellite.get('actual_minus_pure_momentum_pnl', 0.0)):+.2f}）。"
                    ),
                ]
            )
        content = "\n".join(content_lines)
        metadata = {
            "task_id": task_id,
            "entry_month": entry_month,
            "available_after": availability,
            "judge_result": "PORTFOLIO_VALID" if parse_ok else "PORTFOLIO_INVALID",
            "functional_stance": "neutral",
            "source": "trader_episode",
            "source_tasks": [task_id],
            "episode_kind": str(episode_kind or "").strip().lower(),
            "gross_return": round(float(gross_return), 10),
            "net_return": round(float(net_return), 10),
            "index_return": round(float(index_return), 10),
            "active_return": round(float(active_return), 10),
            "cash_weight": round(float(cash), 10),
            "holding_count": len(positions),
            "position_weights": {
                code: round(float(weight), 10) for code, weight in positions
            },
            "net_contributions": {
                str(code): round(float(value), 10)
                for code, value in contributions.items()
            },
            "total_cost": round(float(total_cost), 6),
            "attribution_version": 0,
            "satellite_attribution_version": 0,
            "satellite_attribution": {},
            "satellite_selected_codes": [],
            "satellite_deterministic_codes": [],
            "satellite_selection_fallback": None,
            "satellite_selection_source": "",
            "satellite_market_regime": "",
            "satellite_net_contribution": 0.0,
            "satellite_actual_minus_deterministic_net_return": 0.0,
            "satellite_actual_minus_pure_momentum_net_return": 0.0,
        }
        if attribution:
            metadata.update(
                {
                    "attribution_version": 1,
                    "anchor_net_return": round(
                        float(attribution.get("anchor_net_return", 0.0)), 10
                    ),
                    "anchor_active_return": round(
                        float(attribution.get("anchor_active_return", 0.0)), 10
                    ),
                    "deviation_active_return": round(
                        float(attribution.get("deviation_active_return", 0.0)), 10
                    ),
                    "anchor_cash": round(
                        float(attribution.get("anchor_cash", 0.0)), 10
                    ),
                    "cash_delta": round(
                        float(attribution.get("cash_delta", 0.0)), 10
                    ),
                    "overlap_count": int(attribution.get("overlap_count", 0)),
                    "anchor_codes": sorted(
                        str(code)
                        for code in dict(
                            attribution.get("anchor_weights", {})
                        )
                    ),
                    "market_regime": str(
                        attribution.get("market_regime", "")
                    ),
                }
            )
        if satellite:
            metadata.update(
                {
                    "satellite_attribution_version": 1,
                    "satellite_attribution": satellite,
                    "satellite_selected_codes": satellite["selected_codes"],
                    "satellite_deterministic_codes": satellite[
                        "deterministic_codes"
                    ],
                    "satellite_selection_fallback": satellite.get(
                        "selection_fallback"
                    ),
                    "satellite_selection_source": satellite.get(
                        "selection_source", "unknown"
                    ),
                    "satellite_market_regime": satellite["regime"],
                    "satellite_net_contribution": round(
                        float(satellite["satellite_net_contribution"]), 10
                    ),
                    "satellite_actual_minus_deterministic_net_return": round(
                        float(
                            satellite[
                                "actual_minus_deterministic_net_return"
                            ]
                        ),
                        10,
                    ),
                    "satellite_actual_minus_pure_momentum_net_return": round(
                        float(
                            satellite[
                                "actual_minus_pure_momentum_net_return"
                            ]
                        ),
                        10,
                    ),
                }
            )

        matching = [
            record
            for record in self.store.all_records()
            if record.metadata.get("source") == "trader_episode"
            and str(record.metadata.get("task_id", "")) == task_id
        ]
        primary = matching[0] if matching else None
        for duplicate in matching[1:]:
            self.store.delete(
                duplicate.id,
                source_task=task_id,
                reason="duplicate trader episode task id",
            )

        if primary is None:
            record = self.store.add(
                content,
                metadata=metadata,
                source_task=task_id,
            )
            return f"ADD trader episode {record.id[:8]}: {entry_month}"

        same_metadata = self._comparable_rule_metadata(
            primary.metadata
        ) == self._comparable_rule_metadata(metadata)
        if primary.content == content and same_metadata:
            return f"UNCHANGED trader episode {primary.id[:8]}: {entry_month}"
        self.store.update(
            primary.id,
            content,
            metadata_patch=metadata,
            source_task=task_id,
        )
        return f"UPDATE trader episode {primary.id[:8]}: {entry_month}"

    def trader_episode_block(
        self,
        before_date: str,
        *,
        max_episodes: int = 3,
    ) -> str:
        """Format only factual trader episodes matured by ``before_date``."""
        cutoff = normalize_date(before_date)
        if not cutoff or max_episodes <= 0:
            return ""
        visible = []
        for record in self.store.all_records():
            if record.metadata.get("source") != "trader_episode":
                continue
            available_after = normalize_date(
                record.metadata.get("available_after")
            )
            # Fail closed: a portfolio outcome without an exact liquidation
            # date is never injected into a point-in-time task.
            if available_after and available_after <= cutoff:
                visible.append(record)
        if not visible:
            return ""
        visible.sort(
            key=lambda record: (
                normalize_month(record.metadata.get("entry_month")),
                str(record.metadata.get("task_id", "")),
            )
        )

        portfolio_nav = 1.0
        benchmark_nav = 1.0
        attributed_portfolio_nav = 1.0
        anchor_nav = 1.0
        attributed_count = 0
        selection_actual_nav = 1.0
        selection_counterfactual_nav = 1.0
        satellite_count = 0
        selection_diff_count = 0
        agent_selection_diff_count = 0
        selection_increment_sum = 0.0
        satellite_contribution_sum = 0.0
        core_contribution_sum = 0.0
        core_contribution_count = 0
        fallback_count = 0
        fallback_known_count = 0
        active_wins = 0
        open_market_count = 0
        for record in visible:
            net = _fact_number(record.metadata.get("net_return"))
            index = _fact_number(record.metadata.get("index_return"))
            if net is None or index is None:
                continue
            active = _fact_number(record.metadata.get("active_return"))
            if active is None:
                active = net - index
            portfolio_nav *= 1.0 + net
            benchmark_nav *= 1.0 + index
            active_wins += active > 0
            if record.metadata.get("episode_kind") == "open_market":
                open_market_count += 1
            version = _attribution_version(record.metadata)
            anchor_return = _fact_number(
                record.metadata.get("anchor_net_return")
            )
            if version >= 1 and anchor_return is not None:
                attributed_portfolio_nav *= 1.0 + net
                anchor_nav *= 1.0 + anchor_return
                attributed_count += 1
            if _satellite_attribution_version(record.metadata) < 1:
                continue
            satellite = _stored_satellite_attribution(record.metadata)
            if not satellite:
                continue
            actual_return = _fact_number(satellite.get("actual_net_return"))
            counterfactual_return = _fact_number(
                satellite.get("deterministic_counterfactual_net_return")
            )
            if actual_return is None or counterfactual_return is None:
                continue
            satellite_count += 1
            satellite_contribution = _fact_number(
                satellite.get("satellite_net_contribution")
            )
            if satellite_contribution is not None:
                satellite_contribution_sum += satellite_contribution
            core_contribution = _fact_number(
                satellite.get("core_net_contribution")
            )
            if core_contribution is not None:
                core_contribution_sum += core_contribution
                core_contribution_count += 1
            if satellite.get("selection_differs") is True:
                selection_actual_nav *= 1.0 + actual_return
                selection_counterfactual_nav *= 1.0 + counterfactual_return
                selection_diff_count += 1
                incremental = _fact_number(
                    satellite.get("actual_minus_deterministic_net_return")
                )
                if incremental is not None:
                    selection_increment_sum += incremental
                if satellite.get("selection_source") == "agent":
                    agent_selection_diff_count += 1
            fallback = satellite.get("selection_fallback")
            if isinstance(fallback, bool):
                fallback_known_count += 1
                fallback_count += fallback

        structured = [
            record
            for record in visible
            if _attribution_version(record.metadata) >= 1
            or _satellite_attribution_version(record.metadata) >= 1
            or record.metadata.get("episode_kind") == "open_market"
        ]
        recent = structured[-max_episodes:]
        audit_title = (
            "### 已到期的开放市场组合审计（严格 walk-forward）\n"
            if open_market_count and open_market_count == len(visible)
            else "### 已到期的交易组合审计（严格 walk-forward）\n"
            if open_market_count
            else "### 已到期的动量锚点偏离审计（严格 walk-forward）\n"
        )
        lines = [
            (
                audit_title
                + f"截至 {cutoff} 共有 {len(visible)} 个可见交易窗口；"
                f"组合累计 {portfolio_nav - 1:+.2%}，沪深300同期累计 "
                f"{benchmark_nav - 1:+.2%}，跑赢窗口 {active_wins}/{len(visible)}。"
            ),
        ]
        if attributed_count:
            lines.append(
                (
                    f"有结构化锚点归因 {attributed_count} 期；实际组合相对同期"
                    "动量锚点累计差 "
                    f"{attributed_portfolio_nav / anchor_nav - 1:+.2%}。"
                )
                if anchor_nav > 0
                else f"有结构化锚点归因 {attributed_count} 期。"
            )
        elif not open_market_count:
            lines.append("旧记录没有锚点归因，已禁止注入其中的历史推理。")
        if open_market_count:
            lines.append(
                f"有开放市场事实 episode {open_market_count} 期；"
                "仅包含确定性仓位、到期收益与个股净贡献，不包含模型推理。"
            )
        if satellite_count:
            unknown_fallbacks = satellite_count - fallback_known_count
            fallback_summary = (
                f"明确使用确定性回退 {fallback_count}/{fallback_known_count} 期"
                if fallback_known_count
                else "回退标记均缺失"
            )
            if unknown_fallbacks:
                fallback_summary += f"，未知 {unknown_fallbacks} 期"
            core_summary = (
                f"，核心净贡献逐期合计 {core_contribution_sum:+.2%}"
                if core_contribution_count
                else ""
            )
            lines.append(
                f"有成熟卫星归因 {satellite_count} 期；卫星净贡献逐期合计 "
                f"{satellite_contribution_sum:+.2%}{core_summary}；"
                f"{fallback_summary}。"
            )
        if selection_diff_count and selection_counterfactual_nav > 0:
            lines.append(
                f"其中实际与确定性top卫星不同 {selection_diff_count} 期"
                f"（source=agent {agent_selection_diff_count} 期）；"
                "模型/agent选择相对确定性反事实复合增量 "
                f"{selection_actual_nav / selection_counterfactual_nav - 1:+.2%}，"
                f"逐期增量合计 {selection_increment_sum:+.2%}。"
            )
        elif satellite_count:
            lines.append(
                "成熟记录的实际卫星均与确定性top一致；"
                "模型/agent相对确定性卫星增量效应 +0.00%。"
            )
        if open_market_count:
            lines.append(
                "这些是到期后的固定公式审计，不是当前股票的收益标签。"
                "只校准筛选流程、仓位和集中风险；历史代码不得机械继承。"
            )
        else:
            lines.append(
                "这些是到期后的固定公式审计，不是当前股票的收益标签。"
                "只校准偏离与卫星选择成本；不得继承过去的长篇理由。"
            )
        for record in recent:
            net = _fact_number(record.metadata.get("net_return"))
            index = _fact_number(record.metadata.get("index_return"))
            if net is None or index is None:
                continue
            active = _fact_number(record.metadata.get("active_return"))
            if active is None:
                active = net - index
            entry_month = normalize_month(record.metadata.get("entry_month")) or "?"
            facts = [
                (
                    f"{entry_month}：实际扣费后 {net:+.2%}，"
                    f"沪深300 {index:+.2%}，主动收益 {active:+.2%}"
                )
            ]
            if record.metadata.get("episode_kind") == "open_market":
                cash = _fact_number(record.metadata.get("cash_weight"))
                holding_count = record.metadata.get("holding_count")
                if cash is not None or holding_count is not None:
                    cash_text = f"{cash:.1%}" if cash is not None else "?"
                    facts.append(
                        f"持仓数 {holding_count if holding_count is not None else '?'}，"
                        f"CASH {cash_text}"
                    )
                positions = dict(record.metadata.get("position_weights", {}) or {})
                if positions:
                    position_text = ",".join(
                        f"{code}={float(weight):.1%}"
                        for code, weight in sorted(
                            positions.items(),
                            key=lambda item: (-float(item[1]), str(item[0])),
                        )
                    )
                    facts.append(f"历史仓位={position_text}")
                contributions = dict(
                    record.metadata.get("net_contributions", {}) or {}
                )
                if contributions:
                    ranked = sorted(
                        (
                            (str(code), float(value))
                            for code, value in contributions.items()
                        ),
                        key=lambda item: (-item[1], item[0]),
                    )
                    best = ",".join(
                        f"{code}={value:+.2%}" for code, value in ranked[:2]
                    )
                    worst = ",".join(
                        f"{code}={value:+.2%}" for code, value in ranked[-2:]
                    )
                    facts.append(f"净贡献较强={best}，较弱={worst}")
            if _attribution_version(record.metadata) >= 1:
                anchor_return = _fact_number(
                    record.metadata.get("anchor_net_return")
                )
                deviation = _fact_number(
                    record.metadata.get("deviation_active_return")
                )
                if anchor_return is not None and deviation is not None:
                    facts.append(
                        f"纯动量扣费后 {anchor_return:+.2%}，"
                        f"实际减纯动量 {deviation:+.2%}"
                    )
                overlap = record.metadata.get("overlap_count")
                regime = str(record.metadata.get("market_regime", "") or "")
                if overlap is not None or regime:
                    try:
                        overlap_text = str(int(overlap))
                    except (TypeError, ValueError):
                        overlap_text = "?"
                    facts.append(
                        f"动量重合 {overlap_text}/4，市场状态={regime or 'unknown'}"
                    )

            satellite = (
                _stored_satellite_attribution(record.metadata)
                if _satellite_attribution_version(record.metadata) >= 1
                else {}
            )
            if satellite:
                actual_codes = ",".join(satellite["selected_codes"])
                deterministic_codes = ",".join(
                    satellite["deterministic_codes"]
                )
                fallback = satellite.get("selection_fallback")
                fallback_text = (
                    "是"
                    if fallback is True
                    else "否"
                    if fallback is False
                    else "未知"
                )
                facts.append(
                    f"实际卫星={actual_codes}，确定性top={deterministic_codes}，"
                    f"source={satellite.get('selection_source', 'unknown')}，"
                    f"回退={fallback_text}，regime={satellite['regime']}；"
                    "核心/卫星/CASH="
                    f"{float(satellite.get('core_weight', 0.0)):.1%}/"
                    f"{float(satellite.get('satellite_weight', 0.0)):.1%}/"
                    f"{float(satellite.get('cash_weight', 0.0)):.1%}"
                )
                selection_context = []
                if satellite.get("candidate_signal_date"):
                    selection_context.append(
                        f"信号日={satellite['candidate_signal_date']}"
                    )
                if satellite.get("target"):
                    selection_context.append(f"target={satellite['target']}")
                if selection_context:
                    facts.append("，".join(selection_context))
                candidate_facts: list[str] = []
                for candidate in satellite["candidate_facts"]:
                    rank = f"候选rank={candidate['candidate_rank']}"
                    if "model_rank" in candidate:
                        rank += f",model_rank={candidate['model_rank']}"
                    score = (
                        f",score={candidate['model_score']:+.6f}"
                        if "model_score" in candidate
                        else ""
                    )
                    stock_return = candidate.get("realized_stock_return")
                    excess_return = candidate.get(
                        "realized_excess_vs_csi300"
                    )
                    stock_text = (
                        f"{float(stock_return):+.2%}"
                        if stock_return is not None
                        else "?"
                    )
                    excess_text = (
                        f"{float(excess_return):+.2%}"
                        if excess_return is not None
                        else "?"
                    )
                    roles = []
                    if candidate.get("selected"):
                        roles.append("actual")
                    if candidate.get("deterministic_selected"):
                        roles.append("det")
                    candidate_facts.append(
                        f"{candidate['code']}[{'+'.join(roles)}]({rank}{score},"
                        f"收益/超额={stock_text}/{excess_text},"
                        "实际w/加权收益/净贡献="
                        f"{candidate.get('actual_weight', 0.0):.1%}/"
                        f"{candidate.get('weighted_return_contribution', 0.0):+.2%}/"
                        f"{candidate.get('net_contribution', 0.0):+.2%},"
                        "det w/净贡献="
                        f"{candidate.get('deterministic_weight', 0.0):.1%}/"
                        f"{candidate.get('deterministic_net_contribution', 0.0):+.2%})"
                    )
                if candidate_facts:
                    facts.append("卫星事实：" + "，".join(candidate_facts))
                facts.append(
                    "卫星/核心净贡献 "
                    f"{satellite['satellite_net_contribution']:+.2%}/"
                    f"{float(satellite.get('core_net_contribution', 0.0)):+.2%}，"
                    "确定性卫星净贡献 "
                    f"{float(satellite.get('deterministic_satellite_net_contribution', 0.0)):+.2%}，"
                    "确定性卫星反事实扣费后 "
                    f"{satellite['deterministic_counterfactual_net_return']:+.2%}，"
                    "实际减确定性 "
                    f"{satellite['actual_minus_deterministic_net_return']:+.2%}"
                )
                facts.append(
                    f"纯动量top4扣费后 {satellite['pure_momentum_net_return']:+.2%}，"
                    "实际减纯动量top4 "
                    f"{satellite['actual_minus_pure_momentum_net_return']:+.2%}"
                )
            lines.append("- " + "；".join(facts) + "。")
        return "\n".join(lines)

    # -------------------------------------------- rolling statistical reflection

    @property
    def samples_path(self):
        return self.store.store_dir / f"{self.store.namespace}_samples.jsonl"

    def log_samples(self, samples: Sequence[Mapping[str, Any]]) -> tuple[int, int]:
        """Idempotently upsert structured feature/outcome rows by task id.

        Upsert, rather than append-only deduplication, lets a resumed run repair
        a partial row while guaranteeing that reprocessing a completed month
        cannot give that month extra statistical weight.
        """
        incoming: dict[str, dict[str, Any]] = {}
        for raw in samples:
            task_id = str(raw.get("task_id", "") or "")
            if not task_id:
                continue
            row: dict[str, Any] = {}
            for key, value in raw.items():
                # pandas/numpy scalars expose item(); convert them before JSON.
                row[str(key)] = value.item() if hasattr(value, "item") else value
            incoming[task_id] = row
        if not incoming:
            return 0, 0

        with self.store._locked():
            existing: dict[str, dict[str, Any]] = {}
            if self.samples_path.exists():
                with open(self.samples_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        row = json.loads(line)
                        task_id = str(row.get("task_id", "") or "")
                        if task_id:
                            existing[task_id] = row

            added = sum(task_id not in existing for task_id in incoming)
            updated = sum(
                task_id in existing and existing[task_id] != row
                for task_id, row in incoming.items()
            )
            if not added and not updated:
                return 0, 0

            existing.update(incoming)
            ordered = sorted(
                existing.values(),
                key=lambda row: (str(row.get("entry_date", "")), str(row.get("task_id", ""))),
            )
            tmp = self.samples_path.with_suffix(".jsonl.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                for row in ordered:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            os.replace(tmp, self.samples_path)
        return added, updated

    def load_samples(self) -> list[dict[str, Any]]:
        if not self.samples_path.exists():
            return []
        with self.store._locked():
            with open(self.samples_path, "r", encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()]

    @staticmethod
    def _comparable_rule_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
        # generated_for_date is audit context, not evidence.  If no label has
        # newly matured, advancing to another decision date must be a no-op.
        comparable = {
            k: v
            for k, v in metadata.items()
            if k not in {"generated_for_date", "source_task"}
        }
        if "available_after" in comparable:
            comparable["available_after"] = normalize_date(
                comparable["available_after"]
            )
        return comparable

    def _sync_rolling_rules(
        self, rules: Sequence[ValidatedRule], as_of_date: str
    ) -> list[str]:
        """Make rolling-statistical records exactly match the validated set."""
        existing_records = [
            record
            for record in self.store.all_records()
            if record.metadata.get("source") == "rolling_statistical"
        ]
        existing: dict[str, MemoryRecord] = {}
        duplicates: list[MemoryRecord] = []
        for record in existing_records:
            rule_id = str(record.metadata.get("rule_id", "") or "")
            if rule_id and rule_id not in existing:
                existing[rule_id] = record
            else:
                duplicates.append(record)

        desired = {rule.rule_id: rule for rule in rules}
        source_task = f"rolling_{normalize_date(as_of_date)}"
        ops: list[str] = []

        for record in duplicates:
            if self.store.delete(
                record.id, source_task=source_task, reason="duplicate rolling rule id"
            ):
                ops.append(f"DELETE duplicate {record.id[:8]}")

        for rule_id, record in existing.items():
            if rule_id not in desired and self.store.delete(
                record.id,
                source_task=source_task,
                reason="rolling rule failed current temporal validation",
            ):
                ops.append(f"DELETE {rule_id}")

        for rule_id, rule in desired.items():
            record = existing.get(rule_id)
            if record is None:
                self.store.add(
                    rule.content,
                    metadata=rule.metadata,
                    source_task=source_task,
                )
                ops.append(f"ADD {rule_id}: {rule.content[:60]}")
                continue

            same_content = record.content == rule.content
            same_metadata = self._comparable_rule_metadata(
                record.metadata
            ) == self._comparable_rule_metadata(rule.metadata)
            if same_content and same_metadata:
                continue
            self.store.update(
                record.id,
                rule.content,
                metadata_patch=rule.metadata,
                source_task=source_task,
            )
            ops.append(f"UPDATE {rule_id}: {rule.content[:60]}")

        if rules and not ops:
            return [f"UNCHANGED: {len(rules)} validated rolling rules"]
        return ops

    def refresh_rolling(
        self,
        as_of_date: str,
        config: Mapping[str, Any] | None = None,
    ) -> list[str]:
        """Mine all matured samples and synchronize the validated rule set."""
        rule_config = RollingRuleConfig.from_mapping(config)
        rules = mine_rolling_rules(self.load_samples(), as_of_date, rule_config)
        return self._sync_rolling_rules(rules, as_of_date)

    def rolling_snapshot(self, as_of_date: str) -> dict[str, int]:
        """Return point-in-time sample/month/rule counts for prompt guardrails."""
        cutoff = normalize_date(as_of_date)
        if not cutoff:
            return {"samples": 0, "months": 0, "rules": 0}
        eligible: dict[str, dict[str, Any]] = {}
        for row in self.load_samples():
            task_id = str(row.get("task_id", "") or "")
            exit_date = normalize_date(row.get("exit_date"))
            if task_id and exit_date and exit_date <= cutoff:
                eligible[task_id] = row
        months = {
            normalize_month(row.get("entry_month") or row.get("entry_date"))
            for row in eligible.values()
        }
        months.discard("")
        rules = sum(
            record.metadata.get("source") == "rolling_statistical"
            and (
                not normalize_date(record.metadata.get("available_after"))
                or normalize_date(record.metadata.get("available_after")) <= cutoff
            )
            for record in self.store.all_records()
        )
        return {"samples": len(eligible), "months": len(months), "rules": rules}

    def save_note(self, content: str, tags: list[str], as_of_month: str = "") -> MemoryRecord:
        """Agent-initiated note (MCP memory_save). Notes without a month are
        never visible under a before_month filter (fail-safe against leakage)."""
        return self.store.add(
            content,
            metadata={
                "source": "agent_note",
                "tags": tags,
                "entry_month": as_of_month,
                "available_after": month_available_after(as_of_month),
                "functional_stance": "neutral",
            },
        )

    # ------------------------------------------------------- outcomes ledger

    @property
    def outcomes_path(self):
        return self.store.store_dir / f"{self.store.namespace}_outcomes.jsonl"

    def log_outcome(
        self,
        task_id: str,
        month: str,
        predicted: str,
        judge_result: str,
        available_after: str = "",
    ) -> None:
        """Upsert one judged prediction into the calibration ledger.

        ``available_after`` is the label's exit date.  Exact-date filtering in
        ``calibration_block`` prevents a 20-day outcome from being used merely
        because its entry month is earlier than the current task month.
        """
        if judge_result not in ("CORRECT", "INCORRECT") or predicted not in ("跑赢", "跑输"):
            return
        record = {
            "task_id": task_id,
            "month": normalize_month(month),
            "predicted": predicted,
            "judge_result": judge_result,
            "available_after": normalize_date(available_after),
        }
        with self.store._locked():
            rows: list[dict[str, Any]] = []
            if self.outcomes_path.exists():
                with open(self.outcomes_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            rows.append(json.loads(line))
            existing = next((row for row in rows if row.get("task_id") == task_id), None)
            if existing == record:
                return
            if existing is None:
                rows.append(record)
            else:
                rows = [record if row.get("task_id") == task_id else row for row in rows]
            tmp = self.outcomes_path.with_suffix(".jsonl.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            os.replace(tmp, self.outcomes_path)

    def calibration_block(
        self,
        before_month: str,
        min_samples: int = 16,
        before_date: str = "",
    ) -> str:
        """Self-calibration stats over past months' judged predictions.

        The agent's own bias is the one large-sample, measurable signal in past
        outcomes (unlike n=1 episodic lessons). Direction-neutral by design: it
        reports the agent's error profile, never any stock's future direction.
        """
        if not (before_month or before_date) or not self.outcomes_path.exists():
            return ""
        month_cutoff = normalize_month(before_month or before_date)
        date_cutoff = normalize_date(before_date)
        rows: list[dict[str, str]] = []
        with open(self.outcomes_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                available_after = normalize_date(r.get("available_after"))
                if date_cutoff and available_after:
                    visible = available_after <= date_cutoff
                elif available_after and month_cutoff:
                    visible = normalize_month(available_after) < month_cutoff
                else:
                    visible = bool(r.get("month")) and normalize_month(r["month"]) < month_cutoff
                if visible:
                    rows.append(r)
        if len(rows) < min_samples:
            return ""

        n = len(rows)
        pred_counts = Counter(r["predicted"] for r in rows)
        hits = Counter()
        for r in rows:
            if r["judge_result"] == "CORRECT":
                hits[r["predicted"]] += 1
        # Realized label is recoverable from (prediction, judge): binary task.
        label_counts: Counter = Counter()
        for r in rows:
            if r["judge_result"] == "CORRECT":
                label_counts[r["predicted"]] += 1
            else:
                label_counts["跑赢" if r["predicted"] == "跑输" else "跑输"] += 1

        def pct(x: int, d: int) -> str:
            return f"{x / d * 100:.0f}%" if d else "-"

        lines = [
            f"### 自我校准统计（基于你此前 {n} 次已到期预测，截止 {before_date or before_month}）",
            f"- 你的预测分布：跑输 {pred_counts['跑输']} 次（{pct(pred_counts['跑输'], n)}）、"
            f"跑赢 {pred_counts['跑赢']} 次（{pct(pred_counts['跑赢'], n)}）",
            f"- 分方向命中率：预测「跑输」命中 {pct(hits['跑输'], pred_counts['跑输'])}"
            f"（{hits['跑输']}/{pred_counts['跑输']}）、"
            f"预测「跑赢」命中 {pct(hits['跑赢'], pred_counts['跑赢'])}"
            f"（{hits['跑赢']}/{pred_counts['跑赢']}）",
            f"- 同期实际标签分布：跑输 {pct(label_counts['跑输'], n)}、跑赢 {pct(label_counts['跑赢'], n)}",
            "- 校准提示：若你的预测分布明显偏离实际标签分布、且占多数方向的命中率并不更高，"
            "说明存在系统性方向偏置。请仅基于当前任务自身证据独立判断，不要默认沿用多数方向。",
        ]
        return "\n".join(lines)

    def reliability_block(
        self,
        before_month: str,
        min_samples: int = 16,
        before_date: str = "",
    ) -> str:
        """Direction-free reliability summary over matured predictions.

        Unlike ``calibration_block``, this intentionally omits prediction and
        label direction counts.  Those counts can become a prompt anchor even
        when the accompanying prose asks the model to correct the bias.
        """
        if not (before_month or before_date) or not self.outcomes_path.exists():
            return ""
        month_cutoff = normalize_month(before_month or before_date)
        date_cutoff = normalize_date(before_date)
        rows: list[dict[str, str]] = []
        with open(self.outcomes_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                available_after = normalize_date(row.get("available_after"))
                if date_cutoff and available_after:
                    visible = available_after <= date_cutoff
                elif available_after and month_cutoff:
                    visible = normalize_month(available_after) < month_cutoff
                else:
                    visible = bool(row.get("month")) and (
                        normalize_month(row["month"]) < month_cutoff
                    )
                if visible:
                    rows.append(row)
        if len(rows) < min_samples:
            return ""

        correct = sum(row.get("judge_result") == "CORRECT" for row in rows)
        accuracy = correct / len(rows)
        return "\n".join(
            [
                (
                    "### 历史可靠性提示"
                    f"（基于此前 {len(rows)} 次已到期预测，截止 {before_date or before_month}）"
                ),
                f"- 历史总体命中率：{correct}/{len(rows)}（{accuracy:.0%}）。",
                "- 该统计只描述方法可靠性，不提供任何方向先验；"
                "当历史命中率接近随机时应降低置信度，并以当前任务的点时工具证据为准。",
                "- 不得根据历史预测数量或标签数量默认选择任一方向。",
            ]
        )

    def has_vector_memories(
        self,
        before_month: str = "",
        before_date: str = "",
    ) -> bool:
        """Whether a temporally visible validated/vector memory exists."""
        month_cutoff = normalize_month(before_month or before_date)
        date_cutoff = normalize_date(before_date)
        if getattr(self.store, "is_official_mem0", False):
            filters = self._native_visibility_filters(month_cutoff, date_cutoff)
            if filters is None:
                return bool(self.store.all_records())
            return bool(self.store.all_records(filters=filters))

        for record in self.store.all_records():
            available_after = normalize_date(record.metadata.get("available_after"))
            if date_cutoff and available_after:
                visible = available_after <= date_cutoff
            elif available_after and month_cutoff:
                visible = normalize_month(available_after) < month_cutoff
            else:
                entry_month = normalize_month(record.metadata.get("entry_month"))
                visible = bool(entry_month) and bool(month_cutoff) and entry_month < month_cutoff
            if visible:
                return True
        return False

    # ------------------------------------------------------------ search path

    @staticmethod
    def _native_visibility_filters(
        month_cutoff: str,
        date_cutoff: str,
    ) -> dict[str, Any] | None:
        """Qdrant filter for the fresh-store invariant: every memory has an
        ISO ``available_after`` payload. Missing availability stays hidden."""
        if date_cutoff:
            iso = (
                f"{date_cutoff[:4]}-{date_cutoff[4:6]}-"
                f"{date_cutoff[6:8]}T00:00:00Z"
            )
            return {"available_after": {"lte": iso}}
        if month_cutoff:
            return {
                "available_after": {
                    "lt": f"{month_cutoff[:4]}-{month_cutoff[5:7]}-01T00:00:00Z"
                }
            }
        return None

    def search(
        self,
        query: str,
        top_k: int = 3,
        before_month: str = "",
        before_date: str = "",
        stance_balance: bool = True,
    ) -> list[tuple[MemoryRecord, float]]:
        month_cutoff = normalize_month(before_month or before_date)
        date_cutoff = normalize_date(before_date)

        def predicate(rec: MemoryRecord) -> bool:
            if not (month_cutoff or date_cutoff):
                return True
            available_after = normalize_date(rec.metadata.get("available_after"))
            if date_cutoff and available_after:
                return available_after <= date_cutoff
            if available_after and month_cutoff:
                return normalize_month(available_after) < month_cutoff
            month = normalize_month(rec.metadata.get("entry_month"))
            # Records with unknown availability are hidden under filtering.
            return bool(month) and bool(month_cutoff) and month < month_cutoff

        if top_k <= 0:
            return []

        # Validated rolling rules are few (max three) and their relevance is
        # condition-based, not semantic.  Pin them ahead of vector matches so
        # a generic stock-name embedding cannot hide the statistical rules.
        native_filters = (
            self._native_visibility_filters(month_cutoff, date_cutoff)
            if getattr(self.store, "is_official_mem0", False)
            else None
        )
        all_records = (
            self.store.all_records(filters=native_filters)
            if native_filters is not None
            else self.store.all_records()
        )
        if not all_records:
            return []
        rolling = [
            rec
            for rec in all_records
            if rec.metadata.get("source") == "rolling_statistical"
            and (
                getattr(self.store, "is_official_mem0", False)
                or predicate(rec)
            )
        ]
        rolling.sort(
            key=lambda rec: (
                float(rec.metadata.get("q_value", 1.0)),
                -float(rec.metadata.get("validation_lift", 0.0)),
                -int(rec.metadata.get("validation_support", 0)),
            )
        )
        picked: list[tuple[MemoryRecord, float]] = [
            (rec, max(0.0, 1.0 - float(rec.metadata.get("q_value", 1.0))))
            for rec in rolling[:top_k]
        ]
        rolling_ids = {rec.id for rec in rolling}
        if len(picked) >= top_k or all(rec.id in rolling_ids for rec in all_records):
            return picked[:top_k]

        try:
            search_kwargs: dict[str, Any] = {
                "query": query,
                "top_k": top_k * 4,
                "min_score": 0.05,
            }
            if getattr(self.store, "is_official_mem0", False):
                search_kwargs["filters"] = native_filters
            else:
                search_kwargs["predicate"] = predicate
            wide = self.store.search(**search_kwargs)
        except Exception as exc:
            print(f"    memory search failed (no injection): {exc}")
            return picked

        wide = [
            (rec, score)
            for rec, score in wide
            if rec.id not in rolling_ids
            # Trader episodes have a dedicated fixed-schema renderer.  Never
            # expose legacy episode content, which may contain model reasoning.
            and rec.metadata.get("source") != "trader_episode"
        ]
        if not stance_balance:
            return (picked + wide)[:top_k]

        # Tight quota: at most ~1/3 of slots per directional stance (1 bullish
        # + 1 bearish at top_k=3). ceil(top_k/2) still let 2-vs-1 tilts through,
        # which compounded over months in the mem0 v1 run.
        max_per_stance = max(1, top_k // 3)
        counts: Counter = Counter()
        deferred: list[tuple[MemoryRecord, float]] = []
        for rec, score in wide:
            if len(picked) >= top_k:
                break
            stance = rec.metadata.get("functional_stance", "neutral")
            if stance in ("bullish", "bearish") and counts[stance] >= max_per_stance:
                deferred.append((rec, score))
                continue
            picked.append((rec, score))
            counts[stance] += 1
        for rec, score in deferred:
            if len(picked) >= top_k:
                break
            picked.append((rec, score))
        return picked

    @staticmethod
    def format_results(results: list[tuple[MemoryRecord, float]]) -> str:
        if not results:
            return "No relevant memories found."
        lines = []
        for i, (rec, score) in enumerate(results, 1):
            tags = rec.metadata.get("tags", [])
            tag_str = ", ".join(tags) if tags else "none"
            month = rec.metadata.get("entry_month", "") or "?"
            if rec.metadata.get("source") == "rolling_statistical":
                lines.append(
                    f"{i}. [大样本验证|q={rec.metadata.get('q_value', '?')}|"
                    f"验证n={rec.metadata.get('validation_support', '?')}|"
                    f"截至={rec.metadata.get('available_after', '?')}] {rec.content}"
                )
            else:
                lines.append(
                    f"{i}. [score={score:.3f}|来源月={month}|tags={tag_str}] {rec.content}"
                )
        return "\n".join(lines)
