"""GEPA evolution trace logging."""

import json
import threading
from contextvars import ContextVar
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

_current_trace_file: ContextVar[Optional[Path]] = ContextVar("trace_file", default=None)
_metric_call_count: ContextVar[int] = ContextVar("metric_call_count", default=0)
_lock = threading.Lock()


@dataclass
class TraceEntry:
    iteration: int = 0
    phase: str = ""
    event: str = "metric_eval"
    proposed_text: str = ""
    old_score: float = 0.0
    new_score: float = 0.0
    accepted: Optional[bool] = None
    skip_reason: str = ""
    judge_feedback: str = ""
    task_input: str = ""
    agent_output: str = ""


def set_trace_file(path: Optional[Path]) -> None:
    _current_trace_file.set(path)
    _metric_call_count.set(0)


def get_trace_file() -> Optional[Path]:
    return _current_trace_file.get()


def log_trace(entry: TraceEntry) -> None:
    path = _current_trace_file.get()
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")


def log_mutation(
    phase: str,
    iteration: int,
    proposed_text: str,
    old_score: float,
    new_score: float,
    accepted: bool,
    skip_reason: str = "",
    judge_feedback: str = "",
) -> None:
    log_trace(
        TraceEntry(
            iteration=iteration,
            phase=phase,
            event="mutation",
            proposed_text=proposed_text[:2000],
            old_score=old_score,
            new_score=new_score,
            accepted=accepted,
            skip_reason=skip_reason,
            judge_feedback=judge_feedback,
        )
    )


def bump_metric_call() -> int:
    n = _metric_call_count.get() + 1
    _metric_call_count.set(n)
    return n
