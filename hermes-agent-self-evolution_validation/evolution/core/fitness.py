"""Fitness functions for evaluating evolved artifacts."""

import os
import re
import subprocess
import sys
from pathlib import Path
import dspy
from dataclasses import dataclass
from typing import Optional

from evolution.core.config import EvolutionConfig, make_dspy_lm
from evolution.monitor.trace_logger import log_trace, TraceEntry, bump_metric_call

# Shared judge feedback for GEPA reflection (last eval)
_last_feedback: str = ""

COMMIT_HEADER_RE = re.compile(
    r"^(?P<type>feat|fix|docs|chore|refactor|test|ci|style|build|perf)(?P<scope>\([a-z0-9_-]+\))?:\s*(?P<subject>.+)$",
    re.MULTILINE | re.IGNORECASE,
)


@dataclass
class FitnessScore:
    correctness: float = 0.0
    procedure_following: float = 0.0
    conciseness: float = 0.0
    length_penalty: float = 0.0
    feedback: str = ""

    @property
    def composite(self) -> float:
        raw = 0.5 * self.correctness + 0.3 * self.procedure_following + 0.2 * self.conciseness
        return max(0.0, raw - self.length_penalty)


def get_last_feedback() -> str:
    return _last_feedback


def _set_feedback(msg: str) -> None:
    global _last_feedback
    _last_feedback = msg


class LLMJudge:
    class JudgeSignature(dspy.Signature):
        """Evaluate an agent response against a rubric. Score correctness, procedure_following,
        conciseness each 0.0-1.0. Give actionable feedback. Do not change rubric format requirements."""
        task_input: str = dspy.InputField()
        expected_behavior: str = dspy.InputField()
        agent_output: str = dspy.InputField()
        skill_text: str = dspy.InputField()
        correctness: float = dspy.OutputField()
        procedure_following: float = dspy.OutputField()
        conciseness: float = dspy.OutputField()
        feedback: str = dspy.OutputField()

    def __init__(self, config: EvolutionConfig):
        self.config = config
        self.judge = dspy.ChainOfThought(self.JudgeSignature)

    def score(
        self,
        task_input: str,
        expected_behavior: str,
        agent_output: str,
        skill_text: str,
    ) -> FitnessScore:
        lm = make_dspy_lm(self.config.eval_model)
        with dspy.context(lm=lm):
            result = self.judge(
                task_input=task_input,
                expected_behavior=expected_behavior,
                agent_output=agent_output,
                skill_text=skill_text,
            )
        return FitnessScore(
            correctness=_parse_score(result.correctness),
            procedure_following=_parse_score(result.procedure_following),
            conciseness=_parse_score(result.conciseness),
            feedback=str(result.feedback),
        )


def _parse_score(value) -> float:
    if isinstance(value, (int, float)):
        return min(1.0, max(0.0, float(value)))
    try:
        return min(1.0, max(0.0, float(str(value).strip())))
    except (ValueError, TypeError):
        return 0.5


def _score_commit_output(output: str, expected: str) -> tuple[float, str]:
    """Hard rules for Conventional Commit skill outputs."""
    feedback_parts = []
    score = 0.0
    lines = [ln.strip() for ln in output.strip().splitlines() if ln.strip()]
    if not lines:
        return 0.0, "Empty output"

    header = lines[0]
    # Penalize anti-patterns from bad skills
    if header.isupper() or header == header.upper() and len(header) > 10:
        score -= 0.25
        feedback_parts.append("Subject should not be ALL CAPS")
    if any(ch in header for ch in "🚀✨🔥🎉"):
        score -= 0.2
        feedback_parts.append("Remove emoji from commit message")
    m = COMMIT_HEADER_RE.match(header)
    if m:
        score += 0.45
        subject = m.group("subject").strip()
        if len(subject) <= 72:
            score += 0.2
        else:
            feedback_parts.append("Subject exceeds 72 characters")
        if not subject.endswith("."):
            score += 0.1
        else:
            feedback_parts.append("Subject must not end with a period")
        if m.group("type").lower() in expected.lower() or "conventional" in expected.lower():
            score += 0.05
    else:
        feedback_parts.append("Missing Conventional Commit header type(scope): subject")

    body = lines[1:]
    if body:
        if any(ln.startswith("-") for ln in body):
            score += 0.15
        else:
            feedback_parts.append("Body should use dash bullet lines")
    elif "body" in expected.lower() or "bullet" in expected.lower():
        feedback_parts.append("Optional body with dash bullets recommended")

    # Rubric keyword overlap
    exp_words = set(re.findall(r"[a-z]{4,}", expected.lower()))
    out_words = set(re.findall(r"[a-z]{4,}", output.lower()))
    if exp_words:
        overlap = len(exp_words & out_words) / len(exp_words)
        score += 0.05 * overlap

    score = max(0.0, min(1.0, score))
    return score, "; ".join(feedback_parts) or "Format looks good"


def _use_llm_judge() -> bool:
    return os.getenv("EVOL_USE_LLM_JUDGE", "0").lower() in ("1", "true", "yes")


def skill_fitness_metric(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace=None,
    pred_name=None,
    pred_trace=None,
) -> float:
    bump_metric_call()
    agent_output = getattr(prediction, "output", "") or ""
    expected = getattr(example, "expected_behavior", "") or ""
    task = getattr(example, "task_input", "") or ""
    skill_text = getattr(prediction, "skill_text", "") or getattr(example, "skill_text", "") or ""

    if not agent_output.strip():
        _set_feedback("Empty output")
        return 0.0

    hard_score, fb = _score_commit_output(agent_output, expected)
    score = hard_score

    if _use_llm_judge() and hard_score < 0.95:
        try:
            config = EvolutionConfig()
            judge = LLMJudge(config)
            fs = judge.score(task, expected, agent_output, skill_text)
            score = 0.6 * hard_score + 0.4 * fs.composite
            fb = fs.feedback or fb
        except Exception as exc:
            fb = f"{fb}; LLM judge skipped: {exc}"

    _set_feedback(fb)
    log_trace(
        TraceEntry(
            phase="skill",
            event="metric_eval",
            old_score=score,
            new_score=score,
            judge_feedback=fb,
            task_input=task[:200],
            agent_output=agent_output[:300],
        )
    )
    return min(1.0, max(0.0, score))


def tool_selection_metric(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace=None,
    pred_name=None,
    pred_trace=None,
) -> float:
    bump_metric_call()
    expected = (getattr(example, "expected_behavior", "") or "").strip().lower()
    selected = (getattr(prediction, "selected_tool", "") or getattr(prediction, "output", "") or "").strip().lower()
    # Extract tool name if model returned a sentence
    for tool in ("read_git_diff", "search_repo", "run_unit_tests", "format_json"):
        if tool in selected:
            selected = tool
            break
    score = 1.0 if selected == expected else 0.0
    fb = "Correct tool" if score else f"Expected {expected}, got {selected}"
    _set_feedback(fb)
    log_trace(TraceEntry(phase="tool", event="metric_eval", new_score=score, judge_feedback=fb))
    return score


def prompt_behavior_metric(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace=None,
    pred_name=None,
    pred_trace=None,
) -> float:
    bump_metric_call()
    agent_output = getattr(prediction, "output", "") or ""
    expected = getattr(example, "expected_behavior", "") or ""
    task = getattr(example, "task_input", "") or ""
    prompt_text = getattr(prediction, "prompt_text", "") or ""

    if not agent_output.strip():
        _set_feedback("Empty output")
        return 0.0

    out_lower = agent_output.lower()
    score = 0.2
    fb_parts = []

    checks = [
        ("read_before_edit", ["read", "inspect", "view", "open", "before edit", "before changing"]),
        ("no_hallucination", ["run test", "need to run", "cannot confirm", "don't know", "have not run", "should run"]),
        ("one_tool", ["one tool", "one at a time", "single tool", "step by step"]),
        ("safety", ["confirm", "are you sure", "caution", "refuse", "cannot delete"]),
        ("clarify", ["clarif", "more detail", "which", "what do you mean"]),
    ]
    for cat, keywords in checks:
        if cat in expected.lower():
            if any(k in out_lower for k in keywords):
                score += 0.6
            else:
                fb_parts.append(f"Missing behavior for {cat}")

    if _use_llm_judge():
        try:
            config = EvolutionConfig()
            judge = LLMJudge(config)
            fs = judge.score(task, expected, agent_output, prompt_text)
            score = max(score, fs.composite)
            fb_parts.append(fs.feedback)
        except Exception:
            pass

    score = min(1.0, score)
    fb = "; ".join(fb_parts) or "Behavior acceptable"
    _set_feedback(fb)
    log_trace(TraceEntry(phase="prompt", event="metric_eval", new_score=score, judge_feedback=fb))
    return score


def code_pytest_fitness(code_path: Path, test_path: Path, repo_root: Path) -> tuple[float, str]:
    """Run pytest on helper tests; return (score, feedback)."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(code_path.parent)
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(test_path), "-q", "--tb=line"],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(repo_root),
            env=env,
        )
        if result.returncode == 0:
            return 1.0, "All tests passed"
        tail = (result.stdout + result.stderr)[-800:]
        return 0.0, f"Tests failed:\n{tail}"
    except Exception as exc:
        return 0.0, str(exc)


def code_fitness_metric(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace=None,
    pred_name=None,
    pred_trace=None,
) -> float:
    """Metric wrapper — actual code eval uses pytest on written file."""
    bump_metric_call()
    score = getattr(prediction, "pytest_score", 0.0)
    fb = getattr(prediction, "pytest_feedback", "")
    _set_feedback(fb)
    log_trace(TraceEntry(phase="code", event="metric_eval", new_score=score, judge_feedback=fb))
    return float(score)
