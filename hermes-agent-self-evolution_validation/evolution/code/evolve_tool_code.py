"""Evolve tool implementation code using DSPy + GEPA with pytest gate."""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
import dspy
from rich.console import Console
from rich.table import Table

from evolution.core.config import EvolutionConfig, resolve_hermes_agent_path, make_dspy_lm
from evolution.core.fitness import code_pytest_fitness, code_fitness_metric, get_last_feedback
from evolution.code.code_module import find_code_target, load_code_file
from evolution.monitor.trace_logger import set_trace_file, log_mutation

console = Console()


class CodeGenModule(dspy.Module):
    """Generate/fix Python source; fitness from pytest."""

    class FixSource(dspy.Signature):
        """Produce a complete Python module that implements slugify and truncate_subject.
        slugify: lowercase, spaces to hyphens, remove illegal chars, collapse repeated hyphens.
        truncate_subject: strip and truncate to max_len (default 72) without off-by-one errors.
        Output ONLY valid Python source code, no markdown fences."""
        task_input: str = dspy.InputField(desc="Hint about failing tests or requirements")
        current_source: str = dspy.InputField(desc="Current module source")
        output: str = dspy.OutputField(desc="Full Python source code")

    def __init__(self, code_text: str, test_path: Path, repo_root: Path):
        super().__init__()
        self.code_text = code_text
        self.test_path = test_path
        self.repo_root = repo_root
        self.predictor = dspy.ChainOfThought(self.FixSource)

    def _eval_code(self, source: str) -> tuple[float, str]:
        code_path = self.repo_root / "hermes-agent" / "code" / "text_helpers.py"
        code_path.parent.mkdir(parents=True, exist_ok=True)
        # Strip markdown fences if model added them
        src = source.strip()
        if src.startswith("```"):
            lines = src.splitlines()
            src = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        code_path.write_text(src, encoding="utf-8")
        return code_pytest_fitness(code_path, self.test_path, self.repo_root)

    def forward(self, task_input: str) -> dspy.Prediction:
        result = self.predictor(task_input=task_input, current_source=self.code_text)
        score, feedback = self._eval_code(result.output)
        return dspy.Prediction(
            output=result.output,
            pytest_score=score,
            pytest_feedback=feedback,
        )


def _reflective_fix(code_text: str, feedback: str, reflection_lm) -> str:
    """One-shot reflective code fix when GEPA leaves code unchanged."""
    class Fix(dspy.Signature):
        """Fix the Python module so all pytest tests pass. Return full source only."""
        current_source: str = dspy.InputField()
        test_feedback: str = dspy.InputField()
        fixed_source: str = dspy.OutputField()

    fixer = dspy.ChainOfThought(Fix)
    with dspy.context(lm=reflection_lm):
        result = fixer(current_source=code_text, test_feedback=feedback)
    src = str(result.fixed_source).strip()
    if src.startswith("```"):
        lines = src.splitlines()
        src = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return src


def evolve(
    target: str,
    iterations: int = 5,
    optimizer_model: str = "openai/gpt-4.1",
    eval_model: str = "openai/gpt-4.1-mini",
    hermes_repo: Optional[str] = None,
    dry_run: bool = False,
):
    repo_root = Path(__file__).resolve().parents[2]
    config = EvolutionConfig(hermes_agent_path=resolve_hermes_agent_path(hermes_repo))
    console.print(f"\n[bold cyan]Phase 4[/bold cyan] — Evolving code: [bold]{target}[/bold]\n")

    code_path = find_code_target(target, config.hermes_agent_path)
    if not code_path:
        console.print(f"[red]Code target not found: {target}[/red]")
        sys.exit(1)

    code = load_code_file(code_path)
    test_path = repo_root / "tests" / "code" / "test_text_helpers.py"
    console.print(f"  Loaded: {code_path} ({len(code['raw'])} chars)")
    console.print(f"  Tests: {test_path}")

    baseline_score, baseline_fb = code_pytest_fitness(code_path, test_path, repo_root)
    console.print(f"  Baseline pytest score: {baseline_score:.2f}")
    console.print(f"  {baseline_fb[:200]}")

    if dry_run:
        console.print("[green]DRY RUN OK[/green]")
        return

    lm = make_dspy_lm(eval_model)
    dspy.configure(lm=lm)
    reflection_lm = make_dspy_lm(optimizer_model)

    trace_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    set_trace_file(Path("output") / f"code_{target}" / trace_ts / "gepa_trace.jsonl")

    # Minimal train/val examples describing test requirements
    import dspy as _dspy
    examples = [
        _dspy.Example(
            task_input="Fix slugify to collapse repeated hyphens and truncate_subject max 72 chars",
        ).with_inputs("task_input"),
        _dspy.Example(
            task_input="truncate_subject must not allow 73 characters; slugify strips illegal chars",
        ).with_inputs("task_input"),
    ]

    baseline_mod = CodeGenModule(code["raw"], test_path, repo_root)
    start = time.time()

    evolved_source = code["raw"]
    evolved_score = baseline_score

    try:
        optimizer = dspy.GEPA(metric=code_fitness_metric, auto="light", reflection_lm=reflection_lm)
        optimized = optimizer.compile(baseline_mod, trainset=examples, valset=examples[:1])
        pred = optimized(task_input=examples[0].task_input)
        evolved_source = pred.output
        evolved_score = float(pred.pytest_score)
    except Exception as exc:
        console.print(f"[yellow]GEPA code path skipped: {exc}[/yellow]")

    # Reflective fallback if tests still fail
    if evolved_score < 1.0:
        for i in range(iterations):
            _, fb = code_pytest_fitness(
                config.hermes_agent_path / "code" / "text_helpers.py",
                test_path,
                repo_root,
            )
            fixed = _reflective_fix(evolved_source, fb, reflection_lm)
            score, fb2 = CodeGenModule(fixed, test_path, repo_root)._eval_code(fixed)
            log_mutation("code", i + 1, fixed[:500], evolved_score, score, score > evolved_score, fb2)
            if score >= evolved_score:
                evolved_source = fixed
                evolved_score = score
            if score >= 1.0:
                break

    elapsed = time.time() - start
    out = Path("output") / f"code_{target}" / trace_ts
    out.mkdir(parents=True, exist_ok=True)
    (out / "baseline_code.py").write_text(code["raw"])
    (out / "evolved_code.py").write_text(evolved_source)
    (config.hermes_agent_path / "code" / "text_helpers.py").write_text(evolved_source, encoding="utf-8")
    final_score, _ = code_pytest_fitness(
        config.hermes_agent_path / "code" / "text_helpers.py", test_path, repo_root
    )

    metrics = {
        "phase": "code",
        "target": target,
        "baseline_score": baseline_score,
        "evolved_score": final_score,
        "improvement": final_score - baseline_score,
        "elapsed_seconds": elapsed,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    table = Table(title="Code Evolution")
    table.add_column("Metric")
    table.add_column("Baseline", justify="right")
    table.add_column("Evolved", justify="right")
    table.add_row("Pytest score", f"{baseline_score:.3f}", f"{final_score:.3f}")
    console.print(table)
    console.print(f"  Output: {out}/")


@click.command()
@click.option("--target", default="text_helpers")
@click.option("--iterations", default=5)
@click.option("--optimizer-model", default="openai/gpt-4.1")
@click.option("--eval-model", default="openai/gpt-4.1-mini")
@click.option("--hermes-repo", default=None)
@click.option("--dry-run", is_flag=True)
def main(target, iterations, optimizer_model, eval_model, hermes_repo, dry_run):
    evolve(target, iterations, optimizer_model, eval_model, hermes_repo, dry_run)


if __name__ == "__main__":
    main()
