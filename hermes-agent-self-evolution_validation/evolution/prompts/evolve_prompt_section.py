"""Evolve a system prompt section using DSPy + GEPA."""

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
from evolution.core.dataset_builder import EvalDataset
from evolution.core.fitness import prompt_behavior_metric
from evolution.prompts.prompt_module import PromptSectionModule, find_prompt_section, load_prompt_section
from evolution.monitor.trace_logger import set_trace_file

console = Console()


def evolve(
    section: str,
    iterations: int = 5,
    dataset_path: Optional[str] = None,
    optimizer_model: str = "openai/gpt-4.1",
    eval_model: str = "openai/gpt-4.1-mini",
    hermes_repo: Optional[str] = None,
    dry_run: bool = False,
):
    config = EvolutionConfig(hermes_agent_path=resolve_hermes_agent_path(hermes_repo))
    console.print(f"\n[bold cyan]Phase 3[/bold cyan] — Evolving prompt section: [bold]{section}[/bold]\n")

    path = find_prompt_section(section, config.hermes_agent_path)
    if not path:
        console.print(f"[red]Prompt section not found: {section}[/red]")
        sys.exit(1)

    prompt = load_prompt_section(path)
    ds_path = Path(dataset_path or f"datasets/prompts/{section}")
    dataset = EvalDataset.load(ds_path)
    console.print(f"  Loaded: {path} ({len(prompt['body'])} chars)")
    console.print(f"  Dataset: {len(dataset.train)} train / {len(dataset.val)} val / {len(dataset.holdout)} holdout")

    if dry_run:
        console.print("[green]DRY RUN OK[/green]")
        return

    lm = make_dspy_lm(eval_model)
    dspy.configure(lm=lm)
    baseline = PromptSectionModule(prompt["body"])
    trainset = dataset.to_dspy_examples("train")
    valset = dataset.to_dspy_examples("val")

    trace_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    set_trace_file(Path("output") / f"prompt_{section}" / trace_ts / "gepa_trace.jsonl")

    start = time.time()
    reflection_lm = make_dspy_lm(optimizer_model)
    try:
        optimizer = dspy.GEPA(metric=prompt_behavior_metric, auto="light", reflection_lm=reflection_lm)
        optimized = optimizer.compile(baseline, trainset=trainset, valset=valset)
    except Exception as exc:
        console.print(f"[yellow]GEPA fallback MIPROv2: {exc}[/yellow]")
        optimizer = dspy.MIPROv2(metric=prompt_behavior_metric, auto="light")
        optimized = optimizer.compile(baseline, trainset=trainset)

    elapsed = time.time() - start
    evolved_text = optimized.prompt_text

    holdout = dataset.to_dspy_examples("holdout")
    b_scores, e_scores = [], []
    with dspy.context(lm=lm):
        for ex in holdout:
            b_scores.append(prompt_behavior_metric(ex, baseline(task_input=ex.task_input)))
            e_scores.append(prompt_behavior_metric(ex, optimized(task_input=ex.task_input)))
    avg_b = sum(b_scores) / max(1, len(b_scores))
    avg_e = sum(e_scores) / max(1, len(e_scores))

    out = Path("output") / f"prompt_{section}" / trace_ts
    out.mkdir(parents=True, exist_ok=True)
    (out / "baseline_prompt.md").write_text(prompt["raw"])
    (out / "evolved_prompt.md").write_text(evolved_text)
    metrics = {
        "phase": "prompt_section",
        "section": section,
        "baseline_score": avg_b,
        "evolved_score": avg_e,
        "improvement": avg_e - avg_b,
        "elapsed_seconds": elapsed,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    table = Table(title="Prompt Section Evolution")
    table.add_column("Metric")
    table.add_column("Baseline", justify="right")
    table.add_column("Evolved", justify="right")
    table.add_row("Holdout score", f"{avg_b:.3f}", f"{avg_e:.3f}")
    console.print(table)
    console.print(f"  Output: {out}/")


@click.command()
@click.option("--section", required=True)
@click.option("--iterations", default=5)
@click.option("--dataset-path", default=None)
@click.option("--optimizer-model", default="openai/gpt-4.1")
@click.option("--eval-model", default="openai/gpt-4.1-mini")
@click.option("--hermes-repo", default=None)
@click.option("--dry-run", is_flag=True)
def main(section, iterations, dataset_path, optimizer_model, eval_model, hermes_repo, dry_run):
    evolve(section, iterations, dataset_path, optimizer_model, eval_model, hermes_repo, dry_run)


if __name__ == "__main__":
    main()
