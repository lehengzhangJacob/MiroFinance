"""Evolve tool descriptions using DSPy + GEPA."""

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
from evolution.core.fitness import tool_selection_metric
from evolution.core.constraints import ConstraintValidator
from evolution.tools.tool_module import (
    ToolDescModule,
    find_tool_registry,
    load_tool_registry,
    format_tool_catalog,
    parse_tool_catalog,
)
from evolution.monitor.trace_logger import set_trace_file

console = Console()


def evolve(
    tool_set: str,
    iterations: int = 5,
    dataset_path: Optional[str] = None,
    optimizer_model: str = "openai/gpt-4.1",
    eval_model: str = "openai/gpt-4.1-mini",
    hermes_repo: Optional[str] = None,
    dry_run: bool = False,
):
    config = EvolutionConfig(hermes_agent_path=resolve_hermes_agent_path(hermes_repo))
    console.print(f"\n[bold cyan]Phase 2[/bold cyan] — Evolving tool descriptions: [bold]{tool_set}[/bold]\n")

    registry_path = find_tool_registry(tool_set, config.hermes_agent_path)
    if not registry_path:
        console.print(f"[red]Tool registry not found for {tool_set}[/red]")
        sys.exit(1)

    registry = load_tool_registry(registry_path)
    catalog_text = format_tool_catalog(registry["tools"])
    console.print(f"  Loaded: {registry_path}")
    console.print(f"  Tools: {len(registry['tools'])}")

    ds_path = Path(dataset_path or f"datasets/tools/{tool_set}_tool_selection")
    if not ds_path.exists():
        console.print(f"[red]Dataset not found: {ds_path}[/red]")
        sys.exit(1)
    dataset = EvalDataset.load(ds_path)
    console.print(f"  Dataset: {len(dataset.train)} train / {len(dataset.val)} val / {len(dataset.holdout)} holdout")

    if dry_run:
        console.print("[green]DRY RUN OK[/green]")
        return

    lm = make_dspy_lm(eval_model)
    dspy.configure(lm=lm)
    baseline = ToolDescModule(catalog_text)
    trainset = dataset.to_dspy_examples("train")
    valset = dataset.to_dspy_examples("val")

    trace_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    set_trace_file(Path("output") / f"tools_{tool_set}" / trace_ts / "gepa_trace.jsonl")

    start = time.time()
    reflection_lm = make_dspy_lm(optimizer_model)
    try:
        optimizer = dspy.GEPA(metric=tool_selection_metric, auto="light", reflection_lm=reflection_lm)
        optimized = optimizer.compile(baseline, trainset=trainset, valset=valset)
    except Exception as exc:
        console.print(f"[yellow]GEPA fallback MIPROv2: {exc}[/yellow]")
        optimizer = dspy.MIPROv2(metric=tool_selection_metric, auto="light")
        optimized = optimizer.compile(baseline, trainset=trainset)

    elapsed = time.time() - start
    evolved_catalog = optimized.catalog_text
    evolved_tools = parse_tool_catalog(evolved_catalog)
    if not evolved_tools:
        evolved_tools = registry["tools"]

    holdout = dataset.to_dspy_examples("holdout")
    b_scores, e_scores = [], []
    with dspy.context(lm=lm):
        for ex in holdout:
            b_scores.append(tool_selection_metric(ex, baseline(task_input=ex.task_input)))
            e_scores.append(tool_selection_metric(ex, optimized(task_input=ex.task_input)))
    avg_b = sum(b_scores) / max(1, len(b_scores))
    avg_e = sum(e_scores) / max(1, len(e_scores))

    out = Path("output") / f"tools_{tool_set}" / trace_ts
    out.mkdir(parents=True, exist_ok=True)
    (out / "baseline_tools.json").write_text(json.dumps(registry, indent=2))
    (out / "evolved_tools.json").write_text(json.dumps({"tools": evolved_tools}, indent=2))
    metrics = {
        "phase": "tool_descriptions",
        "tool_set": tool_set,
        "baseline_score": avg_b,
        "evolved_score": avg_e,
        "improvement": avg_e - avg_b,
        "elapsed_seconds": elapsed,
        "iterations": iterations,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    table = Table(title="Tool Description Evolution")
    table.add_column("Metric")
    table.add_column("Baseline", justify="right")
    table.add_column("Evolved", justify="right")
    table.add_row("Holdout accuracy", f"{avg_b:.3f}", f"{avg_e:.3f}")
    console.print(table)
    console.print(f"  Output: {out}/")


@click.command()
@click.option("--tool-set", default="dev", help="Tool set name (dev -> dev_tools.json)")
@click.option("--iterations", default=5)
@click.option("--dataset-path", default=None)
@click.option("--optimizer-model", default="openai/gpt-4.1")
@click.option("--eval-model", default="openai/gpt-4.1-mini")
@click.option("--hermes-repo", default=None)
@click.option("--dry-run", is_flag=True)
def main(tool_set, iterations, dataset_path, optimizer_model, eval_model, hermes_repo, dry_run):
    evolve(tool_set, iterations, dataset_path, optimizer_model, eval_model, hermes_repo, dry_run)


if __name__ == "__main__":
    main()
