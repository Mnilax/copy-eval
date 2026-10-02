"""CLI entry point for copy evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
import yaml
from rich.console import Console
from rich.table import Table
from rich.text import Text

from copyeval.models import EvalReport, Variant
from copyeval.pairwise import compute_elo, generate_pairs
from copyeval.rubric import DEFAULT_RUBRIC, load_rubric

app = typer.Typer(help="Marketing Copy Eval Harness — LLM-as-judge")
console = Console()


def _load_variants(path: Path) -> list[Variant]:
    try:
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() in (".yaml", ".yml"):
            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
        if not isinstance(data, dict) or not isinstance(data.get("variants"), list) or not data["variants"]:
            raise ValueError("Expected a nonempty variants list")
        variants = [Variant(**v) for v in data["variants"]]
        if len({v.id for v in variants}) != len(variants):
            raise ValueError("Variant IDs must be unique")
        return variants
    except (OSError, ValueError, TypeError, yaml.YAMLError) as e:
        raise typer.BadParameter(str(e), param_hint="variants_file") from e


@app.command()
def score(
    variants_file: Annotated[Path, typer.Argument(help="YAML/JSON file with variants")],
    rounds: Annotated[int, typer.Option("--rounds", "-n", min=1, help="Number of eval rounds (averaged)")] = 1,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Show prompts without API calls")] = False,
):
    """Score each variant on the rubric criteria."""
    from copyeval.judge import average_verdicts, build_judge_prompt, parse_verdict

    variants = _load_variants(variants_file)
    rubric = load_rubric()

    console.print(f"[bold]Evaluating {len(variants)} variants on {len(rubric)} criteria[/bold]")

    report = EvalReport()

    for v in variants:
        system, user = build_judge_prompt(v, rubric)
        if dry_run:
            console.print(Text(f"\n--- {v.id} ---\n{user[:200]}..."))
            continue

        from copyeval.client import call_judge
        verdicts = [parse_verdict(v.id, call_judge(system, user), rubric) for _ in range(rounds)]
        report.verdicts.append(average_verdicts(verdicts, rubric))

    if not dry_run and report.verdicts:
        _print_score_table(report)


@app.command()
def elo(
    variants_file: Annotated[Path, typer.Argument(help="YAML/JSON file with variants")],
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Show prompts without API calls")] = False,
):
    """Run pairwise comparisons and compute Elo ratings."""
    from copyeval.pairwise import build_pairwise_prompt, parse_pairwise

    variants = _load_variants(variants_file)
    pairs = generate_pairs(variants)

    console.print(f"[bold]Running {len(pairs)} pairwise comparisons[/bold]")

    results = []
    for a, b in pairs:
        system, user = build_pairwise_prompt(a, b)
        if dry_run:
            console.print(f"  {a.id} vs {b.id}", markup=False)
            continue

        from copyeval.client import call_judge
        data = call_judge(system, user)
        result = parse_pairwise(a.id, b.id, data)
        results.append(result)
        console.print(f"  {a.id} vs {b.id} → winner: {result.winner_id}", markup=False)

    if not dry_run and results:
        ratings = compute_elo(results, [v.id for v in variants])
        _print_elo_table(ratings)


def _print_score_table(report: EvalReport):
    table = Table(title="Rubric Scores")
    table.add_column("Variant", style="cyan")
    for c in DEFAULT_RUBRIC:
        table.add_column(c.name, justify="center")
    table.add_column("Total", justify="center", style="bold")
    table.add_column("Summary")

    for v in sorted(report.verdicts, key=lambda x: x.total_score, reverse=True):
        row = [v.variant_id]
        for c in DEFAULT_RUBRIC:
            s = next((s for s in v.scores if s.criterion == c.name), None)
            row.append(f"{s.score:.1f}" if s else "—")
        row.append(f"{v.total_score:.1f}")
        row.append(v.summary[:60])
        table.add_row(*row)

    console.print(table)


def _print_elo_table(ratings):
    table = Table(title="Elo Ratings")
    table.add_column("Rank", justify="center")
    table.add_column("Variant", style="cyan")
    table.add_column("Elo", justify="center", style="bold")
    table.add_column("W/L", justify="center")

    for i, r in enumerate(ratings, 1):
        table.add_row(str(i), r.variant_id, f"{r.rating:.0f}", f"{r.wins}/{r.losses}")

    console.print(table)


def main():
    app()

if __name__ == "__main__":
    main()
