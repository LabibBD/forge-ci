"""Command-line interface for FORGE-CI."""

from pathlib import Path
from typing import Annotated

import typer
import yaml
from pydantic import ValidationError

from forge_ci import __version__
from forge_ci.comparison import ComparisonError, compare_runs
from forge_ci.config import ExperimentConfig, load_config
from forge_ci.runner import run_evaluation

app = typer.Typer(
    name="forgeci",
    help="Continuous evaluation for robot-learning policies.",
    no_args_is_help=True,
)


def _load_config_or_exit(path: Path) -> ExperimentConfig:
    """Load configuration or terminate with a readable error."""

    try:
        return load_config(path)
    except FileNotFoundError:
        typer.echo(
            f"Configuration file not found: {path}",
            err=True,
        )
        raise typer.Exit(code=1) from None
    except (ValidationError, yaml.YAMLError, ValueError) as exc:
        typer.echo(
            f"Invalid configuration:\n{exc}",
            err=True,
        )
        raise typer.Exit(code=1) from None


@app.command()
def version() -> None:
    """Display the installed FORGE-CI version."""

    typer.echo(f"FORGE-CI {__version__}")


@app.command()
def validate(config: Path) -> None:
    """Validate an experiment YAML configuration."""

    experiment = _load_config_or_exit(config)

    typer.echo("Configuration valid")
    typer.echo(f"Name: {experiment.name}")
    typer.echo(f"Seed: {experiment.seed}")
    typer.echo(f"Episodes: {experiment.episodes}")


@app.command()
def evaluate(
    config: Path,
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            "-o",
            help="Directory in which run artifacts are written.",
        ),
    ] = Path("runs"),
) -> None:
    """Run an experiment and enforce its absolute quality gate."""

    experiment = _load_config_or_exit(config)

    evaluation = run_evaluation(
        experiment,
        output_root=output_dir,
    )

    summary = evaluation.summary

    typer.echo(f"Run: {evaluation.run_dir}")

    typer.echo(
        f"Success rate: {summary.success_rate:.1%} "
        f"({summary.successes}/{summary.episodes})"
    )

    typer.echo(f"Required: {summary.min_success_rate:.1%}")

    typer.echo(
        f"Gate: {'PASS' if summary.gate_passed else 'FAIL'}"
    )

    if not summary.gate_passed:
        raise typer.Exit(code=2)


@app.command()
def compare(
    baseline: Path,
    candidate: Path,
) -> None:
    """Compare a candidate run against a known-good baseline."""

    try:
        comparison = compare_runs(
            baseline_dir=baseline,
            candidate_dir=candidate,
        )
    except ComparisonError as exc:
        typer.echo(f"Comparison error:\n{exc}", err=True)
        raise typer.Exit(code=1) from None

    typer.echo(
        "Success rate: "
        f"{comparison.baseline_success_rate:.1%} -> "
        f"{comparison.candidate_success_rate:.1%}"
    )

    typer.echo(
        f"Success delta: {comparison.success_rate_delta:+.1%}"
    )

    typer.echo(
        "Mean steps: "
        f"{comparison.baseline_mean_steps:.3f} -> "
        f"{comparison.candidate_mean_steps:.3f}"
    )

    typer.echo(
        f"Mean-step delta: {comparison.mean_steps_delta:+.3f}"
    )

    typer.echo(
        f"Gate: {'PASS' if comparison.gate_passed else 'FAIL'}"
    )

    for reason in comparison.reasons:
        typer.echo(f"Reason: {reason}")

    if not comparison.gate_passed:
        raise typer.Exit(code=2)


def main() -> None:
    """Run the FORGE-CI command-line application."""

    app()
