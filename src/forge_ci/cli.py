"""Command-line interface for FORGE-CI."""

from pathlib import Path
from typing import Annotated

import typer
import yaml
from pydantic import ValidationError

from forge_ci import __version__
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
    """Run an experiment and enforce its regression gate."""

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

    typer.echo(
        f"Required: {summary.min_success_rate:.1%}"
    )

    typer.echo(
        f"Gate: {'PASS' if summary.gate_passed else 'FAIL'}"
    )

    if not summary.gate_passed:
        raise typer.Exit(code=2)


def main() -> None:
    """Run the FORGE-CI command-line application."""

    app()
