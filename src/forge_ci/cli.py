"""Command-line interface for FORGE-CI."""

from pathlib import Path
from typing import Annotated

import typer
import yaml
from pydantic import ValidationError

from forge_ci import __version__
from forge_ci.comparison import (
    ComparisonError,
    compare_runs,
)
from forge_ci.config import (
    ExperimentConfig,
    load_config,
)
from forge_ci.boundary_config import (
    BoundarySearchConfig,
    load_boundary_config,
)
from forge_ci.boundary_search import (
    BoundarySearchError,
    run_boundary_search,
)
from forge_ci.boundary_config import (
    BoundarySearchConfig,
    load_boundary_config,
)
from forge_ci.boundary_search import (
    BoundarySearchError,
    run_boundary_search,
)
from forge_ci.failure_analysis import (
    FailureAnalysisError,
    analyze_run_failures,
)
from forge_ci.runner import run_evaluation
from forge_ci.sweep import run_robustness_sweep
from forge_ci.sweep_config import (
    RobustnessSweepConfig,
    load_sweep_config,
)

app = typer.Typer(
    name="forgeci",
    help=(
        "Continuous evaluation for "
        "robot-learning policies."
    ),
    no_args_is_help=True,
)


def _load_config_or_exit(
    path: Path,
) -> ExperimentConfig:
    """Load an experiment or exit with a readable error."""

    try:
        return load_config(path)
    except FileNotFoundError:
        typer.echo(
            f"Configuration file not found: {path}",
            err=True,
        )
        raise typer.Exit(code=1) from None
    except (
        ValidationError,
        yaml.YAMLError,
        ValueError,
    ) as exc:
        typer.echo(
            f"Invalid configuration:\n{exc}",
            err=True,
        )
        raise typer.Exit(code=1) from None


def _load_sweep_or_exit(
    path: Path,
) -> RobustnessSweepConfig:
    """Load a robustness sweep or exit cleanly."""

    try:
        return load_sweep_config(path)
    except FileNotFoundError:
        typer.echo(
            f"Sweep configuration not found: {path}",
            err=True,
        )
        raise typer.Exit(code=1) from None
    except (
        ValidationError,
        yaml.YAMLError,
        ValueError,
    ) as exc:
        typer.echo(
            f"Invalid sweep configuration:\n{exc}",
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
            help="Directory for run artifacts.",
        ),
    ] = Path("runs"),
) -> None:
    """Run an experiment and enforce its absolute gate."""

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
        f"Gate: "
        f"{'PASS' if summary.gate_passed else 'FAIL'}"
    )

    if not summary.gate_passed:
        raise typer.Exit(code=2)


@app.command()
def compare(
    baseline: Path,
    candidate: Path,
) -> None:
    """Compare a candidate against a known-good run."""

    try:
        comparison = compare_runs(
            baseline_dir=baseline,
            candidate_dir=candidate,
        )
    except ComparisonError as exc:
        typer.echo(
            f"Comparison error:\n{exc}",
            err=True,
        )
        raise typer.Exit(code=1) from None

    typer.echo(
        "Success rate: "
        f"{comparison.baseline_success_rate:.1%} -> "
        f"{comparison.candidate_success_rate:.1%}"
    )

    typer.echo(
        f"Success delta: "
        f"{comparison.success_rate_delta:+.1%}"
    )

    typer.echo(
        "Mean steps: "
        f"{comparison.baseline_mean_steps:.3f} -> "
        f"{comparison.candidate_mean_steps:.3f}"
    )

    typer.echo(
        f"Mean-step delta: "
        f"{comparison.mean_steps_delta:+.3f}"
    )

    typer.echo(
        f"Gate: "
        f"{'PASS' if comparison.gate_passed else 'FAIL'}"
    )

    for reason in comparison.reasons:
        typer.echo(f"Reason: {reason}")

    if not comparison.gate_passed:
        raise typer.Exit(code=2)


@app.command()
def sweep(
    config: Path,
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            "-o",
            help="Directory for sweep artifacts.",
        ),
    ] = Path("runs/sweeps"),
) -> None:
    """Run a policy across multiple disturbances."""

    sweep_config = _load_sweep_or_exit(config)

    sweep_run = run_robustness_sweep(
        sweep_config,
        output_root=output_dir,
    )

    summary = sweep_run.summary

    typer.echo(f"Sweep: {sweep_run.sweep_dir}")

    for scenario in summary.scenarios:
        typer.echo(
            f"{scenario.scenario_name}: "
            f"success={scenario.success_rate:.1%}, "
            f"mean_steps={scenario.mean_steps:.3f}, "
            f"gate="
            f"{'PASS' if scenario.gate_passed else 'FAIL'}"
        )

    typer.echo(
        f"Worst success rate: "
        f"{summary.worst_success_rate:.1%}"
    )

    typer.echo(
        f"Worst mean steps: "
        f"{summary.worst_mean_steps:.3f}"
    )

    typer.echo(
        f"Gate: "
        f"{'PASS' if summary.gate_passed else 'FAIL'}"
    )

    for reason in summary.reasons:
        typer.echo(f"Reason: {reason}")

    if not summary.gate_passed:
        raise typer.Exit(code=2)


@app.command()
def analyze_failures(
    run_dir: Path,
) -> None:
    """Classify and cluster failed episodes."""

    try:
        analysis = analyze_run_failures(run_dir)
    except FailureAnalysisError as exc:
        typer.echo(
            f"Failure-analysis error:\n{exc}",
            err=True,
        )
        raise typer.Exit(code=1) from None

    summary = analysis.summary

    typer.echo(f"Analyzed run: {summary.run_id}")

    typer.echo(
        f"Failures: {summary.failed_episodes}/"
        f"{summary.total_episodes} "
        f"({summary.failure_rate:.1%})"
    )

    if summary.clusters:
        typer.echo(
            "Dominant failure: "
            f"{summary.dominant_failure_type}"
        )

        for cluster in summary.clusters:
            typer.echo(
                f"{cluster.failure_type}: "
                f"{cluster.count} episodes, "
                f"confidence="
                f"{cluster.mean_confidence:.1%}, "
                f"severity={cluster.mean_severity:.3f}"
            )

        first_failure = summary.failures[0]

        typer.echo("Evidence:")

        for evidence in first_failure.evidence:
            typer.echo(f"- {evidence}")
    else:
        typer.echo("No failed episodes were found.")

    typer.echo(
        f"Analysis: {analysis.analysis_path}"
    )

    typer.echo(
        f"Clusters: {analysis.clusters_path}"
    )


@app.command()
def discover_boundary(
    config: Path,
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            "-o",
            help="Directory for boundary-search artifacts.",
        ),
    ] = Path("runs/boundaries"),
) -> None:
    """Discover the smallest parameter value that fails."""

    try:
        search_config = load_boundary_config(config)

        search = run_boundary_search(
            search_config,
            output_root=output_dir,
        )
    except FileNotFoundError:
        typer.echo(
            f"Boundary configuration not found: {config}",
            err=True,
        )
        raise typer.Exit(code=1) from None
    except (
        BoundarySearchError,
        ValidationError,
        yaml.YAMLError,
        ValueError,
    ) as exc:
        typer.echo(
            f"Boundary-search error:\n{exc}",
            err=True,
        )
        raise typer.Exit(code=1) from None

    summary = search.summary

    typer.echo(f"Search: {search.search_dir}")

    typer.echo(
        f"Largest passing {summary.parameter}: "
        f"{summary.largest_passing_value:.6f}"
    )

    typer.echo(
        f"Smallest failing {summary.parameter}: "
        f"{summary.smallest_failing_value:.6f}"
    )

    typer.echo(
        f"Boundary width: "
        f"{summary.boundary_width:.6f}"
    )

    typer.echo(
        f"Converged: "
        f"{'YES' if summary.converged else 'NO'}"
    )

    typer.echo(
        "Counterexample diagnosis: "
        f"{summary.dominant_failure_type}"
    )

    typer.echo(
        "Counterexample: "
        f"{search.search_dir / summary.counterexample_config}"
    )


@app.command()
def discover_boundary(
    config: Path,
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            "-o",
            help="Directory for boundary-search artifacts.",
        ),
    ] = Path("runs/boundaries"),
) -> None:
    """Discover the smallest parameter value that fails."""

    try:
        search_config = load_boundary_config(config)

        search = run_boundary_search(
            search_config,
            output_root=output_dir,
        )
    except FileNotFoundError:
        typer.echo(
            f"Boundary configuration not found: {config}",
            err=True,
        )
        raise typer.Exit(code=1) from None
    except (
        BoundarySearchError,
        ValidationError,
        yaml.YAMLError,
        ValueError,
    ) as exc:
        typer.echo(
            f"Boundary-search error:\n{exc}",
            err=True,
        )
        raise typer.Exit(code=1) from None

    summary = search.summary

    typer.echo(f"Search: {search.search_dir}")

    typer.echo(
        f"Largest passing {summary.parameter}: "
        f"{summary.largest_passing_value:.6f}"
    )

    typer.echo(
        f"Smallest failing {summary.parameter}: "
        f"{summary.smallest_failing_value:.6f}"
    )

    typer.echo(
        f"Boundary width: "
        f"{summary.boundary_width:.6f}"
    )

    typer.echo(
        f"Converged: "
        f"{'YES' if summary.converged else 'NO'}"
    )

    typer.echo(
        "Counterexample diagnosis: "
        f"{summary.dominant_failure_type}"
    )

    typer.echo(
        "Counterexample: "
        f"{search.search_dir / summary.counterexample_config}"
    )


def main() -> None:
    """Run the FORGE-CI command-line application."""

    app()
