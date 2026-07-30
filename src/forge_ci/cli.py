"""Command-line interface for FORGE-CI."""

from pathlib import Path
from typing import Annotated

import typer
import yaml
from pydantic import ValidationError

from forge_ci import __version__
from forge_ci.boundary_config import (
    load_boundary_config,
)
from forge_ci.boundary_search import (
    BoundarySearchError,
    run_boundary_search,
)
from forge_ci.comparison import (
    ComparisonError,
    compare_runs,
)
from forge_ci.config import (
    ExperimentConfig,
    load_config,
)
from forge_ci.envelope_config import (
    load_envelope_config,
)
from forge_ci.envelope_search import (
    EnvelopeSearchError,
    run_robustness_envelope,
)
from forge_ci.failure_analysis import (
    FailureAnalysisError,
    analyze_run_failures,
)
from forge_ci.reporting import (
    EnvelopeReportError,
    render_envelope_report,
)
from forge_ci.runner import run_evaluation
from forge_ci.sweep import run_robustness_sweep
from forge_ci.sweep_config import (
    RobustnessSweepConfig,
    load_sweep_config,
)

app = typer.Typer(
    name="forgeci",
    help=("Continuous evaluation for robot-learning policies."),
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

    typer.echo(f"Success rate: {summary.success_rate:.1%} ({summary.successes}/{summary.episodes})")

    typer.echo(f"Required: {summary.min_success_rate:.1%}")

    typer.echo(f"Gate: {'PASS' if summary.gate_passed else 'FAIL'}")

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

    typer.echo(f"Success delta: {comparison.success_rate_delta:+.1%}")

    typer.echo(
        f"Mean steps: {comparison.baseline_mean_steps:.3f} -> {comparison.candidate_mean_steps:.3f}"
    )

    typer.echo(f"Mean-step delta: {comparison.mean_steps_delta:+.3f}")

    typer.echo(f"Gate: {'PASS' if comparison.gate_passed else 'FAIL'}")

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

    typer.echo(f"Worst success rate: {summary.worst_success_rate:.1%}")

    typer.echo(f"Worst mean steps: {summary.worst_mean_steps:.3f}")

    typer.echo(f"Gate: {'PASS' if summary.gate_passed else 'FAIL'}")

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
        f"Failures: {summary.failed_episodes}/{summary.total_episodes} ({summary.failure_rate:.1%})"
    )

    if summary.clusters:
        typer.echo(f"Dominant failure: {summary.dominant_failure_type}")

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

    typer.echo(f"Analysis: {analysis.analysis_path}")

    typer.echo(f"Clusters: {analysis.clusters_path}")


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

    typer.echo(f"Largest passing {summary.parameter}: {summary.largest_passing_value:.6f}")

    typer.echo(f"Smallest failing {summary.parameter}: {summary.smallest_failing_value:.6f}")

    typer.echo(f"Boundary width: {summary.boundary_width:.6f}")

    typer.echo(f"Converged: {'YES' if summary.converged else 'NO'}")

    typer.echo(f"Counterexample diagnosis: {summary.dominant_failure_type}")

    typer.echo(f"Counterexample: {search.search_dir / summary.counterexample_config}")


@app.command()
def discover_envelope(
    config: Path,
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            "-o",
            help="Directory for envelope artifacts.",
        ),
    ] = Path("runs/envelopes"),
) -> None:
    """Discover and rank several robustness boundaries."""

    try:
        envelope_config = load_envelope_config(config)

        envelope = run_robustness_envelope(
            envelope_config,
            output_root=output_dir,
        )
    except FileNotFoundError:
        typer.echo(
            f"Envelope configuration not found: {config}",
            err=True,
        )
        raise typer.Exit(code=1) from None
    except (
        EnvelopeSearchError,
        ValidationError,
        yaml.YAMLError,
        ValueError,
    ) as exc:
        typer.echo(
            f"Envelope-search error:\n{exc}",
            err=True,
        )
        raise typer.Exit(code=1) from None

    summary = envelope.summary

    typer.echo(f"Envelope: {envelope.envelope_dir}")

    typer.echo(f"Weakest dimension: {summary.weakest_dimension}")

    for rank, dimension in enumerate(
        summary.dimensions,
        start=1,
    ):
        typer.echo(
            f"{rank}. {dimension.name}: "
            f"{dimension.parameter}/"
            f"{dimension.direction}, "
            f"failure magnitude="
            f"{dimension.smallest_failing_magnitude:.6f}, "
            f"normalized="
            f"{dimension.normalized_boundary_position:.1%}, "
            f"diagnosis="
            f"{dimension.dominant_failure_type}"
        )

    typer.echo(f"Converged: {'YES' if summary.all_converged else 'NO'}")


@app.command()
def render_report(
    envelope_dir: Path,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output HTML path.",
        ),
    ] = None,
) -> None:
    """Render a self-contained envelope HTML report."""

    try:
        report = render_envelope_report(
            envelope_dir,
            output_path=output,
        )
    except EnvelopeReportError as exc:
        typer.echo(
            f"Report-generation error:\n{exc}",
            err=True,
        )
        raise typer.Exit(code=1) from None

    typer.echo(f"Report: {report.report_path}")

    typer.echo(f"Report data: {report.data_path}")


def main() -> None:
    """Run the FORGE-CI command-line application."""

    app()
