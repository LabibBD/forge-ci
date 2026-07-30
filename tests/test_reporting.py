"""Tests for self-contained robustness reports."""

import json
from pathlib import Path

from typer.testing import CliRunner

from forge_ci.cli import app
from forge_ci.reporting import (
    EnvelopeReportError,
    render_envelope_report,
)

runner = CliRunner()


def _create_report_fixture(
    root: Path,
) -> Path:
    """Create a minimal valid envelope artifact tree."""

    envelope_dir = root / "envelope"
    search_dir = envelope_dir / "dimensions" / "00-delay" / "search"

    failure_run = search_dir / "trials" / "001-upper" / "run"

    failure_run.mkdir(
        parents=True,
    )

    envelope_summary = {
        "envelope_name": "test <envelope>",
        "all_converged": True,
        "weakest_dimension": "delay <unsafe>",
        "dimensions": [
            {
                "name": "delay <unsafe>",
                "parameter": "actuator_delay_steps",
                "direction": "positive",
                "baseline_parameter_value": 0,
                "largest_passing_magnitude": 99.0,
                "smallest_failing_magnitude": 100.0,
                "passing_applied_value": 99,
                "failing_applied_value": 100,
                "boundary_width": 1.0,
                "normalized_boundary_position": 0.5,
                "converged": True,
                "dominant_failure_type": ("delay_induced_oscillation"),
                "search_directory": ("dimensions/00-delay/search"),
            }
        ],
    }

    boundary_summary = {
        "search_name": "delay-boundary",
        "parameter": "actuator_delay_steps",
        "direction": "positive",
        "baseline_parameter_value": 0,
        "largest_passing_value": 99.0,
        "smallest_failing_value": 100.0,
        "passing_applied_value": 99,
        "failing_applied_value": 100,
        "boundary_width": 1.0,
        "converged": True,
        "search_iterations": 5,
        "counterexample_run_directory": ("trials/001-upper/run"),
        "counterexample_config": ("counterexample.yaml"),
        "dominant_failure_type": ("delay_induced_oscillation"),
        "trials": [
            {
                "index": 0,
                "label": "lower-bound",
                "parameter_value": 0.0,
                "applied_parameter_value": 0,
                "success_rate": 1.0,
                "mean_steps": 100.0,
                "gate_passed": True,
                "run_directory": ("trials/000-lower/run"),
            },
            {
                "index": 1,
                "label": "upper-bound",
                "parameter_value": 100.0,
                "applied_parameter_value": 100,
                "success_rate": 0.0,
                "mean_steps": 450.0,
                "gate_passed": False,
                "run_directory": ("trials/001-upper/run"),
            },
        ],
    }

    failure_analysis = {
        "run_id": "failure-run",
        "total_episodes": 3,
        "failed_episodes": 3,
        "failure_rate": 1.0,
        "dominant_failure_type": ("delay_induced_oscillation"),
        "clusters": [
            {
                "failure_type": ("delay_induced_oscillation"),
                "count": 3,
                "fraction_of_failures": 1.0,
                "mean_confidence": 0.88,
                "mean_severity": 4.2,
                "episodes": [0, 1, 2],
            }
        ],
        "failures": [
            {
                "episode": 0,
                "seed": 42,
                "failure_type": ("delay_induced_oscillation"),
                "confidence": 0.88,
                "severity": 4.2,
                "evidence": [
                    "Actuator delay was 100 steps.",
                    "The target was crossed 3 times.",
                ],
            }
        ],
    }

    (envelope_dir / "envelope_summary.json").write_text(
        json.dumps(envelope_summary),
        encoding="utf-8",
    )

    (search_dir / "boundary_summary.json").write_text(
        json.dumps(boundary_summary),
        encoding="utf-8",
    )

    (search_dir / "counterexample.yaml").write_text(
        "name: delay-counterexample\n",
        encoding="utf-8",
    )

    (failure_run / "failure_analysis.json").write_text(
        json.dumps(failure_analysis),
        encoding="utf-8",
    )

    (failure_run / "failure_clusters.csv").write_text(
        "failure_type,count\ndelay_induced_oscillation,3\n",
        encoding="utf-8",
    )

    return envelope_dir


def test_render_envelope_report(
    tmp_path: Path,
) -> None:
    envelope_dir = _create_report_fixture(tmp_path)

    report = render_envelope_report(envelope_dir)

    assert report.report_path.exists()
    assert report.data_path.exists()

    document = report.report_path.read_text(encoding="utf-8")

    assert "Robot Robustness Envelope" in document
    assert "delay &lt;unsafe&gt;" in document
    assert "delay_induced_oscillation" in document
    assert "Actuator delay was 100 steps." in document

    assert "<svg" in document
    assert "counterexample.yaml" in document
    assert "forge-report-data" in document


def test_render_report_cli(
    tmp_path: Path,
) -> None:
    envelope_dir = _create_report_fixture(tmp_path)

    output_path = tmp_path / "custom-report.html"

    result = runner.invoke(
        app,
        [
            "render-report",
            str(envelope_dir),
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    assert output_path.exists()
    assert "Report:" in result.stdout
    assert "Report data:" in result.stdout


def test_missing_envelope_summary_is_rejected(
    tmp_path: Path,
) -> None:
    try:
        render_envelope_report(tmp_path / "missing-envelope")
    except EnvelopeReportError as exc:
        assert "envelope_summary.json" in str(exc)
    else:
        raise AssertionError("Expected EnvelopeReportError.")
