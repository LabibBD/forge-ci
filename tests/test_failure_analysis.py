"""Tests for automatic robot failure analysis."""

from pathlib import Path

from typer.testing import CliRunner

from forge_ci.cli import app
from forge_ci.config import ExperimentConfig
from forge_ci.failure_analysis import (
    analyze_run_failures,
)
from forge_ci.runner import run_evaluation

runner = CliRunner()


def _biased_config() -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "name": "biased-controller",
            "seed": 42,
            "episodes": 3,
            "environment": {
                "name": "mujoco_reach",
                "target_position": 0.8,
                "max_steps": 700,
                "position_tolerance": 0.01,
                "velocity_tolerance": 0.02,
                "initial_position_low": 0.0,
                "initial_position_high": 0.1,
                "actuator_delay_steps": 0,
                "control_noise_std": 0.0,
                "joint_damping": 4.0,
            },
            "policy": {
                "name": "position_servo",
                "kp": 40.0,
                "target_bias": -0.08,
            },
            "gate": {
                "min_success_rate": 1.0,
            },
        }
    )


def _successful_config() -> ExperimentConfig:
    payload = _biased_config().model_dump(
        mode="python"
    )

    payload["name"] = "successful-controller"
    payload["policy"]["target_bias"] = 0.0

    return ExperimentConfig.model_validate(payload)


def test_command_bias_failures_are_clustered(
    tmp_path: Path,
) -> None:
    evaluation = run_evaluation(
        _biased_config(),
        tmp_path / "runs",
    )

    analysis = analyze_run_failures(
        evaluation.run_dir
    )

    assert analysis.summary.failed_episodes == 3

    assert (
        analysis.summary.dominant_failure_type
        == "command_bias"
    )

    assert len(analysis.summary.clusters) == 1

    cluster = analysis.summary.clusters[0]

    assert cluster.failure_type == "command_bias"
    assert cluster.count == 3
    assert cluster.mean_confidence == 0.98

    assert analysis.analysis_path.exists()
    assert analysis.clusters_path.exists()


def test_successful_run_has_no_failure_clusters(
    tmp_path: Path,
) -> None:
    evaluation = run_evaluation(
        _successful_config(),
        tmp_path / "runs",
    )

    analysis = analyze_run_failures(
        evaluation.run_dir
    )

    assert analysis.summary.failed_episodes == 0
    assert analysis.summary.failure_rate == 0.0
    assert analysis.summary.clusters == []

    assert (
        analysis.summary.dominant_failure_type
        is None
    )


def test_failure_analysis_cli(
    tmp_path: Path,
) -> None:
    evaluation = run_evaluation(
        _biased_config(),
        tmp_path / "runs",
    )

    result = runner.invoke(
        app,
        [
            "analyze-failures",
            str(evaluation.run_dir),
        ],
    )

    assert result.exit_code == 0

    assert (
        "Dominant failure: command_bias"
        in result.stdout
    )

    assert "confidence=98.0%" in result.stdout
    assert "Evidence:" in result.stdout
