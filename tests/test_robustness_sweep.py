"""Tests for MuJoCo robustness sweeps."""

from pathlib import Path

from typer.testing import CliRunner

from forge_ci.cli import app
from forge_ci.sweep import run_robustness_sweep
from forge_ci.sweep_config import RobustnessSweepConfig

runner = CliRunner()


def _sweep_config(
    *,
    max_steps: int = 1200,
) -> RobustnessSweepConfig:
    return RobustnessSweepConfig.model_validate(
        {
            "name": "test-sweep",
            "base_experiment": {
                "name": "base",
                "seed": 42,
                "episodes": 2,
                "environment": {
                    "name": "mujoco_reach",
                    "target_position": 0.8,
                    "max_steps": max_steps,
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
                    "target_bias": 0.0,
                },
                "gate": {
                    "min_success_rate": 0.0,
                },
            },
            "scenarios": [
                {
                    "name": "nominal",
                    "actuator_delay_steps": 0,
                    "control_noise_std": 0.0,
                    "joint_damping": 4.0,
                },
                {
                    "name": "delayed",
                    "actuator_delay_steps": 10,
                    "control_noise_std": 0.0,
                    "joint_damping": 4.0,
                },
            ],
            "gate": {
                "min_scenario_success_rate": 1.0,
                "max_scenario_mean_steps": 1200.0,
            },
        }
    )


def test_sweep_is_repeatable_and_writes_artifacts(
    tmp_path: Path,
) -> None:
    config = _sweep_config()

    first = run_robustness_sweep(
        config,
        tmp_path / "first",
    )

    second = run_robustness_sweep(
        config,
        tmp_path / "second",
    )

    first_metrics = [
        (
            result.scenario_name,
            result.success_rate,
            result.mean_steps,
        )
        for result in first.summary.scenarios
    ]

    second_metrics = [
        (
            result.scenario_name,
            result.success_rate,
            result.mean_steps,
        )
        for result in second.summary.scenarios
    ]

    assert first_metrics == second_metrics
    assert first.summary.gate_passed is True

    assert (
        first.sweep_dir / "sweep_manifest.json"
    ).exists()

    assert (
        first.sweep_dir / "sweep_summary.json"
    ).exists()

    assert (
        first.sweep_dir / "robustness_matrix.csv"
    ).exists()


def test_sweep_gate_fails_when_step_budget_is_impossible(
    tmp_path: Path,
) -> None:
    config = _sweep_config(max_steps=20)

    result = run_robustness_sweep(
        config,
        tmp_path / "runs",
    )

    assert result.summary.gate_passed is False
    assert result.summary.worst_success_rate == 0.0
    assert result.summary.reasons


def test_sweep_cli_generates_matrix(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "sweep.yaml"

    config_path.write_text(
        """name: cli-sweep

base_experiment:
  name: cli-base
  seed: 42
  episodes: 1

  environment:
    name: mujoco_reach
    target_position: 0.8
    max_steps: 1200
    position_tolerance: 0.01
    velocity_tolerance: 0.02
    initial_position_low: 0.0
    initial_position_high: 0.1

  policy:
    name: position_servo
    kp: 40.0
    target_bias: 0.0

  gate:
    min_success_rate: 0.0

scenarios:
  - name: nominal
    actuator_delay_steps: 0
    control_noise_std: 0.0
    joint_damping: 4.0

gate:
  min_scenario_success_rate: 1.0
  max_scenario_mean_steps: 1200.0
""",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "sweep",
            str(config_path),
            "--output-dir",
            str(tmp_path / "sweeps"),
        ],
    )

    assert result.exit_code == 0
    assert "nominal:" in result.stdout
    assert "Gate: PASS" in result.stdout

    matrices = list(
        (tmp_path / "sweeps").glob(
            "*/robustness_matrix.csv"
        )
    )

    assert len(matrices) == 1
