"""Tests for automatic failure-boundary discovery."""

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from forge_ci.boundary_config import (
    BoundarySearchConfig,
    build_boundary_experiment,
)
from forge_ci.boundary_search import (
    run_boundary_search,
)
from forge_ci.cli import app

runner = CliRunner()


def _boundary_config() -> BoundarySearchConfig:
    return BoundarySearchConfig.model_validate(
        {
            "name": "test-bias-boundary",
            "base_experiment": {
                "name": "base",
                "seed": 42,
                "episodes": 2,
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
                    "target_bias": 0.0,
                },
                "gate": {
                    "min_success_rate": 1.0,
                },
            },
            "parameter": "target_bias",
            "direction": "negative",
            "low": 0.0,
            "high": 0.08,
            "tolerance": 0.001,
            "max_iterations": 10,
        }
    )


def test_boundary_search_finds_minimal_bias_failure(
    tmp_path: Path,
) -> None:
    search = run_boundary_search(
        _boundary_config(),
        tmp_path / "searches",
    )

    summary = search.summary

    assert summary.converged is True

    assert summary.largest_passing_value < summary.smallest_failing_value

    assert summary.boundary_width <= 0.001

    assert 0.0 < summary.smallest_failing_value <= _boundary_config().high

    assert summary.trials[0].label == "lower-bound"
    assert summary.trials[0].gate_passed is True

    assert summary.trials[1].label == "upper-bound"
    assert summary.trials[1].gate_passed is False

    assert summary.dominant_failure_type == "command_bias"

    counterexample_path = search.search_dir / "counterexample.yaml"

    assert counterexample_path.exists()

    counterexample = yaml.safe_load(counterexample_path.read_text(encoding="utf-8"))

    discovered_bias = counterexample["policy"]["target_bias"]

    assert discovered_bias < 0.0

    assert abs(discovered_bias) == pytest.approx(summary.smallest_failing_value)

    assert (search.search_dir / "boundary_summary.json").exists()

    assert (search.search_dir / "boundary_trials.csv").exists()


def test_boundary_search_is_repeatable(
    tmp_path: Path,
) -> None:
    config = _boundary_config()

    first = run_boundary_search(
        config,
        tmp_path / "first",
    )

    second = run_boundary_search(
        config,
        tmp_path / "second",
    )

    assert first.summary.largest_passing_value == second.summary.largest_passing_value

    assert first.summary.smallest_failing_value == second.summary.smallest_failing_value

    first_values = [trial.parameter_value for trial in first.summary.trials]

    second_values = [trial.parameter_value for trial in second.summary.trials]

    assert first_values == second_values


@pytest.mark.parametrize(
    (
        "parameter",
        "direction",
        "high",
        "tolerance",
        "magnitude",
        "expected",
    ),
    [
        (
            "kp",
            "negative",
            10.0,
            0.1,
            5.0,
            35.0,
        ),
        (
            "control_noise_std",
            "positive",
            0.1,
            0.001,
            0.025,
            0.025,
        ),
        (
            "joint_damping",
            "positive",
            10.0,
            0.1,
            5.0,
            9.0,
        ),
        (
            "actuator_delay_steps",
            "positive",
            20.0,
            1.0,
            7.0,
            7,
        ),
    ],
)
def test_generic_parameter_is_applied(
    parameter: str,
    direction: str,
    high: float,
    tolerance: float,
    magnitude: float,
    expected: float | int,
) -> None:
    payload = _boundary_config().model_dump(mode="python")

    payload["parameter"] = parameter
    payload["direction"] = direction
    payload["high"] = high
    payload["tolerance"] = tolerance

    config = BoundarySearchConfig.model_validate(payload)

    experiment = build_boundary_experiment(
        config,
        magnitude=magnitude,
        name="generic-parameter-test",
    )

    container = experiment.policy if parameter in {"target_bias", "kp"} else experiment.environment

    assert getattr(
        container,
        parameter,
    ) == pytest.approx(expected)


def test_integer_parameter_rejects_fractional_bounds() -> None:
    payload = _boundary_config().model_dump(mode="python")

    payload["parameter"] = "actuator_delay_steps"
    payload["direction"] = "positive"
    payload["high"] = 20.5
    payload["tolerance"] = 1.0

    with pytest.raises(
        ValueError,
        match="whole-number magnitudes",
    ):
        BoundarySearchConfig.model_validate(payload)


def test_boundary_search_cli(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "boundary.yaml"

    config_path.write_text(
        """name: cli-boundary

base_experiment:
  name: base
  seed: 42
  episodes: 1

  environment:
    name: mujoco_reach
    target_position: 0.8
    max_steps: 700
    position_tolerance: 0.01
    velocity_tolerance: 0.02
    initial_position_low: 0.0
    initial_position_high: 0.1

  policy:
    name: position_servo
    kp: 40.0
    target_bias: 0.0

  gate:
    min_success_rate: 1.0

parameter: target_bias
direction: negative
low: 0.0
high: 0.08
tolerance: 0.002
max_iterations: 10
""",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "discover-boundary",
            str(config_path),
            "--output-dir",
            str(tmp_path / "searches"),
        ],
    )

    assert result.exit_code == 0
    assert "Largest passing target_bias" in result.stdout
    assert "Smallest failing target_bias" in result.stdout
    assert "Counterexample diagnosis: command_bias" in result.stdout
