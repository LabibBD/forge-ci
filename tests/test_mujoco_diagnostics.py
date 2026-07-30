"""Tests for MuJoCo episode diagnostic telemetry."""

import json
from pathlib import Path

from forge_ci.config import ExperimentConfig
from forge_ci.runner import run_evaluation


def _diagnostic_config() -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "name": "diagnostic-test",
            "seed": 42,
            "episodes": 2,
            "environment": {
                "name": "mujoco_reach",
                "target_position": 0.8,
                "max_steps": 1000,
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
        }
    )


def _load_first_episode(run_dir: Path) -> dict[str, object]:
    line = (
        run_dir / "episodes.jsonl"
    ).read_text(encoding="utf-8").splitlines()[0]

    return json.loads(line)


def test_mujoco_diagnostics_are_recorded(
    tmp_path: Path,
) -> None:
    evaluation = run_evaluation(
        _diagnostic_config(),
        tmp_path / "runs",
    )

    episode = _load_first_episode(
        evaluation.run_dir
    )

    diagnostics = episode["diagnostics"]

    assert isinstance(diagnostics, dict)

    assert diagnostics["target_position"] == 0.8

    assert (
        diagnostics["initial_position_error"]
        > diagnostics["final_position_error"]
    )

    assert diagnostics["mean_position_error"] > 0.0
    assert diagnostics["peak_abs_velocity"] > 0.0

    assert isinstance(
        diagnostics["overshoot_count"],
        int,
    )

    assert (
        0.0
        <= diagnostics["control_saturation_fraction"]
        <= 1.0
    )


def test_mujoco_diagnostics_are_repeatable(
    tmp_path: Path,
) -> None:
    config = _diagnostic_config()

    first = run_evaluation(
        config,
        tmp_path / "first",
    )

    second = run_evaluation(
        config,
        tmp_path / "second",
    )

    first_records = (
        first.run_dir / "episodes.jsonl"
    ).read_text(encoding="utf-8")

    second_records = (
        second.run_dir / "episodes.jsonl"
    ).read_text(encoding="utf-8")

    assert first_records == second_records
