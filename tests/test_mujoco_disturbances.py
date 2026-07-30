"""Tests for deterministic MuJoCo disturbance injection."""

import json
from pathlib import Path

from forge_ci.config import ExperimentConfig
from forge_ci.runner import run_evaluation


def _disturbed_config() -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "name": "disturbed-controller",
            "seed": 42,
            "episodes": 4,
            "environment": {
                "name": "mujoco_reach",
                "target_position": 0.8,
                "max_steps": 1500,
                "position_tolerance": 0.01,
                "velocity_tolerance": 0.02,
                "initial_position_low": 0.0,
                "initial_position_high": 0.1,
                "actuator_delay_steps": 12,
                "control_noise_std": 0.002,
                "joint_damping": 4.0,
            },
            "policy": {
                "name": "position_servo",
                "kp": 40.0,
                "target_bias": 0.0,
            },
            "gate": {
                "min_success_rate": 1.0,
                "max_success_rate_drop": 0.0,
                "max_mean_steps_increase": 100.0,
            },
        }
    )


def test_disturbed_evaluation_is_repeatable(
    tmp_path: Path,
) -> None:
    config = _disturbed_config()

    first = run_evaluation(
        config,
        tmp_path / "first",
    )

    second = run_evaluation(
        config,
        tmp_path / "second",
    )

    first_episodes = (
        first.run_dir / "episodes.jsonl"
    ).read_text(encoding="utf-8")

    second_episodes = (
        second.run_dir / "episodes.jsonl"
    ).read_text(encoding="utf-8")

    assert first.summary == second.summary
    assert first_episodes == second_episodes


def test_disturbance_metadata_is_recorded(
    tmp_path: Path,
) -> None:
    evaluation = run_evaluation(
        _disturbed_config(),
        tmp_path / "runs",
    )

    episode_line = (
        evaluation.run_dir / "episodes.jsonl"
    ).read_text(encoding="utf-8").splitlines()[0]

    episode = json.loads(episode_line)
    disturbances = episode["disturbance_parameters"]

    assert disturbances["actuator_delay_steps"] == 12
    assert disturbances["control_noise_std"] == 0.002
    assert disturbances["joint_damping"] == 4.0
