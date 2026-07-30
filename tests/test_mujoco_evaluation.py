"""Tests for configuration-driven MuJoCo evaluation."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from forge_ci.cli import app
from forge_ci.config import ExperimentConfig
from forge_ci.runner import run_evaluation

runner = CliRunner()


def _mujoco_config() -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "name": "mujoco-test",
            "seed": 42,
            "episodes": 3,
            "environment": {
                "name": "mujoco_reach",
                "target_position": 0.8,
                "max_steps": 1000,
                "position_tolerance": 0.01,
                "velocity_tolerance": 0.02,
                "initial_position_low": 0.0,
                "initial_position_high": 0.1,
            },
            "policy": {
                "name": "position_servo",
            },
            "gate": {
                "min_success_rate": 1.0,
                "max_success_rate_drop": 0.0,
                "max_mean_steps_increase": 10.0,
            },
        }
    )


def test_mujoco_evaluation_is_repeatable(
    tmp_path: Path,
) -> None:
    config = _mujoco_config()

    first = run_evaluation(
        config,
        tmp_path / "first",
    )

    second = run_evaluation(
        config,
        tmp_path / "second",
    )

    assert first.summary == second.summary
    assert first.summary.success_rate == 1.0
    assert first.summary.gate_passed is True

    first_episodes = (
        first.run_dir / "episodes.jsonl"
    ).read_text(encoding="utf-8")

    second_episodes = (
        second.run_dir / "episodes.jsonl"
    ).read_text(encoding="utf-8")

    assert first_episodes == second_episodes

    first_episode = json.loads(
        first_episodes.splitlines()[0]
    )

    assert (
        first_episode["environment_name"]
        == "mujoco_reach"
    )

    assert (
        first_episode["policy_name"]
        == "position_servo"
    )

    assert first_episode["final_velocity"] is not None


def test_mujoco_rejects_incompatible_policy() -> None:
    with pytest.raises(
        ValidationError,
        match="position_servo",
    ):
        ExperimentConfig.model_validate(
            {
                "name": "invalid-mujoco-policy",
                "environment": {
                    "name": "mujoco_reach",
                },
                "policy": {
                    "name": "greedy",
                },
            }
        )


def test_mujoco_cli_evaluation_passes(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "mujoco.yaml"

    config_path.write_text(
        """name: mujoco-cli-test
seed: 42
episodes: 2

environment:
  name: mujoco_reach
  target_position: 0.8
  max_steps: 1000
  position_tolerance: 0.01
  velocity_tolerance: 0.02
  initial_position_low: 0.0
  initial_position_high: 0.1

policy:
  name: position_servo

gate:
  min_success_rate: 1.0
""",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "evaluate",
            str(config_path),
            "--output-dir",
            str(tmp_path / "runs"),
        ],
    )

    assert result.exit_code == 0
    assert "Success rate: 100.0%" in result.stdout
    assert "Gate: PASS" in result.stdout
