"""Tests for real MuJoCo controller regression detection."""

from pathlib import Path

from forge_ci.comparison import compare_runs
from forge_ci.config import ExperimentConfig
from forge_ci.runner import run_evaluation


def _controller_config(
    *,
    name: str,
    kp: float,
) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "name": name,
            "seed": 42,
            "episodes": 3,
            "environment": {
                "name": "mujoco_reach",
                "target_position": 0.8,
                "max_steps": 1200,
                "position_tolerance": 0.01,
                "velocity_tolerance": 0.02,
                "initial_position_low": 0.0,
                "initial_position_high": 0.1,
            },
            "policy": {
                "name": "position_servo",
                "kp": kp,
                "target_bias": 0.0,
            },
            "gate": {
                "min_success_rate": 1.0,
                "max_success_rate_drop": 0.0,
                "max_mean_steps_increase": 100.0,
            },
        }
    )


def test_slow_mujoco_controller_is_rejected(
    tmp_path: Path,
) -> None:
    baseline = run_evaluation(
        _controller_config(
            name="baseline-controller",
            kp=40.0,
        ),
        tmp_path / "baseline",
    )

    candidate = run_evaluation(
        _controller_config(
            name="slow-controller",
            kp=3.0,
        ),
        tmp_path / "candidate",
    )

    assert baseline.summary.success_rate == 1.0
    assert candidate.summary.success_rate == 1.0

    assert (
        candidate.summary.mean_steps
        > baseline.summary.mean_steps
    )

    comparison = compare_runs(
        baseline.run_dir,
        candidate.run_dir,
    )

    assert comparison.gate_passed is False
    assert comparison.success_rate_delta == 0.0
    assert comparison.mean_steps_delta > 100.0

    assert any(
        "Mean-step increase" in reason
        for reason in comparison.reasons
    )
