"""Tests for baseline-versus-candidate regression comparison."""

from pathlib import Path

from typer.testing import CliRunner

from forge_ci.cli import app
from forge_ci.comparison import compare_runs
from forge_ci.config import ExperimentConfig
from forge_ci.runner import run_evaluation

runner = CliRunner()


def _greedy_config() -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "name": "greedy-baseline",
            "seed": 12,
            "episodes": 5,
            "environment": {
                "goal": 5,
                "max_steps": 10,
                "slip_probability": 0.0,
            },
            "policy": {
                "name": "greedy",
            },
            "gate": {
                "min_success_rate": 1.0,
                "max_success_rate_drop": 0.0,
                "max_mean_steps_increase": 1.0,
            },
        }
    )


def _alternating_config() -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "name": "alternating-candidate",
            "seed": 12,
            "episodes": 5,
            "environment": {
                "goal": 5,
                "max_steps": 10,
                "slip_probability": 0.0,
            },
            "policy": {
                "name": "alternating",
            },
            "gate": {
                "min_success_rate": 1.0,
                "max_success_rate_drop": 0.0,
                "max_mean_steps_increase": 1.0,
            },
        }
    )


def test_identical_policy_comparison_passes(
    tmp_path: Path,
) -> None:
    baseline = run_evaluation(
        _greedy_config(),
        tmp_path / "baseline",
    )

    candidate = run_evaluation(
        _greedy_config(),
        tmp_path / "candidate",
    )

    result = compare_runs(
        baseline.run_dir,
        candidate.run_dir,
    )

    assert result.gate_passed is True
    assert result.success_rate_delta == 0.0
    assert result.mean_steps_delta == 0.0
    assert (candidate.run_dir / "comparison.json").exists()


def test_slower_candidate_fails_comparison(
    tmp_path: Path,
) -> None:
    baseline = run_evaluation(
        _greedy_config(),
        tmp_path / "baseline",
    )

    candidate = run_evaluation(
        _alternating_config(),
        tmp_path / "candidate",
    )

    result = compare_runs(
        baseline.run_dir,
        candidate.run_dir,
    )

    assert result.gate_passed is False
    assert result.success_rate_delta == 0.0
    assert result.mean_steps_delta == 4.0
    assert len(result.reasons) == 1


def test_compare_command_returns_two_for_regression(
    tmp_path: Path,
) -> None:
    baseline = run_evaluation(
        _greedy_config(),
        tmp_path / "baseline",
    )

    candidate = run_evaluation(
        _alternating_config(),
        tmp_path / "candidate",
    )

    result = runner.invoke(
        app,
        [
            "compare",
            str(baseline.run_dir),
            str(candidate.run_dir),
        ],
    )

    assert result.exit_code == 2
    assert "Gate: FAIL" in result.stdout
    assert "Mean-step increase" in result.stdout
