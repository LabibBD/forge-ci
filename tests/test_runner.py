"""Tests for deterministic evaluation and gate decisions."""

from pathlib import Path

from forge_ci.config import ExperimentConfig
from forge_ci.runner import run_evaluation


def test_repeated_runs_produce_identical_results(
    tmp_path: Path,
) -> None:
    config = ExperimentConfig(
        name="repeatable",
        seed=17,
        episodes=6,
    )

    first = run_evaluation(
        config,
        tmp_path / "first",
    )

    second = run_evaluation(
        config,
        tmp_path / "second",
    )

    assert first.summary == second.summary

    first_episodes = (
        first.run_dir / "episodes.jsonl"
    ).read_text(encoding="utf-8")

    second_episodes = (
        second.run_dir / "episodes.jsonl"
    ).read_text(encoding="utf-8")

    assert first_episodes == second_episodes


def test_gate_fails_when_goal_cannot_be_reached(
    tmp_path: Path,
) -> None:
    config = ExperimentConfig.model_validate(
        {
            "name": "intentional-regression",
            "episodes": 4,
            "environment": {
                "name": "line_world",
                "goal": 5,
                "max_steps": 3,
            },
            "gate": {
                "min_success_rate": 1.0,
            },
        }
    )

    evaluation = run_evaluation(
        config,
        tmp_path / "runs",
    )

    assert evaluation.summary.success_rate == 0.0
    assert evaluation.summary.failures == 4
    assert evaluation.summary.gate_passed is False

    assert (
        evaluation.run_dir / "manifest.json"
    ).exists()

    assert (
        evaluation.run_dir / "summary.json"
    ).exists()
