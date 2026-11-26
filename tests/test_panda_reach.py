"""Panda configuration and optional external-model integration tests."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from forge_ci.comparison import compare_runs
from forge_ci.config import ExperimentConfig, load_config
from forge_ci.runner import run_evaluation

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "mujoco_menagerie/franka_emika_panda/scene.xml"


@pytest.mark.parametrize("scenario", ["nominal", "bias", "delay", "noise"])
def test_panda_config(scenario):
    config = load_config(ROOT / f"configs/panda_{scenario}.yaml")
    assert config.environment.name == "panda_reach"
    assert config.policy.name == "panda_reach_ik"
    assert config.seed == 42
    assert config.episodes == 5


def test_panda_rejects_incompatible_policy():
    with pytest.raises(ValidationError, match="panda_reach requires"):
        ExperimentConfig.model_validate({
            "name": "bad-pair",
            "environment": {"name": "panda_reach"},
            "policy": {"name": "greedy"},
        })


@pytest.mark.skipif(not MODEL.exists(), reason="External pinned Menagerie model not installed")
def test_panda_nominal_and_bias_regression(tmp_path):
    pytest.importorskip("mujoco")
    nominal = load_config(ROOT / "configs/panda_nominal.yaml")
    biased = load_config(ROOT / "configs/panda_bias.yaml")
    for config in (nominal, biased):
        config.environment.model_path = str(MODEL)
        config.episodes = 1
    baseline = run_evaluation(nominal, tmp_path / "baseline")
    candidate = run_evaluation(biased, tmp_path / "candidate")
    assert baseline.summary.successes == 1
    assert candidate.summary.successes == 0
    comparison = compare_runs(baseline.run_dir, candidate.run_dir)
    assert not comparison.gate_passed
    assert comparison.success_rate_delta == -1.0


@pytest.mark.skipif(not MODEL.exists(), reason="External pinned Menagerie model not installed")
def test_panda_detects_delay_without_success_drop(tmp_path):
    pytest.importorskip("mujoco")
    nominal = load_config(ROOT / "configs/panda_nominal.yaml")
    delayed = load_config(ROOT / "configs/panda_delay.yaml")
    for config in (nominal, delayed):
        config.environment.model_path = str(MODEL)
        config.episodes = 1
    baseline = run_evaluation(nominal, tmp_path / "baseline")
    candidate = run_evaluation(delayed, tmp_path / "candidate")
    assert baseline.summary.successes == candidate.summary.successes == 1
    comparison = compare_runs(baseline.run_dir, candidate.run_dir)
    assert comparison.success_rate_delta == 0
    assert comparison.mean_steps_delta > 50
    assert not comparison.gate_passed


def test_panda_legacy_step_metric_is_horizon():
    config = ExperimentConfig.model_validate({
        "name": "legacy",
        "environment": {"name": "panda_reach"},
        "policy": {"name": "panda_reach_ik"},
    })
    assert config.environment.step_metric == "horizon"


@pytest.mark.skipif(not MODEL.exists(), reason="External pinned Menagerie model not installed")
def test_panda_feedback_uses_same_seed_and_target():
    pytest.importorskip("mujoco")
    from forge_ci.simulators.panda_reach import run_panda_episode

    original = run_panda_episode(MODEL, seed=42, command_bias_rad=0.04)
    feedback = run_panda_episode(
        MODEL, seed=42, command_bias_rad=0.04, cartesian_feedback_gain=0.12
    )
    repeated = run_panda_episode(
        MODEL, seed=42, command_bias_rad=0.04, cartesian_feedback_gain=0.12
    )
    assert original["target_xyz_m"] == feedback["target_xyz_m"]
    assert original["peak_joint_correction_rad"] == 0.0
    assert 0.0 < feedback["peak_joint_correction_rad"] <= 0.25
    assert feedback == repeated
    assert not original["passed"]
    assert feedback["passed"]
    assert feedback["final_position_error_m"] < original["final_position_error_m"]
