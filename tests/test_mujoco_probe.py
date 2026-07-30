"""Tests for the minimal MuJoCo simulator integration."""

import pytest

from forge_ci.simulators.mujoco_probe import run_probe


def test_mujoco_probe_reaches_target() -> None:
    result = run_probe(seed=42)

    assert result.success is True

    assert result.final_position == pytest.approx(
        result.target_position,
        abs=0.01,
    )

    assert abs(result.final_velocity) <= 0.02
    assert result.steps <= 1000


def test_mujoco_probe_is_deterministic() -> None:
    first = run_probe(seed=91)
    second = run_probe(seed=91)

    assert first == second
