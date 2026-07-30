"""Minimal deterministic MuJoCo physics probe."""

from dataclasses import dataclass

from forge_ci.simulators.mujoco_reach import (
    run_reach_episode,
)


@dataclass(frozen=True)
class ProbeResult:
    """Result of one deterministic MuJoCo simulator probe."""

    seed: int
    success: bool
    steps: int

    initial_position: float
    final_position: float
    final_velocity: float
    target_position: float


def run_probe(seed: int = 42) -> ProbeResult:
    """Run the standard standalone MuJoCo probe."""

    result = run_reach_episode(
        seed=seed,
        target_position=0.8,
        max_steps=1000,
        position_tolerance=0.01,
        velocity_tolerance=0.02,
        initial_position_low=0.0,
        initial_position_high=0.1,
        controller_kp=40.0,
        target_bias=0.0,
        actuator_delay_steps=0,
        control_noise_std=0.0,
        joint_damping=4.0,
    )

    return ProbeResult(
        seed=result.seed,
        success=result.success,
        steps=result.steps,
        initial_position=result.initial_position,
        final_position=result.final_position,
        final_velocity=result.final_velocity,
        target_position=result.target_position,
    )
