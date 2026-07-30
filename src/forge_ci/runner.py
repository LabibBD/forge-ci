"""Evaluation runner and artifact writer."""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from random import Random
from statistics import fmean

from forge_ci import __version__
from forge_ci.config import (
    ExperimentConfig,
    LineWorldEnvironmentConfig,
    MujocoReachEnvironmentConfig,
    PositionServoPolicyConfig,
)
from forge_ci.models import EpisodeResult, RunSummary
from forge_ci.toy import (
    AlternatingPolicy,
    GreedyPolicy,
    LineWorld,
)


@dataclass(frozen=True)
class EvaluationRun:
    """Paths and summary returned after one completed evaluation."""

    run_dir: Path
    summary: RunSummary


def _write_json(path: Path, payload: object) -> None:
    """Write formatted JSON with a trailing newline."""

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _config_digest(config: ExperimentConfig) -> str:
    """Return a stable SHA-256 digest of an experiment configuration."""

    canonical = json.dumps(
        config.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    )

    return sha256(
        canonical.encode("utf-8")
    ).hexdigest()


def _run_line_world_episode(
    config: ExperimentConfig,
    episode: int,
) -> EpisodeResult:
    """Run one LineWorld episode."""

    environment_config = config.environment

    if not isinstance(
        environment_config,
        LineWorldEnvironmentConfig,
    ):
        raise TypeError(
            "Expected a LineWorld environment configuration."
        )

    episode_seed = config.seed + episode
    random_generator = Random(episode_seed)

    environment = LineWorld(
        goal=environment_config.goal,
        max_steps=environment_config.max_steps,
        slip_probability=(
            environment_config.slip_probability
        ),
    )

    if config.policy.name == "greedy":
        policy = GreedyPolicy(
            goal=environment_config.goal
        )
    else:
        policy = AlternatingPolicy(
            goal=environment_config.goal
        )

    observation = environment.reset()
    total_reward = 0.0
    success = False

    while True:
        action = policy.act(observation)

        observation, reward, done, success = (
            environment.step(
                action,
                random_generator,
            )
        )

        total_reward += reward

        if done:
            break

    return EpisodeResult(
        episode=episode,
        seed=episode_seed,
        environment_name=environment_config.name,
        policy_name=config.policy.name,
        success=success,
        steps=environment.steps,
        total_reward=round(total_reward, 6),
        final_position=float(environment.position),
        final_velocity=None,
        failure_reason=(
            None
            if success
            else "max_steps_exceeded"
        ),
    )


def _run_mujoco_reach_episode(
    config: ExperimentConfig,
    episode: int,
) -> EpisodeResult:
    """Run one MuJoCo reaching episode."""

    environment_config = config.environment

    if not isinstance(
        environment_config,
        MujocoReachEnvironmentConfig,
    ):
        raise TypeError(
            "Expected a MuJoCo reach environment configuration."
        )

    policy_config = config.policy

    if not isinstance(
        policy_config,
        PositionServoPolicyConfig,
    ):
        raise TypeError(
            "Expected a position-servo policy configuration."
        )

    from forge_ci.simulators.mujoco_reach import (
        run_reach_episode,
    )

    episode_seed = config.seed + episode

    result = run_reach_episode(
        seed=episode_seed,
        target_position=(
            environment_config.target_position
        ),
        max_steps=environment_config.max_steps,
        position_tolerance=(
            environment_config.position_tolerance
        ),
        velocity_tolerance=(
            environment_config.velocity_tolerance
        ),
        initial_position_low=(
            environment_config.initial_position_low
        ),
        initial_position_high=(
            environment_config.initial_position_high
        ),
        controller_kp=policy_config.kp,
        target_bias=policy_config.target_bias,
        actuator_delay_steps=(
            environment_config.actuator_delay_steps
        ),
        control_noise_std=(
            environment_config.control_noise_std
        ),
        joint_damping=environment_config.joint_damping,
    )

    return EpisodeResult(
        episode=episode,
        seed=episode_seed,
        environment_name=environment_config.name,
        policy_name=config.policy.name,
        success=result.success,
        steps=result.steps,
        total_reward=(
            1.0 if result.success else 0.0
        ),
        final_position=result.final_position,
        final_velocity=result.final_velocity,
        failure_reason=result.failure_reason,
        policy_parameters={
            "kp": policy_config.kp,
            "target_bias": policy_config.target_bias,
            "commanded_target": result.commanded_target,
        },
        disturbance_parameters={
            "actuator_delay_steps": (
                result.actuator_delay_steps
            ),
            "control_noise_std": result.control_noise_std,
            "joint_damping": result.joint_damping,
        },
        diagnostics={
            "target_position": result.target_position,
            "initial_position_error": (
                result.initial_position_error
            ),
            "final_position_error": (
                result.final_position_error
            ),
            "mean_position_error": (
                result.mean_position_error
            ),
            "peak_abs_velocity": (
                result.peak_abs_velocity
            ),
            "overshoot_count": result.overshoot_count,
            "control_saturation_fraction": (
                result.control_saturation_fraction
            ),
        },
    )


def _run_episode(
    config: ExperimentConfig,
    episode: int,
) -> EpisodeResult:
    """Dispatch an episode to the selected environment."""

    if isinstance(
        config.environment,
        LineWorldEnvironmentConfig,
    ):
        return _run_line_world_episode(
            config,
            episode,
        )

    if isinstance(
        config.environment,
        MujocoReachEnvironmentConfig,
    ):
        return _run_mujoco_reach_episode(
            config,
            episode,
        )

    raise TypeError(
        "Unsupported environment configuration."
    )


def run_evaluation(
    config: ExperimentConfig,
    output_root: Path = Path("runs"),
) -> EvaluationRun:
    """Run all configured episodes and write evaluation artifacts."""

    episode_results = [
        _run_episode(config, episode)
        for episode in range(config.episodes)
    ]

    successes = sum(
        result.success
        for result in episode_results
    )

    failures = config.episodes - successes
    success_rate = successes / config.episodes

    mean_steps = fmean(
        result.steps
        for result in episode_results
    )

    summary = RunSummary(
        experiment_name=config.name,
        episodes=config.episodes,
        successes=successes,
        failures=failures,
        success_rate=success_rate,
        mean_steps=mean_steps,
        min_success_rate=(
            config.gate.min_success_rate
        ),
        gate_passed=(
            success_rate
            >= config.gate.min_success_rate
        ),
    )

    created_at = datetime.now(UTC)
    digest = _config_digest(config)

    run_id = (
        f"{created_at.strftime('%Y%m%dT%H%M%S%fZ')}-"
        f"{digest[:8]}"
    )

    run_dir = output_root / run_id

    run_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    manifest = {
        "schema_version": 2,
        "run_id": run_id,
        "created_at_utc": created_at.isoformat(),
        "forge_ci_version": __version__,
        "config_sha256": digest,
        "config": config.model_dump(mode="json"),
    }

    _write_json(
        run_dir / "manifest.json",
        manifest,
    )

    episodes_path = run_dir / "episodes.jsonl"

    with episodes_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for result in episode_results:
            payload = result.model_dump(
                mode="json"
            )

            file.write(
                json.dumps(
                    payload,
                    sort_keys=True,
                )
                + "\n"
            )

    _write_json(
        run_dir / "summary.json",
        summary.model_dump(mode="json"),
    )

    return EvaluationRun(
        run_dir=run_dir,
        summary=summary,
    )
