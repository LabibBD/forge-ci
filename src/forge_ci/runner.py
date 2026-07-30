"""Evaluation runner and artifact writer."""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from random import Random
from statistics import fmean

from forge_ci import __version__
from forge_ci.config import ExperimentConfig
from forge_ci.models import EpisodeResult, RunSummary
from forge_ci.toy import GreedyPolicy, LineWorld


@dataclass(frozen=True)
class EvaluationRun:
    """Paths and summary returned after one completed evaluation."""

    run_dir: Path
    summary: RunSummary


def _write_json(path: Path, payload: object) -> None:
    """Write formatted JSON with a trailing newline."""

    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _config_digest(config: ExperimentConfig) -> str:
    """Return a stable SHA-256 digest of an experiment configuration."""

    canonical = json.dumps(
        config.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    )

    return sha256(canonical.encode("utf-8")).hexdigest()


def _run_episode(
    config: ExperimentConfig,
    episode: int,
) -> EpisodeResult:
    """Run one independently seeded evaluation episode."""

    episode_seed = config.seed + episode
    rng = Random(episode_seed)

    environment = LineWorld(
        goal=config.environment.goal,
        max_steps=config.environment.max_steps,
        slip_probability=config.environment.slip_probability,
    )

    policy = GreedyPolicy(goal=config.environment.goal)

    observation = environment.reset()
    total_reward = 0.0
    success = False

    while True:
        action = policy.act(observation)

        observation, reward, done, success = environment.step(
            action,
            rng,
        )

        total_reward += reward

        if done:
            break

    return EpisodeResult(
        episode=episode,
        seed=episode_seed,
        success=success,
        steps=environment.steps,
        total_reward=round(total_reward, 6),
        final_position=environment.position,
        failure_reason=None if success else "max_steps_exceeded",
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
        result.success for result in episode_results
    )

    failures = config.episodes - successes
    success_rate = successes / config.episodes

    mean_steps = fmean(
        result.steps for result in episode_results
    )

    summary = RunSummary(
        experiment_name=config.name,
        episodes=config.episodes,
        successes=successes,
        failures=failures,
        success_rate=success_rate,
        mean_steps=mean_steps,
        min_success_rate=config.gate.min_success_rate,
        gate_passed=success_rate >= config.gate.min_success_rate,
    )

    created_at = datetime.now(UTC)
    digest = _config_digest(config)

    run_id = (
        f"{created_at.strftime('%Y%m%dT%H%M%S%fZ')}-"
        f"{digest[:8]}"
    )

    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    manifest = {
        "schema_version": 1,
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

    with episodes_path.open("w", encoding="utf-8") as file:
        for result in episode_results:
            payload = result.model_dump(mode="json")
            file.write(
                json.dumps(payload, sort_keys=True) + "\n"
            )

    _write_json(
        run_dir / "summary.json",
        summary.model_dump(mode="json"),
    )

    return EvaluationRun(
        run_dir=run_dir,
        summary=summary,
    )
