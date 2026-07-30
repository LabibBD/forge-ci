"""Deterministic failure classification and clustering."""

import csv
import json
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from forge_ci.config import (
    ExperimentConfig,
    MujocoReachEnvironmentConfig,
    PositionServoPolicyConfig,
)
from forge_ci.models import EpisodeResult


class FailureAnalysisError(ValueError):
    """Raised when run artifacts cannot be analyzed safely."""


class EpisodeFailure(BaseModel):
    """Failure classification for one episode."""

    episode: int = Field(ge=0)
    seed: int

    failure_type: str

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    severity: float = Field(ge=0.0)
    evidence: list[str]


class FailureCluster(BaseModel):
    """Aggregate statistics for one failure type."""

    failure_type: str
    count: int = Field(gt=0)

    fraction_of_failures: float = Field(
        ge=0.0,
        le=1.0,
    )

    mean_confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    mean_severity: float = Field(ge=0.0)
    episodes: list[int]


class FailureAnalysisSummary(BaseModel):
    """Complete analysis report for one evaluation run."""

    run_id: str

    total_episodes: int = Field(ge=0)
    failed_episodes: int = Field(ge=0)

    failure_rate: float = Field(
        ge=0.0,
        le=1.0,
    )

    dominant_failure_type: str | None

    clusters: list[FailureCluster]
    failures: list[EpisodeFailure]


@dataclass(frozen=True)
class FailureAnalysisRun:
    """Files and summary produced by failure analysis."""

    analysis_path: Path
    clusters_path: Path
    summary: FailureAnalysisSummary


def _load_json_object(
    path: Path,
) -> dict[str, Any]:
    """Load a JSON file and require an object at its root."""

    try:
        payload: Any = json.loads(
            path.read_text(encoding="utf-8")
        )
    except FileNotFoundError:
        raise FailureAnalysisError(
            f"Required artifact not found: {path}"
        ) from None
    except (OSError, JSONDecodeError) as exc:
        raise FailureAnalysisError(
            f"Could not read {path}: {exc}"
        ) from exc

    if not isinstance(payload, dict):
        raise FailureAnalysisError(
            f"Artifact must contain a JSON object: {path}"
        )

    return payload


def _load_episodes(
    path: Path,
) -> list[EpisodeResult]:
    """Load and validate episode JSON-lines records."""

    try:
        lines = path.read_text(
            encoding="utf-8"
        ).splitlines()
    except FileNotFoundError:
        raise FailureAnalysisError(
            f"Required artifact not found: {path}"
        ) from None
    except OSError as exc:
        raise FailureAnalysisError(
            f"Could not read {path}: {exc}"
        ) from exc

    episodes: list[EpisodeResult] = []

    for line_number, line in enumerate(
        lines,
        start=1,
    ):
        if not line.strip():
            continue

        try:
            payload: Any = json.loads(line)
        except JSONDecodeError as exc:
            raise FailureAnalysisError(
                f"Invalid JSON on line {line_number} "
                f"of {path}: {exc}"
            ) from exc

        try:
            episode = EpisodeResult.model_validate(
                payload
            )
        except ValidationError as exc:
            raise FailureAnalysisError(
                f"Invalid episode on line {line_number} "
                f"of {path}:\n{exc}"
            ) from exc

        episodes.append(episode)

    return episodes


def _diagnostic_float(
    episode: EpisodeResult,
    key: str,
    default: float = 0.0,
) -> float:
    """Read one numeric diagnostic value."""

    value = episode.diagnostics.get(
        key,
        default,
    )

    return float(value)


def _policy_float(
    episode: EpisodeResult,
    key: str,
    default: float = 0.0,
) -> float:
    """Read one numeric policy parameter."""

    value = episode.policy_parameters.get(
        key,
        default,
    )

    return float(value)


def _classify_mujoco_failure(
    episode: EpisodeResult,
    config: ExperimentConfig,
) -> EpisodeFailure:
    """Classify one failed MuJoCo reaching episode."""

    environment = config.environment
    policy = config.policy

    if not isinstance(
        environment,
        MujocoReachEnvironmentConfig,
    ):
        raise TypeError(
            "Expected a MuJoCo environment configuration."
        )

    if not isinstance(
        policy,
        PositionServoPolicyConfig,
    ):
        raise TypeError(
            "Expected a position-servo policy configuration."
        )

    final_error = _diagnostic_float(
        episode,
        "final_position_error",
    )

    mean_error = _diagnostic_float(
        episode,
        "mean_position_error",
    )

    peak_velocity = _diagnostic_float(
        episode,
        "peak_abs_velocity",
    )

    overshoot_count = int(
        _diagnostic_float(
            episode,
            "overshoot_count",
        )
    )

    final_velocity = abs(
        float(episode.final_velocity or 0.0)
    )

    commanded_target = _policy_float(
        episode,
        "commanded_target",
        environment.target_position,
    )

    target_bias = abs(
        commanded_target
        - environment.target_position
    )

    settled_near_command = (
        abs(
            episode.final_position
            - commanded_target
        )
        <= max(
            3.0 * environment.position_tolerance,
            0.03,
        )
    )

    severity = (
        final_error
        / max(
            environment.position_tolerance,
            1e-12,
        )
    )

    if (
        target_bias > environment.position_tolerance
        and settled_near_command
    ):
        return EpisodeFailure(
            episode=episode.episode,
            seed=episode.seed,
            failure_type="command_bias",
            confidence=0.98,
            severity=severity,
            evidence=[
                (
                    "Commanded target differs from the "
                    f"true target by {target_bias:.6f}."
                ),
                (
                    "Permitted position error is "
                    f"{environment.position_tolerance:.6f}."
                ),
                (
                    "The controller settled near its "
                    "incorrect commanded target."
                ),
            ],
        )

    if (
        policy.kp <= 5.0
        and episode.steps >= environment.max_steps
    ):
        return EpisodeFailure(
            episode=episode.episode,
            seed=episode.seed,
            failure_type="low_gain_timeout",
            confidence=0.90,
            severity=severity,
            evidence=[
                f"Controller gain was only {policy.kp:.3f}.",
                (
                    "The episode exhausted the full "
                    f"{environment.max_steps}-step budget."
                ),
                (
                    "Mean position error was "
                    f"{mean_error:.6f}."
                ),
            ],
        )

    if (
        environment.actuator_delay_steps >= 20
        and overshoot_count >= 1
    ):
        return EpisodeFailure(
            episode=episode.episode,
            seed=episode.seed,
            failure_type="delay_induced_oscillation",
            confidence=0.88,
            severity=severity,
            evidence=[
                (
                    "Actuator delay was "
                    f"{environment.actuator_delay_steps} steps."
                ),
                (
                    f"The target was crossed "
                    f"{overshoot_count} times."
                ),
                (
                    "Peak absolute velocity was "
                    f"{peak_velocity:.6f}."
                ),
            ],
        )

    if (
        environment.control_noise_std >= 0.005
        and final_velocity
        > environment.velocity_tolerance
    ):
        return EpisodeFailure(
            episode=episode.episode,
            seed=episode.seed,
            failure_type="noise_prevented_settling",
            confidence=0.85,
            severity=severity,
            evidence=[
                (
                    "Control-noise standard deviation was "
                    f"{environment.control_noise_std:.6f}."
                ),
                (
                    "Final absolute velocity was "
                    f"{final_velocity:.6f}."
                ),
                (
                    "Permitted final velocity was "
                    f"{environment.velocity_tolerance:.6f}."
                ),
            ],
        )

    if (
        environment.joint_damping >= 10.0
        and episode.steps >= environment.max_steps
    ):
        return EpisodeFailure(
            episode=episode.episode,
            seed=episode.seed,
            failure_type="overdamped_timeout",
            confidence=0.82,
            severity=severity,
            evidence=[
                (
                    "Joint damping was "
                    f"{environment.joint_damping:.3f}."
                ),
                (
                    "The episode exhausted the complete "
                    "step budget."
                ),
                (
                    "Peak absolute velocity was "
                    f"{peak_velocity:.6f}."
                ),
            ],
        )

    return EpisodeFailure(
        episode=episode.episode,
        seed=episode.seed,
        failure_type="unclassified_reach_timeout",
        confidence=0.40,
        severity=severity,
        evidence=[
            (
                "The episode ended without satisfying "
                "the success tolerances."
            ),
            (
                "Final position error was "
                f"{final_error:.6f}."
            ),
            (
                "Final absolute velocity was "
                f"{final_velocity:.6f}."
            ),
        ],
    )


def _classify_failure(
    episode: EpisodeResult,
    config: ExperimentConfig,
) -> EpisodeFailure:
    """Dispatch failure classification by environment."""

    if isinstance(
        config.environment,
        MujocoReachEnvironmentConfig,
    ):
        return _classify_mujoco_failure(
            episode,
            config,
        )

    return EpisodeFailure(
        episode=episode.episode,
        seed=episode.seed,
        failure_type="generic_episode_failure",
        confidence=0.35,
        severity=1.0,
        evidence=[
            (
                "No environment-specific classifier "
                "is available."
            )
        ],
    )


def _build_clusters(
    failures: list[EpisodeFailure],
) -> list[FailureCluster]:
    """Group classified failures by type."""

    if not failures:
        return []

    grouped: dict[
        str,
        list[EpisodeFailure],
    ] = {}

    for failure in failures:
        grouped.setdefault(
            failure.failure_type,
            [],
        ).append(failure)

    clusters: list[FailureCluster] = []

    for failure_type, members in grouped.items():
        count = len(members)

        clusters.append(
            FailureCluster(
                failure_type=failure_type,
                count=count,
                fraction_of_failures=(
                    count / len(failures)
                ),
                mean_confidence=(
                    sum(
                        member.confidence
                        for member in members
                    )
                    / count
                ),
                mean_severity=(
                    sum(
                        member.severity
                        for member in members
                    )
                    / count
                ),
                episodes=sorted(
                    member.episode
                    for member in members
                ),
            )
        )

    return sorted(
        clusters,
        key=lambda cluster: (
            -cluster.count,
            cluster.failure_type,
        ),
    )


def _write_cluster_csv(
    path: Path,
    clusters: list[FailureCluster],
) -> None:
    """Write the clustered failures as CSV."""

    fieldnames = [
        "failure_type",
        "count",
        "fraction_of_failures",
        "mean_confidence",
        "mean_severity",
        "episodes",
    ]

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for cluster in clusters:
            writer.writerow(
                {
                    "failure_type": cluster.failure_type,
                    "count": cluster.count,
                    "fraction_of_failures": (
                        f"{cluster.fraction_of_failures:.10f}"
                    ),
                    "mean_confidence": (
                        f"{cluster.mean_confidence:.10f}"
                    ),
                    "mean_severity": (
                        f"{cluster.mean_severity:.10f}"
                    ),
                    "episodes": ";".join(
                        str(episode)
                        for episode in cluster.episodes
                    ),
                }
            )


def analyze_run_failures(
    run_dir: Path,
) -> FailureAnalysisRun:
    """Classify and cluster failures from one run."""

    manifest = _load_json_object(
        run_dir / "manifest.json"
    )

    config_payload = manifest.get("config")

    if not isinstance(config_payload, dict):
        raise FailureAnalysisError(
            "Run manifest does not contain "
            "a valid configuration."
        )

    try:
        config = ExperimentConfig.model_validate(
            config_payload
        )
    except ValidationError as exc:
        raise FailureAnalysisError(
            f"Run contains an invalid configuration:\n{exc}"
        ) from exc

    episodes = _load_episodes(
        run_dir / "episodes.jsonl"
    )

    failed_records = [
        episode
        for episode in episodes
        if not episode.success
    ]

    failures = [
        _classify_failure(
            episode,
            config,
        )
        for episode in failed_records
    ]

    clusters = _build_clusters(failures)

    failed_count = len(failures)
    total_count = len(episodes)

    summary = FailureAnalysisSummary(
        run_id=str(
            manifest.get(
                "run_id",
                run_dir.name,
            )
        ),
        total_episodes=total_count,
        failed_episodes=failed_count,
        failure_rate=(
            failed_count / total_count
            if total_count > 0
            else 0.0
        ),
        dominant_failure_type=(
            clusters[0].failure_type
            if clusters
            else None
        ),
        clusters=clusters,
        failures=failures,
    )

    analysis_path = (
        run_dir / "failure_analysis.json"
    )

    analysis_path.write_text(
        json.dumps(
            summary.model_dump(mode="json"),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    clusters_path = (
        run_dir / "failure_clusters.csv"
    )

    _write_cluster_csv(
        clusters_path,
        clusters,
    )

    return FailureAnalysisRun(
        analysis_path=analysis_path,
        clusters_path=clusters_path,
        summary=summary,
    )
