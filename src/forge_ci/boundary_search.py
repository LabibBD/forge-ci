"""Automatic discovery of minimal failing parameter values."""

import csv
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import yaml

from forge_ci.boundary_config import (
    BoundarySearchConfig,
)
from forge_ci.boundary_models import (
    BoundarySearchSummary,
    BoundaryTrial,
)
from forge_ci.config import (
    ExperimentConfig,
    PositionServoPolicyConfig,
)
from forge_ci.failure_analysis import (
    analyze_run_failures,
)
from forge_ci.runner import (
    EvaluationRun,
    run_evaluation,
)


class BoundarySearchError(ValueError):
    """Raised when a valid failure boundary cannot be searched."""


@dataclass(frozen=True)
class BoundarySearchRun:
    """Artifacts returned after a boundary search."""

    search_dir: Path
    summary: BoundarySearchSummary


def _write_json(
    path: Path,
    payload: object,
) -> None:
    """Write indented JSON with a trailing newline."""

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _config_digest(
    config: BoundarySearchConfig,
) -> str:
    """Calculate a stable digest of the search configuration."""

    canonical = json.dumps(
        config.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    )

    return sha256(
        canonical.encode("utf-8")
    ).hexdigest()


def _create_experiment(
    config: BoundarySearchConfig,
    *,
    value: float,
    label: str,
) -> ExperimentConfig:
    """Create one experiment at a selected parameter value."""

    payload = config.base_experiment.model_dump(
        mode="python"
    )

    policy_payload = payload["policy"]

    if not isinstance(policy_payload, dict):
        raise BoundarySearchError(
            "Base experiment has an invalid policy object."
        )

    signed_value = (
        value
        if config.direction == "positive"
        else -value
    )

    policy_payload["target_bias"] = signed_value

    payload["name"] = (
        f"{config.name}:{label}:{signed_value:.10f}"
    )

    return ExperimentConfig.model_validate(payload)


def _run_trial(
    config: BoundarySearchConfig,
    *,
    search_dir: Path,
    index: int,
    label: str,
    value: float,
) -> tuple[BoundaryTrial, EvaluationRun]:
    """Evaluate and record one point in the search interval."""

    experiment = _create_experiment(
        config,
        value=value,
        label=label,
    )

    trial_root = (
        search_dir
        / "trials"
        / f"{index:03d}-{label}"
    )

    evaluation = run_evaluation(
        experiment,
        output_root=trial_root,
    )

    trial = BoundaryTrial(
        index=index,
        label=label,
        parameter_value=value,
        success_rate=(
            evaluation.summary.success_rate
        ),
        mean_steps=evaluation.summary.mean_steps,
        gate_passed=(
            evaluation.summary.gate_passed
        ),
        run_directory=str(
            evaluation.run_dir.relative_to(
                search_dir
            )
        ),
    )

    return trial, evaluation


def _write_trials_csv(
    path: Path,
    trials: list[BoundaryTrial],
) -> None:
    """Write all evaluated boundary-search trials."""

    fieldnames = [
        "index",
        "label",
        "parameter_value",
        "success_rate",
        "mean_steps",
        "gate_passed",
        "run_directory",
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

        for trial in trials:
            writer.writerow(
                {
                    "index": trial.index,
                    "label": trial.label,
                    "parameter_value": (
                        f"{trial.parameter_value:.10f}"
                    ),
                    "success_rate": (
                        f"{trial.success_rate:.10f}"
                    ),
                    "mean_steps": (
                        f"{trial.mean_steps:.10f}"
                    ),
                    "gate_passed": trial.gate_passed,
                    "run_directory": (
                        trial.run_directory
                    ),
                }
            )


def run_boundary_search(
    config: BoundarySearchConfig,
    output_root: Path = Path("runs/boundaries"),
) -> BoundarySearchRun:
    """Find the smallest known parameter value that fails."""

    created_at = datetime.now(UTC)
    digest = _config_digest(config)

    search_id = (
        f"{created_at.strftime('%Y%m%dT%H%M%S%fZ')}-"
        f"{digest[:8]}"
    )

    search_dir = output_root / search_id

    search_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    trials: list[BoundaryTrial] = []

    trial_index = 0

    low_value = config.low
    high_value = config.high

    low_trial, _ = _run_trial(
        config,
        search_dir=search_dir,
        index=trial_index,
        label="lower-bound",
        value=low_value,
    )

    trials.append(low_trial)
    trial_index += 1

    if not low_trial.gate_passed:
        raise BoundarySearchError(
            "The lower search bound already fails. "
            "Choose a lower value that passes."
        )

    high_trial, high_evaluation = _run_trial(
        config,
        search_dir=search_dir,
        index=trial_index,
        label="upper-bound",
        value=high_value,
    )

    trials.append(high_trial)
    trial_index += 1

    if high_trial.gate_passed:
        raise BoundarySearchError(
            "The upper search bound still passes. "
            "Choose a higher value that fails."
        )

    failing_evaluation = high_evaluation
    search_iterations = 0

    while (
        high_value - low_value > config.tolerance
        and search_iterations < config.max_iterations
    ):
        midpoint = (
            low_value + high_value
        ) / 2.0

        midpoint_trial, midpoint_evaluation = (
            _run_trial(
                config,
                search_dir=search_dir,
                index=trial_index,
                label=f"iteration-{search_iterations + 1}",
                value=midpoint,
            )
        )

        trials.append(midpoint_trial)
        trial_index += 1
        search_iterations += 1

        if midpoint_trial.gate_passed:
            low_value = midpoint
        else:
            high_value = midpoint
            failing_evaluation = midpoint_evaluation

    boundary_width = (
        high_value - low_value
    )

    analysis = analyze_run_failures(
        failing_evaluation.run_dir
    )

    counterexample_payload = (
        config.base_experiment.model_dump(
            mode="json"
        )
    )

    counterexample_policy = (
        counterexample_payload["policy"]
    )

    if not isinstance(
        counterexample_policy,
        dict,
    ):
        raise BoundarySearchError(
            "Could not construct counterexample policy."
        )

    counterexample_payload["name"] = (
        f"{config.name}:minimal-counterexample"
    )

    counterexample_policy["target_bias"] = (
        high_value
        if config.direction == "positive"
        else -high_value
    )

    counterexample_path = (
        search_dir / "counterexample.yaml"
    )

    counterexample_path.write_text(
        yaml.safe_dump(
            counterexample_payload,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    summary = BoundarySearchSummary(
        search_name=config.name,
        parameter=config.parameter,
        largest_passing_value=low_value,
        smallest_failing_value=high_value,
        boundary_width=boundary_width,
        converged=(
            boundary_width <= config.tolerance
        ),
        search_iterations=search_iterations,
        counterexample_run_directory=str(
            failing_evaluation.run_dir.relative_to(
                search_dir
            )
        ),
        counterexample_config=(
            counterexample_path.name
        ),
        dominant_failure_type=(
            analysis.summary.dominant_failure_type
        ),
        trials=trials,
    )

    manifest = {
        "schema_version": 1,
        "search_id": search_id,
        "created_at_utc": created_at.isoformat(),
        "config_sha256": digest,
        "config": config.model_dump(mode="json"),
    }

    _write_json(
        search_dir / "boundary_manifest.json",
        manifest,
    )

    _write_json(
        search_dir / "boundary_summary.json",
        summary.model_dump(mode="json"),
    )

    _write_trials_csv(
        search_dir / "boundary_trials.csv",
        trials,
    )

    return BoundarySearchRun(
        search_dir=search_dir,
        summary=summary,
    )
