"""MuJoCo robustness-sweep execution and reporting."""

import csv
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

from forge_ci.config import (
    ExperimentConfig,
    MujocoReachEnvironmentConfig,
)
from forge_ci.runner import run_evaluation
from forge_ci.sweep_config import RobustnessSweepConfig
from forge_ci.sweep_models import (
    RobustnessSweepSummary,
    ScenarioSweepResult,
)


@dataclass(frozen=True)
class RobustnessSweepRun:
    """Artifacts returned after a completed sweep."""

    sweep_dir: Path
    summary: RobustnessSweepSummary


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
    config: RobustnessSweepConfig,
) -> str:
    """Calculate a stable digest of a sweep configuration."""

    canonical = json.dumps(
        config.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    )

    return sha256(
        canonical.encode("utf-8")
    ).hexdigest()


def _create_scenario_experiment(
    config: RobustnessSweepConfig,
    scenario_index: int,
) -> ExperimentConfig:
    """Create one validated experiment from a scenario."""

    scenario = config.scenarios[scenario_index]
    base = config.base_experiment

    environment = base.environment

    if not isinstance(
        environment,
        MujocoReachEnvironmentConfig,
    ):
        raise TypeError(
            "Expected a MuJoCo reach environment."
        )

    environment_payload = environment.model_dump(
        mode="python"
    )

    environment_payload.update(
        {
            "actuator_delay_steps": (
                scenario.actuator_delay_steps
            ),
            "control_noise_std": (
                scenario.control_noise_std
            ),
            "joint_damping": scenario.joint_damping,
        }
    )

    experiment_payload = base.model_dump(
        mode="python"
    )

    experiment_payload["name"] = (
        f"{config.name}:{scenario.name}"
    )

    experiment_payload["environment"] = (
        environment_payload
    )

    return ExperimentConfig.model_validate(
        experiment_payload
    )


def _write_matrix(
    path: Path,
    results: list[ScenarioSweepResult],
) -> None:
    """Write a CSV robustness matrix."""

    fieldnames = [
        "scenario",
        "actuator_delay_steps",
        "control_noise_std",
        "joint_damping",
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

        for result in results:
            writer.writerow(
                {
                    "scenario": result.scenario_name,
                    "actuator_delay_steps": (
                        result.actuator_delay_steps
                    ),
                    "control_noise_std": (
                        result.control_noise_std
                    ),
                    "joint_damping": (
                        result.joint_damping
                    ),
                    "success_rate": (
                        f"{result.success_rate:.10f}"
                    ),
                    "mean_steps": (
                        f"{result.mean_steps:.10f}"
                    ),
                    "gate_passed": result.gate_passed,
                    "run_directory": (
                        result.run_directory
                    ),
                }
            )


def run_robustness_sweep(
    config: RobustnessSweepConfig,
    output_root: Path = Path("runs/sweeps"),
) -> RobustnessSweepRun:
    """Evaluate one policy across all disturbance scenarios."""

    created_at = datetime.now(UTC)
    digest = _config_digest(config)

    sweep_id = (
        f"{created_at.strftime('%Y%m%dT%H%M%S%fZ')}-"
        f"{digest[:8]}"
    )

    sweep_dir = output_root / sweep_id

    sweep_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    scenario_results: list[
        ScenarioSweepResult
    ] = []

    for index, scenario in enumerate(config.scenarios):
        experiment = _create_scenario_experiment(
            config,
            index,
        )

        evaluation = run_evaluation(
            experiment,
            output_root=(
                sweep_dir
                / "scenarios"
                / scenario.name
            ),
        )

        success_gate_passed = (
            evaluation.summary.success_rate
            >= config.gate.min_scenario_success_rate
        )

        steps_gate_passed = (
            evaluation.summary.mean_steps
            <= config.gate.max_scenario_mean_steps
        )

        scenario_result = ScenarioSweepResult(
            scenario_name=scenario.name,
            run_directory=str(
                evaluation.run_dir.relative_to(
                    sweep_dir
                )
            ),
            actuator_delay_steps=(
                scenario.actuator_delay_steps
            ),
            control_noise_std=(
                scenario.control_noise_std
            ),
            joint_damping=scenario.joint_damping,
            success_rate=(
                evaluation.summary.success_rate
            ),
            mean_steps=(
                evaluation.summary.mean_steps
            ),
            gate_passed=(
                success_gate_passed
                and steps_gate_passed
            ),
        )

        scenario_results.append(scenario_result)

    worst_success_rate = min(
        result.success_rate
        for result in scenario_results
    )

    worst_mean_steps = max(
        result.mean_steps
        for result in scenario_results
    )

    reasons: list[str] = []

    for result in scenario_results:
        if (
            result.success_rate
            < config.gate.min_scenario_success_rate
        ):
            reasons.append(
                f"Scenario '{result.scenario_name}' "
                f"success rate {result.success_rate:.1%} "
                "is below required "
                f"{config.gate.min_scenario_success_rate:.1%}."
            )

        if (
            result.mean_steps
            > config.gate.max_scenario_mean_steps
        ):
            reasons.append(
                f"Scenario '{result.scenario_name}' "
                f"mean steps {result.mean_steps:.3f} "
                "exceeds allowed "
                f"{config.gate.max_scenario_mean_steps:.3f}."
            )

    summary = RobustnessSweepSummary(
        sweep_name=config.name,
        scenario_count=len(scenario_results),
        worst_success_rate=worst_success_rate,
        worst_mean_steps=worst_mean_steps,
        min_scenario_success_rate=(
            config.gate.min_scenario_success_rate
        ),
        max_scenario_mean_steps=(
            config.gate.max_scenario_mean_steps
        ),
        gate_passed=not reasons,
        reasons=reasons,
        scenarios=scenario_results,
    )

    manifest = {
        "schema_version": 1,
        "sweep_id": sweep_id,
        "created_at_utc": created_at.isoformat(),
        "config_sha256": digest,
        "config": config.model_dump(mode="json"),
    }

    _write_json(
        sweep_dir / "sweep_manifest.json",
        manifest,
    )

    _write_json(
        sweep_dir / "sweep_summary.json",
        summary.model_dump(mode="json"),
    )

    _write_matrix(
        sweep_dir / "robustness_matrix.csv",
        scenario_results,
    )

    return RobustnessSweepRun(
        sweep_dir=sweep_dir,
        summary=summary,
    )
