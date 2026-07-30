"""Regression comparison between baseline and candidate runs."""

import json
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from forge_ci.config import ExperimentConfig
from forge_ci.models import ComparisonResult, RunSummary


class ComparisonError(ValueError):
    """Raised when two run artifacts cannot be compared safely."""


@dataclass(frozen=True)
class LoadedRun:
    """Validated information loaded from one run directory."""

    run_id: str
    summary: RunSummary
    config: ExperimentConfig


def _load_json_object(path: Path) -> dict[str, Any]:
    """Load a JSON file and require an object at its root."""

    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ComparisonError(f"Required artifact not found: {path}") from None
    except (OSError, JSONDecodeError) as exc:
        raise ComparisonError(f"Could not read {path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ComparisonError(f"Artifact must contain a JSON object: {path}")

    return payload


def _load_run(run_dir: Path) -> LoadedRun:
    """Load and validate one completed FORGE-CI run."""

    manifest_payload = _load_json_object(run_dir / "manifest.json")
    summary_payload = _load_json_object(run_dir / "summary.json")

    config_payload = manifest_payload.get("config")

    if not isinstance(config_payload, dict):
        raise ComparisonError(
            f"Manifest does not contain a valid config object: {run_dir}"
        )

    try:
        config = ExperimentConfig.model_validate(config_payload)
        summary = RunSummary.model_validate(summary_payload)
    except ValidationError as exc:
        raise ComparisonError(
            f"Run contains invalid structured artifacts: {run_dir}\n{exc}"
        ) from exc

    run_id = str(manifest_payload.get("run_id", run_dir.name))

    return LoadedRun(
        run_id=run_id,
        summary=summary,
        config=config,
    )


def _validate_compatibility(
    baseline: LoadedRun,
    candidate: LoadedRun,
) -> None:
    """Ensure the two experiments are suitable for direct comparison."""

    if baseline.config.seed != candidate.config.seed:
        raise ComparisonError(
            "Baseline and candidate must use the same root seed."
        )

    if baseline.config.episodes != candidate.config.episodes:
        raise ComparisonError(
            "Baseline and candidate must use the same episode count."
        )

    if baseline.config.environment != candidate.config.environment:
        raise ComparisonError(
            "Baseline and candidate must use identical environments."
        )


def compare_runs(
    baseline_dir: Path,
    candidate_dir: Path,
) -> ComparisonResult:
    """Compare a candidate against a known-good baseline run."""

    baseline = _load_run(baseline_dir)
    candidate = _load_run(candidate_dir)

    _validate_compatibility(baseline, candidate)

    success_rate_delta = (
        candidate.summary.success_rate
        - baseline.summary.success_rate
    )

    mean_steps_delta = (
        candidate.summary.mean_steps
        - baseline.summary.mean_steps
    )

    success_rate_drop = max(0.0, -success_rate_delta)
    mean_steps_increase = max(0.0, mean_steps_delta)

    max_success_rate_drop = (
        candidate.config.gate.max_success_rate_drop
    )

    max_mean_steps_increase = (
        candidate.config.gate.max_mean_steps_increase
    )

    reasons: list[str] = []

    if success_rate_drop > max_success_rate_drop + 1e-12:
        reasons.append(
            "Success-rate drop "
            f"{success_rate_drop:.1%} exceeds allowed "
            f"{max_success_rate_drop:.1%}."
        )

    if mean_steps_increase > max_mean_steps_increase + 1e-12:
        reasons.append(
            "Mean-step increase "
            f"{mean_steps_increase:.3f} exceeds allowed "
            f"{max_mean_steps_increase:.3f}."
        )

    result = ComparisonResult(
        baseline_run_id=baseline.run_id,
        candidate_run_id=candidate.run_id,
        baseline_success_rate=baseline.summary.success_rate,
        candidate_success_rate=candidate.summary.success_rate,
        success_rate_delta=round(success_rate_delta, 10),
        baseline_mean_steps=baseline.summary.mean_steps,
        candidate_mean_steps=candidate.summary.mean_steps,
        mean_steps_delta=round(mean_steps_delta, 10),
        max_success_rate_drop=max_success_rate_drop,
        max_mean_steps_increase=max_mean_steps_increase,
        gate_passed=not reasons,
        reasons=reasons,
    )

    comparison_path = candidate_dir / "comparison.json"

    comparison_path.write_text(
        json.dumps(
            result.model_dump(mode="json"),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return result
