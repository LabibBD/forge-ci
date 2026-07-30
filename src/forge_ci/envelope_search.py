"""Multi-dimensional robustness-envelope discovery."""

import csv
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

from forge_ci.boundary_search import (
    BoundarySearchError,
    run_boundary_search,
)
from forge_ci.envelope_config import (
    RobustnessEnvelopeConfig,
)
from forge_ci.envelope_models import (
    EnvelopeDimensionResult,
    RobustnessEnvelopeSummary,
)


class EnvelopeSearchError(ValueError):
    """Raised when an envelope dimension cannot be searched."""


@dataclass(frozen=True)
class RobustnessEnvelopeRun:
    """Artifacts produced by a robustness-envelope search."""

    envelope_dir: Path
    summary: RobustnessEnvelopeSummary


def _write_json(
    path: Path,
    payload: object,
) -> None:
    """Write stable, indented JSON."""

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
    config: RobustnessEnvelopeConfig,
) -> str:
    """Calculate a stable configuration digest."""

    canonical = json.dumps(
        config.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    )

    return sha256(
        canonical.encode("utf-8")
    ).hexdigest()


def _slug(value: str) -> str:
    """Create a filesystem-safe dimension name."""

    slug = re.sub(
        r"[^a-zA-Z0-9]+",
        "-",
        value,
    ).strip("-").lower()

    return slug or "dimension"


def _normalized_boundary_position(
    *,
    low: float,
    high: float,
    failing: float,
) -> float:
    """Locate the failure boundary within its configured range."""

    position = (
        failing - low
    ) / (
        high - low
    )

    return min(
        1.0,
        max(
            0.0,
            position,
        ),
    )


def _write_envelope_csv(
    path: Path,
    dimensions: list[EnvelopeDimensionResult],
) -> None:
    """Write the ranked robustness envelope as CSV."""

    fieldnames = [
        "rank",
        "name",
        "parameter",
        "direction",
        "baseline_parameter_value",
        "largest_passing_magnitude",
        "smallest_failing_magnitude",
        "passing_applied_value",
        "failing_applied_value",
        "boundary_width",
        "normalized_boundary_position",
        "converged",
        "dominant_failure_type",
        "search_directory",
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

        for rank, dimension in enumerate(
            dimensions,
            start=1,
        ):
            writer.writerow(
                {
                    "rank": rank,
                    "name": dimension.name,
                    "parameter": dimension.parameter,
                    "direction": dimension.direction,
                    "baseline_parameter_value": (
                        dimension.baseline_parameter_value
                    ),
                    "largest_passing_magnitude": (
                        f"{dimension.largest_passing_magnitude:.10f}"
                    ),
                    "smallest_failing_magnitude": (
                        f"{dimension.smallest_failing_magnitude:.10f}"
                    ),
                    "passing_applied_value": (
                        dimension.passing_applied_value
                    ),
                    "failing_applied_value": (
                        dimension.failing_applied_value
                    ),
                    "boundary_width": (
                        f"{dimension.boundary_width:.10f}"
                    ),
                    "normalized_boundary_position": (
                        f"{dimension.normalized_boundary_position:.10f}"
                    ),
                    "converged": dimension.converged,
                    "dominant_failure_type": (
                        dimension.dominant_failure_type
                    ),
                    "search_directory": (
                        dimension.search_directory
                    ),
                }
            )


def run_robustness_envelope(
    config: RobustnessEnvelopeConfig,
    output_root: Path = Path("runs/envelopes"),
) -> RobustnessEnvelopeRun:
    """Discover and rank every configured failure boundary."""

    created_at = datetime.now(UTC)
    digest = _config_digest(config)

    envelope_id = (
        f"{created_at.strftime('%Y%m%dT%H%M%S%fZ')}-"
        f"{digest[:8]}"
    )

    envelope_dir = output_root / envelope_id

    envelope_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    results: list[EnvelopeDimensionResult] = []

    for index, dimension in enumerate(
        config.dimensions,
    ):
        boundary_config = (
            dimension.to_boundary_config(
                envelope_name=config.name,
                base_experiment=config.base_experiment,
            )
        )

        dimension_root = (
            envelope_dir
            / "dimensions"
            / f"{index:02d}-{_slug(dimension.name)}"
        )

        try:
            boundary_run = run_boundary_search(
                boundary_config,
                output_root=dimension_root,
            )
        except BoundarySearchError as exc:
            raise EnvelopeSearchError(
                f"Envelope dimension "
                f"'{dimension.name}' failed: {exc}"
            ) from exc

        boundary = boundary_run.summary

        results.append(
            EnvelopeDimensionResult(
                name=dimension.name,
                parameter=boundary.parameter,
                direction=boundary.direction,
                baseline_parameter_value=(
                    boundary.baseline_parameter_value
                ),
                largest_passing_magnitude=(
                    boundary.largest_passing_value
                ),
                smallest_failing_magnitude=(
                    boundary.smallest_failing_value
                ),
                passing_applied_value=(
                    boundary.passing_applied_value
                ),
                failing_applied_value=(
                    boundary.failing_applied_value
                ),
                boundary_width=boundary.boundary_width,
                normalized_boundary_position=(
                    _normalized_boundary_position(
                        low=dimension.low,
                        high=dimension.high,
                        failing=(
                            boundary.smallest_failing_value
                        ),
                    )
                ),
                converged=boundary.converged,
                dominant_failure_type=(
                    boundary.dominant_failure_type
                ),
                search_directory=str(
                    boundary_run.search_dir.relative_to(
                        envelope_dir
                    )
                ),
            )
        )

    ranked_results = sorted(
        results,
        key=lambda result: (
            result.normalized_boundary_position,
            result.name,
        ),
    )

    summary = RobustnessEnvelopeSummary(
        envelope_name=config.name,
        all_converged=all(
            result.converged
            for result in ranked_results
        ),
        weakest_dimension=ranked_results[0].name,
        dimensions=ranked_results,
    )

    manifest = {
        "schema_version": 1,
        "envelope_id": envelope_id,
        "created_at_utc": created_at.isoformat(),
        "config_sha256": digest,
        "config": config.model_dump(mode="json"),
    }

    _write_json(
        envelope_dir / "envelope_manifest.json",
        manifest,
    )

    _write_json(
        envelope_dir / "envelope_summary.json",
        summary.model_dump(mode="json"),
    )

    _write_envelope_csv(
        envelope_dir / "robustness_envelope.csv",
        ranked_results,
    )

    return RobustnessEnvelopeRun(
        envelope_dir=envelope_dir,
        summary=summary,
    )
