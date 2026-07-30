"""Tests for the permanent multidimensional robustness envelope."""

import json
from pathlib import Path

from forge_ci.envelope_config import (
    load_envelope_config,
)
from forge_ci.envelope_search import (
    run_robustness_envelope,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = PROJECT_ROOT / "configs" / "mujoco_multidimensional_envelope.yaml"

EXPECTED_DIMENSIONS = {
    "positive-command-bias",
    "negative-command-bias",
    "controller-gain-loss",
    "actuator-delay",
    "control-noise",
    "joint-damping",
}


def test_multidimensional_envelope_configuration() -> None:
    """The permanent envelope should cover six dimensions."""

    config = load_envelope_config(CONFIG_PATH)

    names = {dimension.name for dimension in config.dimensions}

    parameters = {dimension.parameter for dimension in config.dimensions}

    assert names == EXPECTED_DIMENSIONS

    assert parameters == {
        "target_bias",
        "kp",
        "actuator_delay_steps",
        "control_noise_std",
        "joint_damping",
    }

    assert len(config.dimensions) == 6


def test_multidimensional_envelope_discovers_boundaries(
    tmp_path: Path,
) -> None:
    """Every configured robustness dimension should be searchable."""

    config = load_envelope_config(CONFIG_PATH)

    envelope = run_robustness_envelope(
        config,
        tmp_path / "envelopes",
    )

    summary = envelope.summary

    assert summary.all_converged is True
    assert len(summary.dimensions) == 6

    names = {dimension.name for dimension in summary.dimensions}

    assert names == EXPECTED_DIMENSIONS

    normalized_positions = [
        dimension.normalized_boundary_position for dimension in summary.dimensions
    ]

    assert normalized_positions == sorted(normalized_positions)

    assert summary.weakest_dimension == summary.dimensions[0].name

    for dimension in summary.dimensions:
        assert dimension.converged is True

        assert dimension.largest_passing_magnitude < dimension.smallest_failing_magnitude

        assert 0.0 < dimension.normalized_boundary_position <= 1.0

        assert dimension.dominant_failure_type is not None

        search_dir = envelope.envelope_dir / dimension.search_directory

        assert (search_dir / "boundary_summary.json").exists()

        assert (search_dir / "counterexample.yaml").exists()

        boundary_summary = json.loads(
            (search_dir / "boundary_summary.json").read_text(encoding="utf-8")
        )

        counterexample_run = search_dir / boundary_summary["counterexample_run_directory"]

        assert (counterexample_run / "failure_analysis.json").exists()

    assert (envelope.envelope_dir / "envelope_summary.json").exists()

    assert (envelope.envelope_dir / "robustness_envelope.csv").exists()
