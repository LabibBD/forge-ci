"""Tests for experiment configuration loading."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from forge_ci.config import load_config


def test_load_valid_config(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        "name: test-experiment\nseed: 42\nepisodes: 5\n",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.name == "test-experiment"
    assert config.seed == 42
    assert config.episodes == 5


def test_reject_zero_episodes(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(
        "name: invalid-experiment\nepisodes: 0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_config(config_path)
