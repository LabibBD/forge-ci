"""Experiment configuration models."""

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, PositiveInt


class EnvironmentConfig(BaseModel):
    """Configuration for the first deterministic test environment."""

    name: Literal["line_world"] = "line_world"
    goal: PositiveInt = 5
    max_steps: PositiveInt = 10
    slip_probability: float = Field(default=0.0, ge=0.0, le=1.0)


class PolicyConfig(BaseModel):
    """Configuration for the policy under evaluation."""

    name: Literal["greedy"] = "greedy"


class GateConfig(BaseModel):
    """Thresholds that decide whether an evaluation passes CI."""

    min_success_rate: float = Field(default=1.0, ge=0.0, le=1.0)


class ExperimentConfig(BaseModel):
    """Validated configuration for one evaluation experiment."""

    name: str = Field(min_length=1)
    seed: int = 42
    episodes: PositiveInt = 10
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    gate: GateConfig = Field(default_factory=GateConfig)


def load_config(path: Path) -> ExperimentConfig:
    """Load and validate an experiment configuration from YAML."""

    raw_data: Any = yaml.safe_load(path.read_text(encoding="utf-8"))

    if raw_data is None:
        raw_data = {}

    if not isinstance(raw_data, dict):
        raise ValueError("Configuration root must be a YAML mapping.")

    return ExperimentConfig.model_validate(raw_data)
