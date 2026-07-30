"""Experiment configuration models."""

from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import (
    BaseModel,
    Field,
    PositiveInt,
    model_validator,
)


class LineWorldEnvironmentConfig(BaseModel):
    """Configuration for the deterministic LineWorld environment."""

    name: Literal["line_world"] = "line_world"
    goal: PositiveInt = 5
    max_steps: PositiveInt = 10
    slip_probability: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )


class MujocoReachEnvironmentConfig(BaseModel):
    """Configuration for the MuJoCo slider-reaching environment."""

    name: Literal["mujoco_reach"] = "mujoco_reach"

    target_position: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
    )

    max_steps: PositiveInt = 1000

    position_tolerance: float = Field(
        default=0.01,
        gt=0.0,
    )

    velocity_tolerance: float = Field(
        default=0.02,
        ge=0.0,
    )

    initial_position_low: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    initial_position_high: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
    )

    @model_validator(mode="after")
    def validate_initial_position_range(
        self,
    ) -> "MujocoReachEnvironmentConfig":
        """Require a valid initial-position interval."""

        if self.initial_position_low > self.initial_position_high:
            raise ValueError(
                "initial_position_low cannot exceed "
                "initial_position_high."
            )

        return self


EnvironmentConfig = Annotated[
    LineWorldEnvironmentConfig
    | MujocoReachEnvironmentConfig,
    Field(discriminator="name"),
]


class GreedyPolicyConfig(BaseModel):
    """Configuration for the greedy LineWorld policy."""

    name: Literal["greedy"] = "greedy"


class AlternatingPolicyConfig(BaseModel):
    """Configuration for the slower LineWorld policy."""

    name: Literal["alternating"] = "alternating"


class PositionServoPolicyConfig(BaseModel):
    """Configuration for the MuJoCo position-servo policy."""

    name: Literal["position_servo"] = "position_servo"


PolicyConfig = Annotated[
    GreedyPolicyConfig
    | AlternatingPolicyConfig
    | PositionServoPolicyConfig,
    Field(discriminator="name"),
]


class GateConfig(BaseModel):
    """Thresholds that decide whether an evaluation passes CI."""

    min_success_rate: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
    )

    max_success_rate_drop: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    max_mean_steps_increase: float = Field(
        default=0.0,
        ge=0.0,
    )


class ExperimentConfig(BaseModel):
    """Validated configuration for one evaluation experiment."""

    name: str = Field(min_length=1)
    seed: int = 42
    episodes: PositiveInt = 10

    environment: EnvironmentConfig = Field(
        default_factory=LineWorldEnvironmentConfig
    )

    policy: PolicyConfig = Field(
        default_factory=GreedyPolicyConfig
    )

    gate: GateConfig = Field(
        default_factory=GateConfig
    )

    @model_validator(mode="after")
    def validate_environment_policy_pair(
        self,
    ) -> "ExperimentConfig":
        """Reject policies that cannot control the selected environment."""

        if isinstance(
            self.environment,
            LineWorldEnvironmentConfig,
        ):
            if not isinstance(
                self.policy,
                GreedyPolicyConfig | AlternatingPolicyConfig,
            ):
                raise ValueError(
                    "line_world requires the greedy or "
                    "alternating policy."
                )

        if isinstance(
            self.environment,
            MujocoReachEnvironmentConfig,
        ):
            if not isinstance(
                self.policy,
                PositionServoPolicyConfig,
            ):
                raise ValueError(
                    "mujoco_reach requires the "
                    "position_servo policy."
                )

        return self


def load_config(path: Path) -> ExperimentConfig:
    """Load and validate an experiment configuration from YAML."""

    raw_data: Any = yaml.safe_load(
        path.read_text(encoding="utf-8")
    )

    if raw_data is None:
        raw_data = {}

    if not isinstance(raw_data, dict):
        raise ValueError(
            "Configuration root must be a YAML mapping."
        )

    return ExperimentConfig.model_validate(raw_data)
