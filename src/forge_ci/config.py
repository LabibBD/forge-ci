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

    actuator_delay_steps: int = Field(
        default=0,
        ge=0,
        le=200,
    )

    control_noise_std: float = Field(
        default=0.0,
        ge=0.0,
        le=0.5,
    )

    joint_damping: float = Field(
        default=4.0,
        gt=0.0,
        le=100.0,
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


class PandaReachEnvironmentConfig(BaseModel):
    """Pinned external Panda model and task settings."""

    name: Literal["panda_reach"] = "panda_reach"
    model_path: str = "mujoco_menagerie/franka_emika_panda/scene.xml"
    model_revision: str = Field(
        default="eea4c7431b63da8278b0ff31638e0b0e941cec0e",
        pattern=r"^[0-9a-f]{40}$",
    )
    duration_s: float = Field(default=6.0, ge=2.0, le=30.0)
    # Legacy runs without this field used the full observation horizon.
    step_metric: Literal["horizon", "settling_or_horizon"] = "horizon"
    position_tolerance_m: float = Field(default=0.01, gt=0.0)
    speed_tolerance_m_s: float = Field(default=0.02, ge=0.0)


EnvironmentConfig = Annotated[
    LineWorldEnvironmentConfig
    | MujocoReachEnvironmentConfig
    | PandaReachEnvironmentConfig,
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

    kp: float = Field(
        default=40.0,
        gt=0.0,
        le=200.0,
    )

    target_bias: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
    )


class PandaReachPolicyConfig(BaseModel):
    """Position-only IK with an optional degraded command channel."""

    name: Literal["panda_reach_ik"] = "panda_reach_ik"
    cartesian_feedback_gain: float = Field(default=0.0, ge=0.0, le=0.3)
    command_bias_rad: float = Field(default=0.0, ge=-0.5, le=0.5)
    command_delay_steps: int = Field(default=0, ge=0, le=2000)
    command_noise_std_rad: float = Field(default=0.0, ge=0.0, le=0.2)


PolicyConfig = Annotated[
    GreedyPolicyConfig
    | AlternatingPolicyConfig
    | PositionServoPolicyConfig
    | PandaReachPolicyConfig,
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

        if isinstance(self.environment, PandaReachEnvironmentConfig):
            if not isinstance(self.policy, PandaReachPolicyConfig):
                raise ValueError("panda_reach requires the panda_reach_ik policy.")

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
