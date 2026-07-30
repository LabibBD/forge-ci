"""Configuration models for failure-boundary discovery."""

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import (
    BaseModel,
    Field,
    PositiveInt,
    model_validator,
)

from forge_ci.config import (
    ExperimentConfig,
    MujocoReachEnvironmentConfig,
    PositionServoPolicyConfig,
)

BoundaryParameter = Literal[
    "target_bias",
    "kp",
    "actuator_delay_steps",
    "control_noise_std",
    "joint_damping",
]

BoundaryDirection = Literal[
    "positive",
    "negative",
]

POLICY_PARAMETERS = frozenset(
    {
        "target_bias",
        "kp",
    }
)

ENVIRONMENT_PARAMETERS = frozenset(
    {
        "actuator_delay_steps",
        "control_noise_std",
        "joint_damping",
    }
)

INTEGER_PARAMETERS = frozenset(
    {
        "actuator_delay_steps",
    }
)


class BoundarySearchConfig(BaseModel):
    """Configuration for deterministic parameter-boundary search."""

    name: str = Field(min_length=1)
    base_experiment: ExperimentConfig

    parameter: BoundaryParameter = "target_bias"
    direction: BoundaryDirection = "negative"

    low: float = Field(ge=0.0)
    high: float = Field(gt=0.0)

    tolerance: float = Field(
        default=0.001,
        gt=0.0,
    )

    max_iterations: PositiveInt = 12

    @model_validator(mode="after")
    def validate_boundary_search(
        self,
    ) -> "BoundarySearchConfig":
        """Require compatible bounds and a MuJoCo servo experiment."""

        if self.low >= self.high:
            raise ValueError("Boundary-search low must be below high.")

        if self.tolerance >= self.high - self.low:
            raise ValueError(
                "Boundary-search tolerance must be smaller than the initial search interval."
            )

        if not isinstance(
            self.base_experiment.environment,
            MujocoReachEnvironmentConfig,
        ):
            raise ValueError("Boundary discovery currently requires a mujoco_reach environment.")

        if not isinstance(
            self.base_experiment.policy,
            PositionServoPolicyConfig,
        ):
            raise ValueError("Boundary discovery currently requires a position_servo policy.")

        if self.parameter in INTEGER_PARAMETERS:
            if not (float(self.low).is_integer() and float(self.high).is_integer()):
                raise ValueError(
                    "Integer parameters require whole-number magnitudes for low and high."
                )

            if self.tolerance < 1.0:
                raise ValueError("Integer parameters require a tolerance of at least 1.")

        for label, magnitude in (
            ("low", self.low),
            ("high", self.high),
        ):
            try:
                build_boundary_experiment(
                    self,
                    magnitude=magnitude,
                    name=f"{self.name}:{label}-validation",
                )
            except ValueError as exc:
                raise ValueError(
                    f"The {label} boundary produces an invalid {self.parameter} value: {exc}"
                ) from exc

        return self


def baseline_parameter_value(
    config: BoundarySearchConfig,
) -> float | int:
    """Return the selected parameter's baseline value."""

    if config.parameter in POLICY_PARAMETERS:
        return getattr(
            config.base_experiment.policy,
            config.parameter,
        )

    if config.parameter in ENVIRONMENT_PARAMETERS:
        return getattr(
            config.base_experiment.environment,
            config.parameter,
        )

    raise ValueError(f"Unsupported boundary parameter: {config.parameter}")


def boundary_parameter_value(
    config: BoundarySearchConfig,
    magnitude: float,
) -> float | int:
    """Convert a search magnitude into an applied parameter value."""

    baseline = baseline_parameter_value(config)

    sign = 1.0 if config.direction == "positive" else -1.0

    applied = float(baseline) + sign * magnitude

    if config.parameter in INTEGER_PARAMETERS:
        return int(round(applied))

    return applied


def build_boundary_experiment(
    config: BoundarySearchConfig,
    *,
    magnitude: float,
    name: str,
) -> ExperimentConfig:
    """Create an experiment with the selected parameter modified."""

    payload = config.base_experiment.model_dump(mode="python")

    container_name = "policy" if config.parameter in POLICY_PARAMETERS else "environment"

    container = payload.get(container_name)

    if not isinstance(container, dict):
        raise ValueError(f"Base experiment has an invalid {container_name} object.")

    container[config.parameter] = boundary_parameter_value(
        config,
        magnitude,
    )

    payload["name"] = name

    return ExperimentConfig.model_validate(payload)


def load_boundary_config(
    path: Path,
) -> BoundarySearchConfig:
    """Load and validate a boundary-search YAML file."""

    raw_data: Any = yaml.safe_load(path.read_text(encoding="utf-8"))

    if raw_data is None:
        raw_data = {}

    if not isinstance(raw_data, dict):
        raise ValueError("Boundary configuration root must be a YAML mapping.")

    return BoundarySearchConfig.model_validate(raw_data)
