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


class BoundarySearchConfig(BaseModel):
    """Configuration for deterministic parameter-boundary search."""

    name: str = Field(min_length=1)

    base_experiment: ExperimentConfig

    parameter: Literal["target_bias"] = "target_bias"

    direction: Literal["positive", "negative"] = "negative"

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
            raise ValueError(
                "Boundary-search low must be below high."
            )

        if self.tolerance >= self.high - self.low:
            raise ValueError(
                "Boundary-search tolerance must be smaller "
                "than the initial search interval."
            )

        if not isinstance(
            self.base_experiment.environment,
            MujocoReachEnvironmentConfig,
        ):
            raise ValueError(
                "Boundary discovery currently requires "
                "a mujoco_reach environment."
            )

        if not isinstance(
            self.base_experiment.policy,
            PositionServoPolicyConfig,
        ):
            raise ValueError(
                "Boundary discovery currently requires "
                "a position_servo policy."
            )

        return self


def load_boundary_config(
    path: Path,
) -> BoundarySearchConfig:
    """Load and validate a boundary-search YAML file."""

    raw_data: Any = yaml.safe_load(
        path.read_text(encoding="utf-8")
    )

    if raw_data is None:
        raw_data = {}

    if not isinstance(raw_data, dict):
        raise ValueError(
            "Boundary configuration root must be "
            "a YAML mapping."
        )

    return BoundarySearchConfig.model_validate(
        raw_data
    )
