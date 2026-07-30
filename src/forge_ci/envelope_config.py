"""Configuration models for robustness-envelope discovery."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import (
    BaseModel,
    Field,
    PositiveInt,
    model_validator,
)

from forge_ci.boundary_config import (
    BoundaryDirection,
    BoundaryParameter,
    BoundarySearchConfig,
)
from forge_ci.config import ExperimentConfig


class EnvelopeDimensionConfig(BaseModel):
    """One parameter direction within a robustness envelope."""

    name: str = Field(min_length=1)

    parameter: BoundaryParameter
    direction: BoundaryDirection

    low: float = Field(ge=0.0)
    high: float = Field(gt=0.0)

    tolerance: float = Field(
        default=0.001,
        gt=0.0,
    )

    max_iterations: PositiveInt = 12

    def to_boundary_config(
        self,
        *,
        envelope_name: str,
        base_experiment: ExperimentConfig,
    ) -> BoundarySearchConfig:
        """Convert this dimension into a boundary search."""

        return BoundarySearchConfig(
            name=f"{envelope_name}:{self.name}",
            base_experiment=base_experiment,
            parameter=self.parameter,
            direction=self.direction,
            low=self.low,
            high=self.high,
            tolerance=self.tolerance,
            max_iterations=self.max_iterations,
        )


class RobustnessEnvelopeConfig(BaseModel):
    """Configuration for several related boundary searches."""

    name: str = Field(min_length=1)

    base_experiment: ExperimentConfig

    dimensions: list[EnvelopeDimensionConfig] = Field(
        min_length=2
    )

    @model_validator(mode="after")
    def validate_dimensions(
        self,
    ) -> "RobustnessEnvelopeConfig":
        """Require unique names and valid boundary searches."""

        names = [
            dimension.name
            for dimension in self.dimensions
        ]

        if len(names) != len(set(names)):
            raise ValueError(
                "Envelope dimension names must be unique."
            )

        for dimension in self.dimensions:
            dimension.to_boundary_config(
                envelope_name=self.name,
                base_experiment=self.base_experiment,
            )

        return self


def load_envelope_config(
    path: Path,
) -> RobustnessEnvelopeConfig:
    """Load and validate a robustness-envelope YAML file."""

    raw_data: Any = yaml.safe_load(
        path.read_text(encoding="utf-8")
    )

    if raw_data is None:
        raw_data = {}

    if not isinstance(raw_data, dict):
        raise ValueError(
            "Envelope configuration root must be "
            "a YAML mapping."
        )

    return RobustnessEnvelopeConfig.model_validate(
        raw_data
    )
