"""Configuration models for MuJoCo robustness sweeps."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator

from forge_ci.config import (
    ExperimentConfig,
    MujocoReachEnvironmentConfig,
)


class DisturbanceScenarioConfig(BaseModel):
    """One named MuJoCo disturbance scenario."""

    name: str = Field(
        min_length=1,
        pattern=r"^[a-z0-9][a-z0-9_-]*$",
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


class SweepGateConfig(BaseModel):
    """Robustness requirements applied to every scenario."""

    min_scenario_success_rate: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
    )

    max_scenario_mean_steps: float = Field(
        default=1500.0,
        gt=0.0,
    )


class RobustnessSweepConfig(BaseModel):
    """Configuration for one MuJoCo robustness sweep."""

    name: str = Field(min_length=1)

    base_experiment: ExperimentConfig

    scenarios: list[DisturbanceScenarioConfig] = Field(
        min_length=1
    )

    gate: SweepGateConfig = Field(
        default_factory=SweepGateConfig
    )

    @model_validator(mode="after")
    def validate_sweep(
        self,
    ) -> "RobustnessSweepConfig":
        """Require MuJoCo and uniquely named scenarios."""

        if not isinstance(
            self.base_experiment.environment,
            MujocoReachEnvironmentConfig,
        ):
            raise ValueError(
                "Robustness sweeps currently require "
                "a mujoco_reach base environment."
            )

        scenario_names = [
            scenario.name
            for scenario in self.scenarios
        ]

        if len(scenario_names) != len(set(scenario_names)):
            raise ValueError(
                "Scenario names must be unique."
            )

        return self


def load_sweep_config(
    path: Path,
) -> RobustnessSweepConfig:
    """Load and validate a robustness-sweep YAML file."""

    raw_data: Any = yaml.safe_load(
        path.read_text(encoding="utf-8")
    )

    if raw_data is None:
        raw_data = {}

    if not isinstance(raw_data, dict):
        raise ValueError(
            "Sweep configuration root must be "
            "a YAML mapping."
        )

    return RobustnessSweepConfig.model_validate(
        raw_data
    )
