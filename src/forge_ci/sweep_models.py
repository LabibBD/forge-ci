"""Structured results for robustness sweeps."""

from pydantic import BaseModel, Field


class ScenarioSweepResult(BaseModel):
    """Metrics produced by one sweep scenario."""

    scenario_name: str
    run_directory: str

    actuator_delay_steps: int = Field(ge=0)
    control_noise_std: float = Field(ge=0.0)
    joint_damping: float = Field(gt=0.0)

    success_rate: float = Field(
        ge=0.0,
        le=1.0,
    )

    mean_steps: float = Field(ge=0.0)
    gate_passed: bool


class RobustnessSweepSummary(BaseModel):
    """Aggregate result of a robustness sweep."""

    sweep_name: str

    scenario_count: int = Field(gt=0)
    worst_success_rate: float = Field(
        ge=0.0,
        le=1.0,
    )

    worst_mean_steps: float = Field(ge=0.0)

    min_scenario_success_rate: float = Field(
        ge=0.0,
        le=1.0,
    )

    max_scenario_mean_steps: float = Field(gt=0.0)

    gate_passed: bool
    reasons: list[str] = Field(default_factory=list)

    scenarios: list[ScenarioSweepResult]
