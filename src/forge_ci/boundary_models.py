"""Structured results for parameter-boundary discovery."""

from pydantic import BaseModel, Field


class BoundaryTrial(BaseModel):
    """One evaluated point during boundary discovery."""

    index: int = Field(ge=0)
    label: str

    parameter_value: float
    applied_parameter_value: float | int

    success_rate: float = Field(
        ge=0.0,
        le=1.0,
    )

    mean_steps: float = Field(ge=0.0)
    gate_passed: bool

    run_directory: str


class BoundarySearchSummary(BaseModel):
    """Summary of a completed parameter-boundary search."""

    search_name: str
    parameter: str
    direction: str

    baseline_parameter_value: float | int

    largest_passing_value: float
    smallest_failing_value: float

    passing_applied_value: float | int
    failing_applied_value: float | int

    boundary_width: float = Field(ge=0.0)

    converged: bool
    search_iterations: int = Field(ge=0)

    counterexample_run_directory: str
    counterexample_config: str

    dominant_failure_type: str | None

    trials: list[BoundaryTrial]
