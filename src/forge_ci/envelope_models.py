"""Structured results for robustness-envelope discovery."""

from pydantic import BaseModel, Field


class EnvelopeDimensionResult(BaseModel):
    """Boundary result for one envelope dimension."""

    name: str

    parameter: str
    direction: str

    baseline_parameter_value: float | int

    largest_passing_magnitude: float
    smallest_failing_magnitude: float

    passing_applied_value: float | int
    failing_applied_value: float | int

    boundary_width: float = Field(ge=0.0)

    normalized_boundary_position: float = Field(
        ge=0.0,
        le=1.0,
    )

    converged: bool

    dominant_failure_type: str | None
    search_directory: str


class RobustnessEnvelopeSummary(BaseModel):
    """Summary and ranking of all envelope dimensions."""

    envelope_name: str

    all_converged: bool
    weakest_dimension: str

    dimensions: list[EnvelopeDimensionResult]
