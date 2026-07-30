"""Structured evaluation and comparison results."""

from pydantic import BaseModel, Field, NonNegativeInt


class EpisodeResult(BaseModel):
    """Result of one independent evaluation episode."""

    episode: NonNegativeInt
    seed: int
    success: bool
    steps: NonNegativeInt
    total_reward: float
    final_position: NonNegativeInt
    failure_reason: str | None = None


class RunSummary(BaseModel):
    """Aggregate metrics and CI-gate decision for an evaluation run."""

    experiment_name: str
    episodes: int = Field(gt=0)
    successes: NonNegativeInt
    failures: NonNegativeInt
    success_rate: float = Field(ge=0.0, le=1.0)
    mean_steps: float = Field(ge=0.0)
    min_success_rate: float = Field(ge=0.0, le=1.0)
    gate_passed: bool


class ComparisonResult(BaseModel):
    """Regression comparison between a baseline and candidate run."""

    baseline_run_id: str
    candidate_run_id: str

    baseline_success_rate: float = Field(ge=0.0, le=1.0)
    candidate_success_rate: float = Field(ge=0.0, le=1.0)
    success_rate_delta: float

    baseline_mean_steps: float = Field(ge=0.0)
    candidate_mean_steps: float = Field(ge=0.0)
    mean_steps_delta: float

    max_success_rate_drop: float = Field(ge=0.0, le=1.0)
    max_mean_steps_increase: float = Field(ge=0.0)

    gate_passed: bool
    reasons: list[str] = Field(default_factory=list)
