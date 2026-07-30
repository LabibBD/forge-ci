"""Structured evaluation results."""

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
