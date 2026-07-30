"""Small deterministic environment used to validate the evaluation engine."""

from dataclasses import dataclass
from random import Random


@dataclass
class LineWorld:
    """A one-dimensional world in which the agent must reach a goal position."""

    goal: int
    max_steps: int
    slip_probability: float
    position: int = 0
    steps: int = 0

    def reset(self) -> int:
        """Reset the environment and return the initial observation."""

        self.position = 0
        self.steps = 0
        return self.position

    def step(self, action: int, rng: Random) -> tuple[int, float, bool, bool]:
        """Apply one action and return observation, reward, done, and success."""

        if action not in {-1, 0, 1}:
            raise ValueError("LineWorld actions must be -1, 0, or 1.")

        self.steps += 1

        applied_action = (
            0 if rng.random() < self.slip_probability else action
        )

        self.position = max(
            0,
            min(self.goal, self.position + applied_action),
        )

        success = self.position >= self.goal
        done = success or self.steps >= self.max_steps
        reward = 1.0 if success else -0.01

        return self.position, reward, done, success


@dataclass(frozen=True)
class GreedyPolicy:
    """A baseline policy that always moves toward the goal."""

    goal: int

    def act(self, observation: int) -> int:
        """Choose the next action from the current position."""

        return 1 if observation < self.goal else 0
