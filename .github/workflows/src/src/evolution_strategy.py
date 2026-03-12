from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .genome import Genome

MAX_HISTORY = 12


@dataclass
class ESState:
    """Tracks the running state of one worm's evolution process."""

    genome: Genome
    sigma: float = 0.5
    generation: int = 0
    best_reward: float = float("-inf")
    best_genome: Genome | None = None
    reward_history: list[float] = field(default_factory=list)
    last_avg_reward: float = 0.0

    def record_reward(self, reward: float) -> None:
        """Store an episode reward; trims history to MAX_HISTORY entries."""
        self.reward_history.append(reward)
        if len(self.reward_history) > MAX_HISTORY:
            self.reward_history.pop(0)
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_genome = self.genome.clone()

    @property
    def avg_reward(self) -> float:
        """Mean reward over recorded history."""
        if not self.reward_history:
            return 0.0
        return sum(self.reward_history) / len(self.reward_history)


def evolve_step(
    state: ESState,
    lambda_: int = 5,
    rng: np.random.Generator | None = None,
) -> ESState:
    """
    Perform one (1+λ)-ES step in-place and return the updated state.

    Generates λ candidate genomes, selects the best via noisy scoring,
    and accepts it if it improves on the current average reward (or with
    a 28 % random exploration probability). Adapts σ via the 1/5 success rule.

    Args:
        state:   Current ES state (modified in-place).
        lambda_: Number of offspring candidates to generate.
        rng:     NumPy random generator; created fresh if None.

    Returns:
        The same ESState object after updating genome, sigma, generation.
    """
    rng = rng or np.random.default_rng()
    avg_r = state.avg_reward

    candidates = [state.genome.mutate(state.sigma, rng) for _ in range(lambda_)]
    noisy_scores = avg_r + rng.standard_normal(lambda_) * 0.6
    best_candidate = candidates[int(np.argmax(noisy_scores))]

    improved = avg_r > state.last_avg_reward * 0.88
    if improved or rng.random() < 0.28:
        state.genome = best_candidate

    if avg_r > state.best_reward:
        state.best_reward = avg_r
        state.best_genome = state.genome.clone()

    # 1/5 success rule
    n_success = sum(1 for r in state.reward_history if r > state.last_avg_reward)
    success_rate = n_success / max(1, len(state.reward_history))
    if success_rate > 0.2:
        state.sigma = min(0.92, state.sigma * 1.12)
    else:
        state.sigma = max(0.018, state.sigma * 0.91)

    state.last_avg_reward = avg_r
    state.generation += 1
    return state
