import numpy as np
import pytest

from src.environment import EnvironmentType, Obstacle, speed_multiplier
from src.evolution_strategy import ESState, evolve_step
from src.genome import Genome


# ── ESState ──────────────────────────────────────────────────────────────────

def test_es_initial_state():
    state = ESState(genome=Genome())
    assert state.generation == 0
    assert state.sigma == 0.5
    assert state.avg_reward == 0.0
    assert state.best_reward == float("-inf")


def test_record_reward_updates_avg():
    state = ESState(genome=Genome())
    for r in [2.0, 4.0, 6.0]:
        state.record_reward(r)
    assert abs(state.avg_reward - 4.0) < 1e-9


def test_record_reward_tracks_best():
    state = ESState(genome=Genome())
    state.record_reward(3.0)
    state.record_reward(10.0)
    state.record_reward(5.0)
    assert state.best_reward == 10.0
    assert state.best_genome is not None


def test_history_capped_at_max():
    state = ESState(genome=Genome())
    for i in range(30):
        state.record_reward(float(i))
    from src.evolution_strategy import MAX_HISTORY
    assert len(state.reward_history) == MAX_HISTORY


def test_avg_reward_empty():
    state = ESState(genome=Genome())
    assert state.avg_reward == 0.0


# ── evolve_step ───────────────────────────────────────────────────────────────

def test_evolve_increments_generation():
    rng = np.random.default_rng(42)
    state = ESState(genome=Genome())
    state.record_reward(2.0)
    state = evolve_step(state, lambda_=5, rng=rng)
    assert state.generation == 1


def test_evolve_multiple_steps():
    rng = np.random.default_rng(99)
    state = ESState(genome=Genome())
    for _ in range(5):
        state.record_reward(3.0)
    for _ in range(10):
        state = evolve_step(state, rng=rng)
    assert state.generation == 10


def test_sigma_decreases_on_no_improvement():
    rng = np.random.default_rng(5)
    state = ESState(genome=Genome(), sigma=0.5, last_avg_reward=100.0)
    for _ in range(12):
        state.record_reward(0.1)  # all worse than last_avg_reward=100
    initial_sigma = state.sigma
    state = evolve_step(state, rng=rng)
    assert state.sigma < initial_sigma


def test_sigma_increases_on_improvement():
    rng = np.random.default_rng(3)
    state = ESState(genome=Genome(), sigma=0.5, last_avg_reward=0.0)
    for _ in range(12):
        state.record_reward(5.0)  # all better than last_avg_reward=0
    initial_sigma = state.sigma
    state = evolve_step(state, rng=rng)
    assert state.sigma > initial_sigma


def test_evolve_sigma_stays_bounded():
    rng = np.random.default_rng(8)
    state = ESState(genome=Genome())
    for _ in range(50):
        state.record_reward(float(rng.uniform(0, 10)))
        state = evolve_step(state, rng=rng)
    assert 0.018 <= state.sigma <= 0.92


# ── Environment ───────────────────────────────────────────────────────────────

def test_obstacle_contains():
    obs = Obstacle(100, 100, 20)
    assert obs.contains(100, 100)
    assert obs.contains(115, 100)
    assert not obs.contains(125, 100)


def test_obstacle_contains_with_margin():
    obs = Obstacle(50, 50, 10)
    assert not obs.contains(65, 50)
    assert obs.contains(65, 50, margin=10)


def test_speed_multiplier_water():
    val = speed_multiplier(EnvironmentType.WATER, 0.0)
    assert abs(val - 0.42) < 1e-9
    val_full = speed_multiplier(EnvironmentType.WATER, 1.0)
    assert val_full > val


def test_speed_multiplier_land():
    val = speed_multiplier(EnvironmentType.LAND, 0.0)
    assert abs(val - 0.68) < 1e-9


def test_speed_multiplier_mountain():
    val = speed_multiplier(EnvironmentType.MOUNTAIN, 0.0)
    assert abs(val - 0.32) < 1e-9
    # Mountain should benefit most from adaptation
    val_adapted = speed_multiplier(EnvironmentType.MOUNTAIN, 1.0)
    assert val_adapted > speed_multiplier(EnvironmentType.LAND, 1.0) - 0.1


def test_speed_multiplier_increases_with_adaptation():
    for env in EnvironmentType:
        low = speed_multiplier(env, 0.0)
        high = speed_multiplier(env, 1.0)
        assert high > low
