import numpy as np
import pytest

from src.genome import Genome


def test_default_values():
    g = Genome()
    assert g.speed == 1.8
    assert g.agility == 1.2
    assert g.segments == 9
    assert 0.0 <= g.adaptation <= 1.0
    assert len(g.color) == 3


def test_random_is_within_bounds():
    rng = np.random.default_rng(42)
    for _ in range(50):
        g = Genome.random(rng)
        assert 0.3 <= g.speed <= 6.0
        assert 0.1 <= g.agility <= 6.0
        assert 4 <= g.segments <= 14
        assert 0.0 <= g.adaptation <= 1.0


def test_mutate_stays_in_bounds():
    rng = np.random.default_rng(0)
    g = Genome()
    for _ in range(200):
        m = g.mutate(sigma=0.5, rng=rng)
        assert 0.3 <= m.speed <= 6.0
        assert 0.1 <= m.agility <= 6.0
        assert 4 <= m.segments <= 14
        assert 0.0 <= m.adaptation <= 1.0
        assert 0.05 <= m.und_amp <= 3.0
        assert 0.3 <= m.und_freq <= 8.0


def test_mutate_large_sigma_still_bounded():
    rng = np.random.default_rng(7)
    g = Genome()
    for _ in range(100):
        m = g.mutate(sigma=5.0, rng=rng)
        assert 0.3 <= m.speed <= 6.0
        assert 4 <= m.segments <= 14


def test_clone_is_independent():
    g = Genome(speed=3.0, segments=11)
    c = g.clone()
    assert c.speed == g.speed
    assert c.segments == g.segments
    # Mutating clone does not affect original
    c.speed = 99.0
    assert g.speed == 3.0


def test_repr_contains_key_fields():
    g = Genome()
    r = repr(g)
    assert "Genome" in r
    assert "speed" in r
    assert "segments" in r


def test_color_tuple_range():
    rng = np.random.default_rng(1)
    for _ in range(30):
        g = Genome.random(rng)
        for channel in g.color:
            assert 30 <= channel <= 220
