"""Genome module – genetic parameters for the worm evolution simulation."""
from __future__ import annotations

import numpy as np


class Genome:
    """Represents the heritable parameters of a worm creature."""

    def __init__(
        self,
        speed: float = 1.8,
        agility: float = 1.2,
        und_freq: float = 2.2,
        und_amp: float = 1.1,
        segments: int = 9,
        adaptation: float = 0.05,
        color: tuple[float, float, float] | None = None,
    ) -> None:
        self.speed = float(speed)
        self.agility = float(agility)
        self.und_freq = float(und_freq)
        self.und_amp = float(und_amp)
        self.segments = int(segments)
        self.adaptation = float(adaptation)
        rng = np.random.default_rng()
        self.color: tuple[float, float, float] = color or (
            float(rng.uniform(30, 220)),
            float(rng.uniform(30, 220)),
            float(rng.uniform(30, 180)),
        )

    @classmethod
    def random(cls, rng: np.random.Generator | None = None) -> "Genome":
        """Create a genome with randomised starting values."""
        rng = rng or np.random.default_rng()
        return cls(
            speed=float(rng.uniform(1.0, 2.5)),
            agility=float(rng.uniform(0.8, 1.8)),
            und_freq=float(rng.uniform(1.0, 3.0)),
            und_amp=float(rng.uniform(0.5, 1.5)),
            segments=int(rng.integers(6, 11)),
            adaptation=float(rng.uniform(0.0, 0.15)),
            color=(float(rng.uniform(30, 220)), float(rng.uniform(30, 220)), float(rng.uniform(30, 180))),
        )

    def mutate(self, sigma: float, rng: np.random.Generator | None = None) -> "Genome":
        """Return a new genome perturbed by Gaussian noise scaled by sigma."""
        rng = rng or np.random.default_rng()
        n = rng.standard_normal

        def clip(val: float, lo: float, hi: float) -> float:
            return float(np.clip(val, lo, hi))

        return Genome(
            speed=clip(self.speed + n() * sigma * 0.35, 0.3, 6.0),
            agility=clip(self.agility + n() * sigma * 0.28, 0.1, 6.0),
            und_freq=clip(self.und_freq + n() * sigma * 0.45, 0.3, 8.0),
            und_amp=clip(self.und_amp + n() * sigma * 0.28, 0.05, 3.0),
            segments=int(np.clip(round(self.segments + n() * sigma * 0.7), 4, 14)),
            adaptation=clip(self.adaptation + n() * sigma * 0.12, 0.0, 1.0),
            color=tuple(clip(c + n() * sigma * 12, 30, 220) for c in self.color),  # type: ignore[arg-type]
        )

    def clone(self) -> "Genome":
        """Return an exact copy."""
        return Genome(
            speed=self.speed,
            agility=self.agility,
            und_freq=self.und_freq,
            und_amp=self.und_amp,
            segments=self.segments,
            adaptation=self.adaptation,
            color=self.color,
        )

    def __repr__(self) -> str:
        return (
            f"Genome(speed={self.speed:.2f}, agility={self.agility:.2f}, "
            f"segments={self.segments}, adaptation={self.adaptation:.3f}, "
            f"und_freq={self.und_freq:.2f}, und_amp={self.und_amp:.2f})"
        )
