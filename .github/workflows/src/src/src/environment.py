from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class EnvironmentType(str, Enum):
    WATER = "water"
    LAND = "land"
    MOUNTAIN = "mountain"


@dataclass(frozen=True)
class Obstacle:
    x: float
    y: float
    radius: float

    def contains(self, px: float, py: float, margin: float = 0.0) -> bool:
        """Return True if point (px, py) is within radius + margin."""
        return math.hypot(px - self.x, py - self.y) < (self.radius + margin)


OBSTACLES: dict[EnvironmentType, list[Obstacle]] = {
    EnvironmentType.WATER: [
        Obstacle(74, 72, 21),
        Obstacle(158, 92, 19),
        Obstacle(48, 162, 25),
        Obstacle(174, 172, 17),
        Obstacle(104, 232, 21),
    ],
    EnvironmentType.LAND: [
        Obstacle(84, 66, 17),
        Obstacle(174, 106, 21),
        Obstacle(54, 150, 15),
        Obstacle(150, 190, 23),
        Obstacle(100, 246, 18),
        Obstacle(192, 244, 14),
    ],
    EnvironmentType.MOUNTAIN: [
        Obstacle(66, 50, 29),
        Obstacle(166, 80, 25),
        Obstacle(46, 180, 21),
        Obstacle(180, 156, 31),
        Obstacle(120, 220, 23),
        Obstacle(94, 120, 18),
    ],
}


def speed_multiplier(env: EnvironmentType, adaptation: float) -> float:
    """
    Compute the environment-specific speed multiplier.

    Higher adaptation values reduce the terrain penalty.

    Args:
        env:        The environment type.
        adaptation: Genome adaptation value in [0, 1].

    Returns:
        A float multiplier applied to the worm's base speed.
    """
    match env:
        case EnvironmentType.WATER:
            return 0.42 + adaptation * 0.62
        case EnvironmentType.LAND:
            return 0.68 + adaptation * 0.38
        case EnvironmentType.MOUNTAIN:
            return 0.32 + adaptation * 0.78
        case _:
            return 1.0
