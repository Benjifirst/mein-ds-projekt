from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .environment import EnvironmentType, Obstacle, speed_multiplier
from .genome import Genome

CANVAS_W: float = 226.0
CANVAS_H: float = 296.0
SEG_LEN: float = 9.0
HEAD_RADIUS: float = 9.0
EPISODE_TIMEOUT: float = 14.0


@dataclass
class Segment:
    x: float
    y: float


@dataclass
class RewardWeights:
    speed: float = 1.0
    obstacle: float = 1.0
    proximity: float = 1.0


@dataclass
class WormState:
    env: EnvironmentType
    genome: Genome
    segments: list[Segment] = field(default_factory=list)
    head_angle: float = 0.0
    episode_reward: float = 0.0
    episode_time: float = 0.0
    reached: bool = False
    light_x: float = 50.0
    light_y: float = 50.0

    def __post_init__(self) -> None:
        if not self.segments:
            self.segments = _init_segments(self.genome.segments)


def _init_segments(n: int, cx: float = CANVAS_W / 2, cy: float = CANVAS_H / 2) -> list[Segment]:
    return [Segment(cx - i * SEG_LEN, cy) for i in range(n)]


def place_light(
    obstacles: list[Obstacle],
    rng: np.random.Generator,
    margin: float = 22.0,
) -> tuple[float, float]:
    """Find a light position not inside any obstacle."""
    for _ in range(100):
        lx = float(rng.uniform(margin, CANVAS_W - margin))
        ly = float(rng.uniform(margin, CANVAS_H - margin))
        if not any(o.contains(lx, ly, margin=20) for o in obstacles):
            return lx, ly
    return CANVAS_W / 2, margin


def step(
    state: WormState,
    obstacles: list[Obstacle],
    weights: RewardWeights,
    dt: float,
    rng: np.random.Generator | None = None,
) -> tuple[WormState, float]:
    """
    Advance the worm by one physics step and return (updated_state, reward).

    Applies stochastic steering toward the light source, obstacle repulsion,
    follow-the-leader segment kinematics, and collision resolution.

    Args:
        state:     Current worm state (mutated in-place).
        obstacles: List of obstacles for the current environment.
        weights:   Reward weight configuration.
        dt:        Time delta in seconds.
        rng:       Random generator; created if None.

    Returns:
        Tuple of (state, step_reward).
    """
    rng = rng or np.random.default_rng()
    state.episode_time += dt
    head = state.segments[0]

    dx = state.light_x - head.x
    dy = state.light_y - head.y
    dist = math.hypot(dx, dy)

    # Obstacle repulsion
    rep_x, rep_y = 0.0, 0.0
    for obs in obstacles:
        ox, oy = head.x - obs.x, head.y - obs.y
        d = math.hypot(ox, oy)
        threshold = obs.radius + HEAD_RADIUS + 24
        if d < threshold and d > 0:
            strength = (threshold - d) / threshold
            rep_x += (ox / d) * strength * 5
            rep_y += (oy / d) * strength * 5

    target_angle = math.atan2(dy + rep_y * dist / 80, dx + rep_x * dist / 80)
    noise = float(rng.standard_normal()) / (state.genome.agility * 0.75 + 0.35)
    desired = target_angle + noise

    # Smooth steering
    da = desired - state.head_angle
    while da > math.pi:
        da -= 2 * math.pi
    while da < -math.pi:
        da += 2 * math.pi
    max_turn = state.genome.agility * 2.2 * dt
    state.head_angle += max(-max_turn, min(max_turn, da))

    mult = speed_multiplier(state.env, state.genome.adaptation)
    spd = state.genome.speed * mult * 48 * dt
    nx = max(HEAD_RADIUS, min(CANVAS_W - HEAD_RADIUS, head.x + math.cos(state.head_angle) * spd))
    ny = max(HEAD_RADIUS, min(CANVAS_H - HEAD_RADIUS, head.y + math.sin(state.head_angle) * spd))

    collided = False
    for obs in obstacles:
        ox, oy = nx - obs.x, ny - obs.y
        d = math.hypot(ox, oy)
        if d < obs.radius + HEAD_RADIUS and d > 0:
            push = obs.radius + HEAD_RADIUS + 0.5
            nx = obs.x + (ox / d) * push
            ny = obs.y + (oy / d) * push
            collided = True

    state.segments[0] = Segment(nx, ny)
    for i in range(1, len(state.segments)):
        p, c = state.segments[i - 1], state.segments[i]
        sdx, sdy = c.x - p.x, c.y - p.y
        sd = math.hypot(sdx, sdy)
        if sd > SEG_LEN and sd > 0:
            state.segments[i] = Segment(p.x + (sdx / sd) * SEG_LEN, p.y + (sdy / sd) * SEG_LEN)

    reward = max(0.0, (1 - dist / 220) * dt * 3 * weights.proximity)
    if collided:
        reward -= 0.18 * weights.obstacle
    if dist < 15:
        state.reached = True
        reward += 6.0 + max(0.0, 7.0 - state.episode_time) * weights.speed

    state.episode_reward += reward
    return state, reward
