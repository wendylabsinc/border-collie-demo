"""Deterministic Go2 environment simulator B component."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class SimulatedInterpretation:
    stream: str
    observation_type: str
    frame_id: str
    confidence: float
    payload: dict[str, Any]


def interpret_world(at: datetime) -> list[SimulatedInterpretation]:
    """Return a repeatable toy world using time as the only input."""
    seconds = at.astimezone(timezone.utc).timestamp()
    phase = seconds % 20.0
    x = round(phase * 0.05, 3)
    obstacle_distance = round(0.6 + abs(math.sin(phase / 4.0)) * 2.4, 3)
    return [
        SimulatedInterpretation(
            stream="localization.pose",
            observation_type="robot.pose.2d",
            frame_id="map",
            confidence=0.99,
            payload={"x_m": x, "y_m": 0.0, "yaw_rad": 0.0},
        ),
        SimulatedInterpretation(
            stream="vision.obstacles",
            observation_type="obstacles.nearest",
            frame_id="base_link",
            confidence=0.95,
            payload={
                "nearest_distance_m": obstacle_distance,
                "bearing_rad": 0.2,
                "class": "simulated-box",
            },
        ),
        SimulatedInterpretation(
            stream="terrain.clearance",
            observation_type="terrain.clearance",
            frame_id="base_link",
            confidence=1.0,
            payload={"clearance_m": 0.35, "traversable": True},
        ),
    ]

