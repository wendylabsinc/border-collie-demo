from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from time import monotonic


class MissionPhase(str, Enum):
    IDLE = "idle"
    PREFLIGHT = "preflight"
    CAPTURE_HOME = "capture_home"
    WAIT_FOR_COMMAND = "wait_for_command"
    TURN_TO_FRUIT = "turn_to_fruit"
    FIND_FRUIT = "find_fruit"
    APPROACH_FRUIT = "approach_fruit"
    ARRIVED = "arrived"
    SIT_AND_BARK = "sit_and_bark"
    STAND = "stand"
    TURN_TOWARD_HOME = "turn_toward_home"
    RETURN_HOME = "return_home"
    RESTORE_HEADING = "restore_heading"
    COMPLETE = "complete"
    STOPPED = "stopped"
    FAILED = "failed"
    REMOTE_TAKEOVER = "remote_takeover"


@dataclass(frozen=True)
class CameraFrame:
    frame_id: int
    captured_monotonic_s: float
    generation: int
    jpeg: bytes


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    bbox_xyxy: tuple[int, int, int, int]
    frame_id: int
    captured_monotonic_s: float


@dataclass(frozen=True)
class Pose:
    x_m: float
    y_m: float
    yaw_rad: float
    captured_monotonic_s: float


@dataclass(frozen=True)
class VelocityCommand:
    forward_mps: float = 0.0
    yaw_rps: float = 0.0


@dataclass(frozen=True)
class MissionEvent:
    phase: MissionPhase
    reason: str
    recorded_monotonic_s: float

    @classmethod
    def record(cls, phase: MissionPhase, reason: str) -> MissionEvent:
        return cls(phase, reason, monotonic())


@dataclass(frozen=True)
class RemoteInput:
    source: str
    control: str
    received_monotonic_s: float
