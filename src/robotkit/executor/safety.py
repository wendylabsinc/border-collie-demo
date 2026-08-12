"""Deterministic safety gate between abstract effects and ROS2."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from robotkit.contracts import EffectRecord, WorldSnapshot


@dataclass(frozen=True)
class SafetyVerdict:
    allowed: bool
    reason: str


def validate_effect(
    effect: EffectRecord,
    snapshot: WorldSnapshot,
    at: datetime,
    *,
    active_goal_id: UUID | None = None,
    max_linear_mps: float = 0.5,
    max_angular_rps: float = 1.0,
) -> SafetyVerdict:
    if active_goal_id is not None and effect.goal_id != active_goal_id:
        return SafetyVerdict(False, "effect does not target the active goal")
    if effect.valid_until <= at:
        return SafetyVerdict(False, "effect expired")
    drift = snapshot.state_revision - effect.based_on_revision
    if drift < 0 or drift > effect.preconditions.max_world_revision_drift:
        return SafetyVerdict(False, f"world-state drift is {drift} revisions")

    by_stream = {item.stream: item for item in snapshot.observations}
    for stream in effect.preconditions.require_fresh_streams:
        observation = by_stream.get(stream)
        if observation is None:
            return SafetyVerdict(False, f"required stream missing: {stream}")
        if observation.is_stale(at):
            return SafetyVerdict(False, f"required stream stale: {stream}")

    mission_metadata = {
        "mission_schema_version",
        "mission_type",
        "mission_stage",
        "trigger_event_id",
        "stage_complete",
        "decision_reason",
    }
    if effect.effect_type in {"unitree_bark", "audio"}:
        # Audio assets are selected by deployment configuration, never a path in
        # an effect.  Keep the accepted semantic vocabulary deliberately tiny.
        if set(effect.parameters) - ({"sound"} | mission_metadata):
            return SafetyVerdict(False, "audio effect contains unsupported parameters")
        if effect.parameters.get("sound", "bark") != "bark":
            return SafetyVerdict(False, "unsupported audio sound")
        return SafetyVerdict(True, "allowed")
    if effect.effect_type == "unitree_lie_down":
        if set(effect.parameters) - ({"posture"} | mission_metadata):
            return SafetyVerdict(False, "lie-down effect contains unsupported parameters")
        if effect.parameters.get("posture", "lie_down") != "lie_down":
            return SafetyVerdict(False, "unsupported posture")
        return SafetyVerdict(True, "allowed")
    if effect.effect_type != "cmd_vel":
        return SafetyVerdict(False, f"unsupported effect type: {effect.effect_type}")
    try:
        linear = float(effect.parameters["linear_x_mps"])
        angular = float(effect.parameters["angular_z_rps"])
    except (KeyError, TypeError, ValueError):
        return SafetyVerdict(False, "cmd_vel requires numeric linear_x_mps and angular_z_rps")
    if not math.isfinite(linear) or not math.isfinite(angular):
        return SafetyVerdict(False, "cmd_vel values must be finite")
    if abs(linear) > max_linear_mps:
        return SafetyVerdict(False, "linear velocity exceeds safety limit")
    if abs(angular) > max_angular_rps:
        return SafetyVerdict(False, "angular velocity exceeds safety limit")
    return SafetyVerdict(True, "allowed")
