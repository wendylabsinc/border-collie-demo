"""Pure deterministic (mission Goal + world state) -> short-term Effect."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from robotkit.contracts import GoalRecord, ObservationRecord, WorldSnapshot
from robotkit.executor.mission import MissionConfig, decide_mission_action
from robotkit.fruits import SUPPORTED_FRUITS


@dataclass(frozen=True)
class EffectDecision:
    effect_type: str
    parameters: dict[str, Any] = field(default_factory=dict)
    require_fresh_streams: list[str] = field(default_factory=list)


def _mission_metadata(goal: GoalRecord, *, stage_complete: bool, reason: str) -> dict[str, Any]:
    mission_type = str(goal.parameters.get("mission_type", "")).casefold()
    metadata = {
        "mission_schema_version": str(
            goal.parameters.get("mission_schema_version", "1")
        ),
        "mission_type": goal.parameters.get("mission_type"),
        "mission_stage": goal.parameters.get("mission_stage", goal.goal_type),
        "trigger_event_id": goal.parameters.get("trigger_event_id"),
        "stage_complete": stage_complete,
        "decision_reason": reason,
    }
    if mission_type in SUPPORTED_FRUITS:
        metadata["target"] = goal.parameters.get("target", mission_type)
    return metadata


def _fresh_yolo_failure(snapshot: WorldSnapshot) -> ObservationRecord | None:
    failures = [
        observation
        for observation in snapshot.observations
        if observation.stream == "diagnostics.yolo"
        and not observation.is_stale(snapshot.captured_at)
        and str(observation.payload.get("status", "")).strip().casefold() == "failed"
    ]
    return max(
        failures,
        key=lambda item: (item.observed_at, item.revision, str(item.event_id)),
        default=None,
    )


def _fresh_stop_command(snapshot: WorldSnapshot) -> ObservationRecord | None:
    commands = [
        observation
        for observation in snapshot.observations
        if observation.stream in {"voice.intent", "website.intent"}
        and not observation.is_stale(snapshot.captured_at)
    ]
    latest = max(
        commands,
        key=lambda item: (item.observed_at, item.revision, str(item.event_id)),
        default=None,
    )
    if latest is None:
        return None
    intent = str(
        latest.payload.get("intent", latest.payload.get("command", ""))
    ).strip().casefold()
    return latest if intent == "stop" else None


def control(
    snapshot: WorldSnapshot,
    goal: GoalRecord,
    *,
    mission_config: MissionConfig = MissionConfig(),
) -> EffectDecision:
    """Translate a mission stage into one bounded, independently auditable action."""

    stage = str(goal.parameters.get("mission_stage", goal.goal_type))

    # Do not wait for the planner to replace the current mission before
    # cancelling motion. The resulting zero-velocity effect is still tied to
    # the active goal, so it can pass the normal executor safety checks during
    # that handoff.
    stop_command = _fresh_stop_command(snapshot)
    if stop_command is not None:
        return EffectDecision(
            "cmd_vel",
            {
                "linear_x_mps": 0.0,
                "angular_z_rps": 0.0,
                **_mission_metadata(
                    goal,
                    stage_complete=False,
                    reason=f"stopped by {stop_command.stream} command",
                ),
                "stop_reason": "command",
                "stop_event_id": str(stop_command.event_id),
            },
        )

    if stage == "lie_down":
        return EffectDecision("unitree_lie_down")

    # Defense in depth: do not wait for the planner cycle to cancel a fruit
    # mission when the perception producer has explicitly reported failure.
    # Search and approach both depend on YOLO; zero velocity is the only safe
    # controller output while that dependency is unhealthy.
    yolo_failure = _fresh_yolo_failure(snapshot)
    mission_type = str(goal.parameters.get("mission_type", "")).casefold()
    fruit_stage = mission_type in SUPPORTED_FRUITS and stage in {
        f"search_{mission_type}",
        f"approach_{mission_type}",
    }
    if fruit_stage and yolo_failure is not None:
        error = str(yolo_failure.payload.get("error", "YOLO perception failed"))
        return EffectDecision(
            "cmd_vel",
            {
                "linear_x_mps": 0.0,
                "angular_z_rps": 0.0,
                **_mission_metadata(
                    goal,
                    stage_complete=False,
                    reason=f"refusing motion because YOLO failed: {error}",
                ),
            },
        )

    mission = decide_mission_action(
        stage,
        snapshot,
        at=snapshot.captured_at,
        config=mission_config,
        target=mission_type,
    )
    # Semantic hardware effects deliberately accept tiny parameter vocabularies
    # at the safety boundary. Bark progress is its APPLIED status itself.
    if mission.effect_type == "unitree_bark":
        return EffectDecision("unitree_bark", {"sound": "bark"})
    return EffectDecision(
        mission.effect_type,
        {
            **mission.parameters,
            **_mission_metadata(
                goal,
                stage_complete=mission.stage_complete,
                reason=mission.reason,
            ),
        },
        list(mission.require_fresh_streams),
    )
