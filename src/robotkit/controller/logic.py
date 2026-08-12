"""Pure deterministic (mission Goal + world state) -> short-term Effect."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from robotkit.contracts import GoalRecord, WorldSnapshot
from robotkit.executor.mission import MissionConfig, decide_mission_action


@dataclass(frozen=True)
class EffectDecision:
    effect_type: str
    parameters: dict[str, Any] = field(default_factory=dict)
    require_fresh_streams: list[str] = field(default_factory=list)


def _mission_metadata(goal: GoalRecord, *, stage_complete: bool, reason: str) -> dict[str, Any]:
    return {
        "mission_schema_version": str(
            goal.parameters.get("mission_schema_version", "1")
        ),
        "mission_type": goal.parameters.get("mission_type"),
        "mission_stage": goal.parameters.get("mission_stage", goal.goal_type),
        "trigger_event_id": goal.parameters.get("trigger_event_id"),
        "stage_complete": stage_complete,
        "decision_reason": reason,
    }


def control(
    snapshot: WorldSnapshot,
    goal: GoalRecord,
    *,
    mission_config: MissionConfig = MissionConfig(),
) -> EffectDecision:
    """Translate a mission stage into one bounded, independently auditable action."""

    stage = str(goal.parameters.get("mission_stage", goal.goal_type))
    if stage == "lie_down":
        return EffectDecision("unitree_lie_down")

    mission = decide_mission_action(
        stage, snapshot, at=snapshot.captured_at, config=mission_config
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
