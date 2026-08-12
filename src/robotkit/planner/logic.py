"""Pure deterministic world-state -> mission-stage planner.

Mission progress is encoded in durable Goal parameters and applied Effects. The
planner therefore has no process-local state and can be replaced between any
two invocations without changing its result.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping, Sequence

from robotkit.contracts import (
    EffectRecord,
    EffectStatus,
    GoalRecord,
    ObservationRecord,
    WorldSnapshot,
)


MISSION_SCHEMA_VERSION = "1"
APPLE_STAGES = ("search_apple", "approach_apple", "bark", "go_home", "lie_down")
HEALTH_STAGES = ("go_home", "lie_down")


@dataclass(frozen=True)
class GoalDecision:
    goal_type: str
    priority: int
    parameters: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""


def _fresh(snapshot: WorldSnapshot, at: datetime) -> dict[str, ObservationRecord]:
    selected: dict[str, ObservationRecord] = {}
    for observation in snapshot.observations:
        if observation.is_stale(at):
            continue
        current = selected.get(observation.stream)
        if current is None or (
            observation.observed_at,
            observation.revision,
            observation.confidence if observation.confidence is not None else -1.0,
        ) > (
            current.observed_at,
            current.revision,
            current.confidence if current.confidence is not None else -1.0,
        ):
            selected[observation.stream] = observation
    return selected


def _mission_parameters(
    mission_type: str,
    stage: str,
    trigger: ObservationRecord | None,
    *,
    trigger_event_id: str | None = None,
    trigger_stream: str | None = None,
) -> dict[str, Any]:
    stages = APPLE_STAGES if mission_type == "apple" else HEALTH_STAGES
    return {
        "mission_schema_version": MISSION_SCHEMA_VERSION,
        "mission_type": mission_type,
        "mission_stage": stage,
        "stage_index": stages.index(stage),
        "trigger_event_id": str(trigger.event_id) if trigger is not None else trigger_event_id,
        "trigger_stream": trigger.stream if trigger is not None else trigger_stream,
    }


def _decision(
    mission_type: str,
    stage: str,
    *,
    priority: int,
    trigger: ObservationRecord | None = None,
    current_goal: GoalRecord | None = None,
    rationale: str,
) -> GoalDecision:
    return GoalDecision(
        stage,
        priority,
        _mission_parameters(
            mission_type,
            stage,
            trigger,
            trigger_event_id=(
                str(current_goal.parameters.get("trigger_event_id"))
                if current_goal is not None
                and current_goal.parameters.get("trigger_event_id") is not None
                else None
            ),
            trigger_stream=(
                str(current_goal.parameters.get("trigger_stream"))
                if current_goal is not None
                and current_goal.parameters.get("trigger_stream") is not None
                else None
            ),
        ),
        rationale,
    )


def _current_mission(goal: GoalRecord | None) -> tuple[str, str] | None:
    if goal is None:
        return None
    mission_type = goal.parameters.get("mission_type")
    stage = goal.parameters.get("mission_stage", goal.goal_type)
    valid_stages = APPLE_STAGES if mission_type == "apple" else HEALTH_STAGES
    if mission_type not in {"apple", "health_return"} or stage not in valid_stages:
        return None
    return str(mission_type), str(stage)


def _offending_health(
    observations: Mapping[str, ObservationRecord],
) -> ObservationRecord | None:
    offenders: list[ObservationRecord] = []
    temperature = observations.get("health.temperature")
    if temperature and str(temperature.payload.get("band", "")).casefold() in {
        "high",
        "critical_high",
    }:
        offenders.append(temperature)
    battery = observations.get("health.battery")
    if battery and str(battery.payload.get("band", "")).casefold() == "critical_low":
        offenders.append(battery)
    return max(
        offenders,
        key=lambda item: (item.observed_at, item.revision, str(item.event_id)),
        default=None,
    )


def _apple_voice_intent(voice: ObservationRecord | None) -> bool:
    if voice is None:
        return False
    payload = voice.payload
    intent = str(payload.get("intent", payload.get("command", ""))).casefold()
    slots = payload.get("slots")
    slots = slots if isinstance(slots, Mapping) else {}
    target = " ".join(
        str(slots.get(key, payload.get(key, "")))
        for key in ("target", "destination", "object")
    ).casefold()
    if "apple" in target and intent in {"find", "locate", "search", "go", "move", "walk"}:
        return True
    transcript = str(
        payload.get("source_transcript", payload.get("transcript", payload.get("command", "")))
    ).casefold()
    return bool(
        re.search(r"\b(?:find|locate|search for|look for)\s+(?:the\s+)?apple\b", transcript)
        or re.search(r"\b(?:go|move|walk)\s+to\s+(?:the\s+)?apple\b", transcript)
    )


def _fresh_apple(vision: ObservationRecord | None) -> bool:
    if vision is None:
        return False
    detections = vision.payload.get("detections")
    if not isinstance(detections, Sequence) or isinstance(detections, (str, bytes)):
        return False
    return any(
        isinstance(item, Mapping)
        and str(item.get("class_name", item.get("label", ""))).casefold() == "apple"
        for item in detections
    )


def _applied_to_goal(effect: EffectRecord | None, goal: GoalRecord | None) -> bool:
    return bool(
        effect is not None
        and goal is not None
        and effect.goal_id == goal.goal_id
        and effect.status == EffectStatus.APPLIED
    )


def _effect_stage_complete(effect: EffectRecord | None) -> bool:
    if effect is None:
        return False
    if effect.parameters.get("stage_complete") is True:
        return True
    return bool(effect.result and effect.result.get("stage_complete") is True)


def _advance_stage(
    mission_type: str,
    stage: str,
    *,
    observations: Mapping[str, ObservationRecord],
    current_goal: GoalRecord,
    latest_effect: EffectRecord | None,
) -> str:
    if mission_type == "apple" and stage == "search_apple":
        return "approach_apple" if _fresh_apple(observations.get("vision.fruits")) else stage
    if not _applied_to_goal(latest_effect, current_goal):
        return stage
    if stage == "bark":
        return "go_home"
    if stage in {"approach_apple", "go_home"} and _effect_stage_complete(latest_effect):
        stages = APPLE_STAGES if mission_type == "apple" else HEALTH_STAGES
        return stages[stages.index(stage) + 1]
    return stage


def plan(
    snapshot: WorldSnapshot,
    at: datetime,
    *,
    current_goal: GoalRecord | None = None,
    latest_effect: EffectRecord | None = None,
) -> GoalDecision:
    """Select the next durable mission stage from explicit durable inputs only."""

    observations = _fresh(snapshot, at)
    current = _current_mission(current_goal)
    health = _offending_health(observations)

    # Once a health-return mission starts, retain its original event correlation
    # while newer health samples arrive. This prevents repeated go_home resets.
    if current is not None and current[0] == "health_return":
        stage = _advance_stage(
            *current,
            observations=observations,
            current_goal=current_goal,  # type: ignore[arg-type]
            latest_effect=latest_effect,
        )
        return _decision(
            "health_return",
            stage,
            priority=100,
            current_goal=current_goal,
            rationale="complete safety return and lie-down sequence",
        )

    # A high/critical-high temperature or critical-low battery always preempts
    # an ordinary mission. The selected event ID is durable incident identity.
    if health is not None:
        return _decision(
            "health_return",
            "go_home",
            priority=100,
            trigger=health,
            rationale=f"safety preemption from {health.stream}",
        )

    voice = observations.get("voice.intent")
    voice_requests_apple = _apple_voice_intent(voice)
    current_trigger = (
        str(current_goal.parameters.get("trigger_event_id"))
        if current_goal is not None and current_goal.parameters.get("trigger_event_id")
        else None
    )

    if current is not None and current[0] == "apple":
        # A distinct command starts a distinct mission. The same observation is
        # ignored at the terminal stage, so its TTL cannot retrigger forever.
        if (
            voice_requests_apple
            and voice is not None
            and str(voice.event_id) != current_trigger
        ):
            return _decision(
                "apple",
                "search_apple",
                priority=60,
                trigger=voice,
                rationale="new voice command requested an apple mission",
            )
        stage = _advance_stage(
            *current,
            observations=observations,
            current_goal=current_goal,  # type: ignore[arg-type]
            latest_effect=latest_effect,
        )
        return _decision(
            "apple",
            stage,
            priority=60,
            current_goal=current_goal,
            rationale="continue durable apple mission",
        )

    if voice_requests_apple and voice is not None:
        return _decision(
            "apple",
            "search_apple",
            priority=60,
            trigger=voice,
            rationale="voice command requested an apple mission",
        )

    return GoalDecision(
        "idle_at_home",
        0,
        {
            "mission_schema_version": MISSION_SCHEMA_VERSION,
            "mission_type": "idle",
            "mission_stage": "idle_at_home",
            "stage_index": 0,
            "trigger_event_id": None,
            "trigger_stream": None,
        },
        "no active mission",
    )
