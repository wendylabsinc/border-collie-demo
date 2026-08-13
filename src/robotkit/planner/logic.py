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
from robotkit.fruits import SUPPORTED_FRUITS, normalize_fruit


MISSION_SCHEMA_VERSION = "1"
HEALTH_STAGES = ("go_home", "lie_down")


def _fruit_stages(target: str) -> tuple[str, ...]:
    return (f"search_{target}", f"approach_{target}", "bark", "go_home", "lie_down")


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
    stages = _fruit_stages(mission_type) if mission_type in SUPPORTED_FRUITS else HEALTH_STAGES
    parameters = {
        "mission_schema_version": MISSION_SCHEMA_VERSION,
        "mission_type": mission_type,
        "mission_stage": stage,
        "stage_index": stages.index(stage),
        "trigger_event_id": str(trigger.event_id) if trigger is not None else trigger_event_id,
        "trigger_stream": trigger.stream if trigger is not None else trigger_stream,
    }
    # Preserve the original apple mission contract while making the target
    # explicit for newly supported fruit missions.
    if mission_type in SUPPORTED_FRUITS and mission_type != "apple":
        parameters["target"] = mission_type
    return parameters


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
    valid_stages = (
        _fruit_stages(str(mission_type))
        if mission_type in SUPPORTED_FRUITS
        else HEALTH_STAGES
    )
    if mission_type not in {*SUPPORTED_FRUITS, "health_return"} or stage not in valid_stages:
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


def _command_target(command: ObservationRecord | None) -> str | None:
    if command is None:
        return None
    payload = command.payload
    intent = str(payload.get("intent", payload.get("command", ""))).casefold()
    slots = payload.get("slots")
    slots = slots if isinstance(slots, Mapping) else {}
    if intent in {"find", "locate", "search", "go", "move", "walk"}:
        for key in ("target", "destination", "object"):
            target = str(slots.get(key, payload.get(key, ""))).strip().casefold()
            if target:
                return target
    transcript = str(
        payload.get("source_transcript", payload.get("transcript", payload.get("command", "")))
    ).casefold()
    match = re.search(
        r"\b(?:find|locate|search for|look for)\s+(?:the\s+)?([\w-]+)",
        transcript,
    )
    match = match or re.search(
        r"\b(?:go|move|walk)\s+to\s+(?:the\s+)?([\w-]+)", transcript
    )
    return match.group(1).casefold() if match else None


def _fruit_command_target(command: ObservationRecord | None) -> str | None:
    return normalize_fruit(_command_target(command) or "")


def _command_intent(command: ObservationRecord | None) -> str:
    if command is None:
        return ""
    return str(
        command.payload.get("intent", command.payload.get("command", ""))
    ).strip().casefold()


def _stop_command_decision(command: ObservationRecord) -> GoalDecision:
    slots = command.payload.get("slots")
    slots = slots if isinstance(slots, Mapping) else {}
    return GoalDecision(
        "idle_at_home",
        100 if slots.get("emergency") is True else 90,
        {
            "mission_schema_version": MISSION_SCHEMA_VERSION,
            "mission_type": "idle",
            "mission_stage": "idle_at_home",
            "stage_index": 0,
            "trigger_event_id": str(command.event_id),
            "trigger_stream": command.stream,
            "stop_reason": "command",
            "emergency": slots.get("emergency") is True,
        },
        f"{command.stream} stop command cancelled active motion",
    )


def _failed_yolo(
    observations: Mapping[str, ObservationRecord],
) -> ObservationRecord | None:
    diagnostic = observations.get("diagnostics.yolo")
    if diagnostic is None:
        return None
    return (
        diagnostic
        if str(diagnostic.payload.get("status", "")).strip().casefold() == "failed"
        else None
    )


def _perception_failure_decision(
    diagnostic: ObservationRecord, target: str
) -> GoalDecision:
    return GoalDecision(
        "idle_at_home",
        80,
        {
            "mission_schema_version": MISSION_SCHEMA_VERSION,
            "mission_type": "idle",
            "mission_stage": "idle_at_home",
            "stage_index": 0,
            "trigger_event_id": str(diagnostic.event_id),
            "trigger_stream": diagnostic.stream,
            "stop_reason": "perception_failure",
            "failed_stream": diagnostic.stream,
        },
        f"{target} mission stopped because YOLO perception failed",
    )


def _unsupported_target_decision(
    command: ObservationRecord, target: str
) -> GoalDecision:
    return GoalDecision(
        "idle_at_home",
        0,
        {
            "mission_schema_version": MISSION_SCHEMA_VERSION,
            "mission_type": "idle",
            "mission_stage": "idle_at_home",
            "stage_index": 0,
            "trigger_event_id": str(command.event_id),
            "trigger_stream": command.stream,
            "rejected_target": target,
            "supported_targets": sorted(SUPPORTED_FRUITS),
        },
        (
            f"unsupported target {target!r}; supported mission targets: "
            f"{', '.join(sorted(SUPPORTED_FRUITS))}"
        ),
    )


def _latest_command(
    observations: Mapping[str, ObservationRecord],
) -> ObservationRecord | None:
    commands = [
        observation
        for stream in ("voice.intent", "website.intent")
        if (observation := observations.get(stream)) is not None
    ]
    return max(
        commands,
        key=lambda item: (item.observed_at, item.revision, str(item.event_id)),
        default=None,
    )


def _fresh_fruit(vision: ObservationRecord | None, target: str) -> bool:
    if vision is None:
        return False
    detections = vision.payload.get("detections")
    if not isinstance(detections, Sequence) or isinstance(detections, (str, bytes)):
        return False
    return any(
        isinstance(item, Mapping)
        and str(item.get("class_name", item.get("label", ""))).casefold() == target
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
    if mission_type in SUPPORTED_FRUITS and stage == f"search_{mission_type}":
        return (
            f"approach_{mission_type}"
            if _fresh_fruit(observations.get("vision.fruits"), mission_type)
            else stage
        )
    if not _applied_to_goal(latest_effect, current_goal):
        return stage
    if stage == "bark":
        return "go_home"
    if (
        stage == f"approach_{mission_type}" or stage == "go_home"
    ) and _effect_stage_complete(latest_effect):
        stages = (
            _fruit_stages(mission_type)
            if mission_type in SUPPORTED_FRUITS
            else HEALTH_STAGES
        )
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
    command = _latest_command(observations)
    command_intent = _command_intent(command)

    # A fresh operator stop is the highest-level motion interlock. It cancels
    # ordinary missions as well as an in-progress health return; once the
    # short-lived command expires, unresolved health state can initiate a new
    # return mission.
    if command is not None and command_intent == "stop":
        return _stop_command_decision(command)

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

    command_target = _command_target(command)
    requested_fruit = _fruit_command_target(command)
    failed_yolo = _failed_yolo(observations)
    unsupported_target = (
        command_target
        if command is not None and command_target is not None and requested_fruit is None
        else None
    )
    current_trigger = (
        str(current_goal.parameters.get("trigger_event_id"))
        if current_goal is not None and current_goal.parameters.get("trigger_event_id")
        else None
    )

    if current is not None and current[0] in SUPPORTED_FRUITS:
        current_target = current[0]
        # A distinct command starts a distinct mission. The same observation is
        # ignored at the terminal stage, so its TTL cannot retrigger forever.
        if command is not None and str(command.event_id) != current_trigger:
            if unsupported_target is not None:
                return _unsupported_target_decision(command, unsupported_target)
        if failed_yolo is not None:
            return _perception_failure_decision(failed_yolo, current_target)
        if requested_fruit and command is not None and str(command.event_id) != current_trigger:
            return _decision(
                requested_fruit,
                f"search_{requested_fruit}",
                priority=60,
                trigger=command,
                rationale=(
                    f"new {command.stream} command requested a {requested_fruit} mission"
                ),
            )
        stage = _advance_stage(
            *current,
            observations=observations,
            current_goal=current_goal,  # type: ignore[arg-type]
            latest_effect=latest_effect,
        )
        return _decision(
            current_target,
            stage,
            priority=60,
            current_goal=current_goal,
            rationale=f"continue durable {current_target} mission",
        )

    if requested_fruit and command is not None:
        if failed_yolo is not None:
            return _perception_failure_decision(failed_yolo, requested_fruit)
        return _decision(
            requested_fruit,
            f"search_{requested_fruit}",
            priority=60,
            trigger=command,
            rationale=f"{command.stream} command requested a {requested_fruit} mission",
        )

    if command is not None and unsupported_target is not None:
        return _unsupported_target_decision(command, unsupported_target)

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
