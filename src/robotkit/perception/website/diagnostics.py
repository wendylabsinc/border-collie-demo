"""Pure reduction of durable A state into an operator-facing pipeline view."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping, Sequence

from robotkit.contracts import (
    EffectRecord,
    EffectStatus,
    EventRecord,
    GoalRecord,
    ObservationRecord,
    WorldSnapshot,
)
from robotkit.fruits import SUPPORTED_FRUITS


_SEVERITY = {"ok": 0, "waiting": 1, "warning": 2, "stale": 3, "error": 4}
_COMMAND_STREAMS = ("voice.intent", "website.intent")


def _latest(
    observations: Sequence[ObservationRecord], *streams: str
) -> ObservationRecord | None:
    candidates = [item for item in observations if item.stream in streams]
    return max(
        candidates,
        key=lambda item: (item.observed_at, item.revision, str(item.event_id)),
        default=None,
    )


def _age_seconds(observation: ObservationRecord | None, at: datetime) -> float | None:
    if observation is None:
        return None
    return round(max(0.0, (at - observation.observed_at).total_seconds()), 3)


def _requested_target(command: ObservationRecord | None) -> str | None:
    if command is None:
        return None
    slots = command.payload.get("slots")
    if isinstance(slots, Mapping):
        target = str(slots.get("target", "")).strip().casefold()
        return target or None
    return None


def _detected_labels(vision: ObservationRecord | None) -> list[str]:
    if vision is None:
        return []
    detections = vision.payload.get("detections")
    if not isinstance(detections, Sequence) or isinstance(detections, (str, bytes)):
        return []
    return sorted(
        {
            str(item.get("class_name", item.get("label", ""))).strip().casefold()
            for item in detections
            if isinstance(item, Mapping)
            and str(item.get("class_name", item.get("label", ""))).strip()
        }
    )


def _stage(
    name: str,
    status: str,
    summary: str,
    *,
    age_seconds: float | None = None,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "summary": summary,
        "age_seconds": age_seconds,
        "details": dict(details or {}),
    }


def _event_summary(event: EventRecord) -> dict[str, Any]:
    data = event.data
    summary: dict[str, Any] = {}
    if event.category == "observation.published":
        summary = {
            "stream": data.get("stream"),
            "producer_id": data.get("producer_id"),
            "observed_at": data.get("observed_at"),
        }
        payload = data.get("payload")
        if isinstance(payload, Mapping):
            for key in ("status", "intent", "slots", "count", "model"):
                if key in payload:
                    summary[key] = payload[key]
    elif event.category.startswith("goal"):
        summary = {
            "goal_type": data.get("goal_type"),
            "status": data.get("status"),
            "rationale": data.get("rationale"),
        }
    elif event.category.startswith("effect"):
        summary = {
            "effect_type": data.get("effect_type"),
            "status": data.get("status"),
            "result": data.get("result"),
        }
    return {
        "revision": event.revision,
        "category": event.category,
        "entity_id": event.entity_id,
        "recorded_at": event.recorded_at.isoformat(),
        "summary": summary,
    }


def build_debug_snapshot(
    snapshot: WorldSnapshot,
    *,
    current_goal: GoalRecord | None,
    latest_effect: EffectRecord | None,
    events: Sequence[EventRecord] = (),
    at: datetime | None = None,
    supported_targets: Sequence[str] = tuple(sorted(SUPPORTED_FRUITS)),
) -> dict[str, Any]:
    """Build a bounded debug projection without retaining local state."""

    at = at or snapshot.captured_at
    supported = sorted({value.strip().casefold() for value in supported_targets if value.strip()})
    command = _latest(snapshot.observations, *_COMMAND_STREAMS)
    vision = _latest(snapshot.observations, "vision.fruits")
    yolo = _latest(snapshot.observations, "diagnostics.yolo")
    target = _requested_target(command)
    labels = _detected_labels(vision)

    if command is None:
        command_stage = _stage("command", "waiting", "No voice or website command in A")
    elif command.is_stale(at):
        command_stage = _stage(
            "command",
            "stale",
            f"Latest {command.stream} command has expired",
            age_seconds=_age_seconds(command, at),
            details={"target": target, "event_id": str(command.event_id)},
        )
    else:
        command_stage = _stage(
            "command",
            "ok",
            f"{command.stream} requested {target or 'an unspecified target'}",
            age_seconds=_age_seconds(command, at),
            details={
                "target": target,
                "intent": command.payload.get("intent"),
                "event_id": str(command.event_id),
                "revision": command.revision,
            },
        )

    unsupported = bool(target and target not in supported)
    diagnostic_status = str(yolo.payload.get("status", "")) if yolo else ""
    if unsupported:
        perception_stage = _stage(
            "perception",
            "error",
            f"Unsupported target {target!r}; configured targets: {', '.join(supported)}",
            age_seconds=_age_seconds(yolo or vision, at),
            details={"requested_target": target, "supported_targets": supported},
        )
    elif yolo is not None and diagnostic_status == "failed":
        perception_stage = _stage(
            "perception",
            "error",
            str(yolo.payload.get("error", "YOLO inference failed")),
            age_seconds=_age_seconds(yolo, at),
            details=yolo.payload,
        )
    elif vision is None:
        perception_stage = _stage(
            "perception",
            "error",
            "No vision.fruits observation has reached A",
            age_seconds=_age_seconds(yolo, at),
            details=yolo.payload if yolo else {},
        )
    elif vision.is_stale(at):
        perception_stage = _stage(
            "perception",
            "stale",
            "The latest vision.fruits observation is stale",
            age_seconds=_age_seconds(vision, at),
            details={"labels": labels, "revision": vision.revision},
        )
    elif target and target in labels:
        perception_stage = _stage(
            "perception",
            "ok",
            f"Fresh YOLO frame contains {target}",
            age_seconds=_age_seconds(vision, at),
            details={"labels": labels, "count": vision.payload.get("count")},
        )
    else:
        perception_stage = _stage(
            "perception",
            "waiting",
            f"Fresh YOLO frames do not contain {target or 'the target'}",
            age_seconds=_age_seconds(vision, at),
            details={"labels": labels, "count": vision.payload.get("count")},
        )

    if unsupported:
        planner_stage = _stage(
            "planner",
            "error",
            f"Planner rejected unsupported target {target!r}",
            details={"supported_targets": supported},
        )
    elif current_goal is None:
        planner_stage = _stage("planner", "waiting", "A has no current goal")
    else:
        rejected = current_goal.parameters.get("rejected_target")
        planner_stage = _stage(
            "planner",
            "error" if rejected else "ok",
            current_goal.rationale or current_goal.goal_type,
            details={
                "goal_id": str(current_goal.goal_id),
                "goal_type": current_goal.goal_type,
                "mission_stage": current_goal.parameters.get("mission_stage"),
                "status": current_goal.status,
                "revision": current_goal.revision,
                "rejected_target": rejected,
            },
        )

    if current_goal is None or latest_effect is None:
        controller_stage = _stage(
            "controller", "waiting", "No effect exists for the current goal"
        )
        executor_stage = _stage("executor", "waiting", "No effect to execute")
    else:
        effect_details = {
            "effect_id": str(latest_effect.effect_id),
            "effect_type": latest_effect.effect_type,
            "status": latest_effect.status.value,
            "goal_id": str(latest_effect.goal_id),
            "revision": latest_effect.revision,
            "decision_reason": latest_effect.parameters.get("decision_reason"),
        }
        controller_stage = _stage(
            "controller",
            "ok" if latest_effect.goal_id == current_goal.goal_id else "warning",
            str(
                latest_effect.parameters.get(
                    "decision_reason", f"Produced {latest_effect.effect_type}"
                )
            ),
            details=effect_details,
        )
        if latest_effect.status == EffectStatus.APPLIED:
            effect_status = "ok"
            summary = f"Applied {latest_effect.effect_type}"
        elif latest_effect.status in {EffectStatus.REJECTED, EffectStatus.FAILED}:
            effect_status = "error"
            summary = f"{latest_effect.effect_type} {latest_effect.status.value}"
        elif latest_effect.status == EffectStatus.EXPIRED:
            effect_status = "stale"
            summary = f"{latest_effect.effect_type} expired"
        else:
            effect_status = "waiting"
            summary = f"{latest_effect.effect_type} is {latest_effect.status.value}"
        executor_stage = _stage(
            "executor",
            effect_status,
            summary,
            details={**effect_details, "result": latest_effect.result},
        )

    stages = [
        command_stage,
        perception_stage,
        planner_stage,
        controller_stage,
        executor_stage,
    ]
    overall = max(stages, key=lambda item: _SEVERITY[item["status"]])["status"]
    observations = [
        {
            "stream": item.stream,
            "producer_id": item.producer_id,
            "revision": item.revision,
            "observed_at": item.observed_at.isoformat(),
            "age_seconds": _age_seconds(item, at),
            "ttl_seconds": item.ttl_seconds,
            "stale": item.is_stale(at),
            "frame_id": item.frame_id,
        }
        for item in sorted(snapshot.observations, key=lambda value: value.stream)
    ]
    return {
        "captured_at": at.isoformat(),
        "revision": snapshot.revision,
        "state_revision": snapshot.state_revision,
        "overall_status": overall,
        "requested_target": target,
        "supported_targets": supported,
        "stages": stages,
        "observations": observations,
        "timeline": [_event_summary(event) for event in events[-50:]][::-1],
    }
