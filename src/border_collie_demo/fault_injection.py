"""Deterministic, simulation-only fault injection at trusted module seams."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class FaultAction(str, Enum):
    RAISE = "raise"
    RESPONSE_LOST = "response_lost"
    PROCESS_RESTART = "process_restart"
    DROP = "drop"
    STALE_PERCEPTION = "stale_perception"
    WEAK_PHANTOM = "weak_phantom"
    GENERATION_CHANGE = "generation_change"
    PARTIAL_WRITE = "partial_write"


class InjectedFault(RuntimeError):
    pass


class AmbiguousResponseInjected(InjectedFault):
    """The operation committed, but its response was lost."""


class ProcessRestartInjected(InjectedFault):
    pass


@dataclass(frozen=True)
class FaultSpec:
    point: str
    occurrence: int
    action: FaultAction
    parameters: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.point.strip() or self.occurrence < 1:
            raise ValueError("fault point and positive occurrence are required")


class DeterministicFaultInjector:
    """Apply an ordered fault plan only when explicitly in simulation mode."""

    def __init__(self, specs: list[FaultSpec], *, runtime_mode: str) -> None:
        if runtime_mode != "simulation":
            raise ValueError("fault injection is available only in simulation mode")
        identities = {(spec.point, spec.occurrence) for spec in specs}
        if len(identities) != len(specs):
            raise ValueError("only one fault may target a point occurrence")
        self._specs = {(spec.point, spec.occurrence): spec for spec in specs}
        self._counts: dict[str, int] = {}
        self._events: list[dict[str, object]] = []

    @property
    def events(self) -> list[dict[str, object]]:
        return deepcopy(self._events)

    def hit(self, point: str, value: Any = None) -> Any:
        occurrence = self._counts.get(point, 0) + 1
        self._counts[point] = occurrence
        spec = self._specs.get((point, occurrence))
        if spec is None:
            return value
        self._events.append(
            {
                "point": point,
                "occurrence": occurrence,
                "action": spec.action.value,
                "parameters": deepcopy(spec.parameters),
            }
        )
        message = str(spec.parameters.get("message") or f"injected {spec.action.value}")
        if spec.action is FaultAction.RAISE:
            raise InjectedFault(message)
        if spec.action is FaultAction.RESPONSE_LOST:
            raise AmbiguousResponseInjected(message)
        if spec.action is FaultAction.PROCESS_RESTART:
            raise ProcessRestartInjected(message)
        if spec.action is FaultAction.DROP:
            return None
        if spec.action is FaultAction.PARTIAL_WRITE:
            raw = bytes(value)
            keep = int(spec.parameters.get("bytes", max(1, len(raw) // 2)))
            return raw[: max(0, min(keep, len(raw)))]
        if not isinstance(value, dict):
            raise TypeError(f"{spec.action.value} requires dictionary evidence")
        transformed = deepcopy(value)
        if spec.action is FaultAction.STALE_PERCEPTION:
            source = transformed.setdefault("source", {})
            detection = transformed.setdefault("detection", {})
            if not isinstance(source, dict) or not isinstance(detection, dict):
                raise TypeError("perception evidence sections must be dictionaries")
            stale_at = float(spec.parameters.get("timestamp", 0.0))
            source["received_monotonic_s"] = stale_at
            detection["completed_monotonic_s"] = stale_at
            return transformed
        if spec.action is FaultAction.WEAK_PHANTOM:
            detection = transformed.setdefault("detection", {})
            if not isinstance(detection, dict):
                raise TypeError("detection evidence must be a dictionary")
            detection.update(
                {
                    "label": str(spec.parameters.get("label", "pear")),
                    "confidence": float(spec.parameters.get("confidence", 0.01)),
                    "consecutive_detections": 1,
                }
            )
            transformed["ready"] = False
            transformed["target_ready"] = False
            return transformed
        if spec.action is FaultAction.GENERATION_CHANGE:
            transformed["generation"] = str(
                spec.parameters.get("generation", "injected-generation")
            )
            return transformed
        raise AssertionError(f"unhandled fault action: {spec.action}")


class FaultInjectingCallable:
    """Wrap a synchronous seam with deterministic before/after checkpoints."""

    def __init__(
        self,
        wrapped: Callable[..., Any],
        injector: DeterministicFaultInjector,
        point: str,
    ) -> None:
        self._wrapped = wrapped
        self._injector = injector
        self._point = point

    def __call__(self, *args: object, **kwargs: object) -> Any:
        self._injector.hit(f"{self._point}.before")
        result = self._wrapped(*args, **kwargs)
        return self._injector.hit(f"{self._point}.after", result)
