"""Compact operator-facing projections of durable Demo Run events."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Protocol


class OperatorEventSink(Protocol):
    """One-method seam for observing an already-durable black-box event."""

    def emit(self, run_id: str, event: dict[str, Any]) -> None: ...


class _InfoLogger(Protocol):
    def info(self, message: str, *args: object) -> None: ...


class OperatorEventLogger:
    """Turn high-value black-box events into compact JSON INFO lines."""

    def __init__(
        self,
        *,
        logger: _InfoLogger | None = None,
        enabled: bool = True,
    ) -> None:
        # Reuse Uvicorn's configured INFO logger so Wendy captures these lines
        # without adding a second handler or changing global logging policy.
        self._logger = logger or logging.getLogger("uvicorn.error")
        self._enabled = enabled

    @classmethod
    def from_env(cls) -> OperatorEventLogger:
        raw = os.environ.get("BORDER_COLLIE_OPERATOR_LOG_ENABLED", "1").strip()
        if raw not in {"0", "1"}:
            raise ValueError("BORDER_COLLIE_OPERATOR_LOG_ENABLED must be 0 or 1")
        return cls(enabled=raw == "1")

    def emit(self, run_id: str, event: dict[str, Any]) -> None:
        if not self._enabled:
            return
        projected = _project(run_id, event)
        if projected is None:
            return
        try:
            self._logger.info(
                "demo_event %s",
                json.dumps(projected, separators=(",", ":"), sort_keys=False),
            )
        except Exception:  # noqa: BLE001 - observability never affects safety
            return


def _project(run_id: str, event: dict[str, Any]) -> dict[str, object] | None:
    kind = str(event.get("kind") or "")
    payload = event.get("payload")
    if not isinstance(payload, dict):
        payload = {}
    base: dict[str, object] = {
        "run_id": run_id,
        "sequence": event.get("sequence"),
    }
    phase = event.get("phase")

    if kind == "run_started":
        return {
            **base,
            "event": "run_started",
            "phase": phase,
            **_take(payload, "target_fruit", "activation_source", "activation_id"),
        }
    if kind == "mission_event":
        return {
            **base,
            "event": "state",
            "phase": phase,
            **_take(payload, "reason", "message"),
        }
    if kind == "motion_command":
        return {
            **base,
            "event": "motion",
            "phase": phase,
            "command_sequence": payload.get("sequence"),
            **_take(
                payload,
                "motion_path",
                "forward_mps",
                "yaw_rps",
                "reason",
            ),
        }
    if kind == "stage_result":
        return {
            **base,
            "event": "stage_result",
            "phase": phase,
            **_take(
                payload,
                "label",
                "guidance_phase",
                "guidance_reason",
                "arrival_confirmed",
                "home_distance_m",
                "motion_path",
            ),
        }
    if kind == "home_position_retry":
        return {
            **base,
            "event": "home_retry",
            "phase": phase,
            **_take(
                payload,
                "reason",
                "retry_index",
                "home_distance_m",
                "position_spread_m",
            ),
        }
    if kind == "home_verification_terminal":
        return {
            **base,
            "event": "home_verification",
            "phase": phase,
            **_take(
                payload,
                "stable",
                "reason",
                "distance_min_m",
                "distance_max_m",
                "position_spread_m",
                "terminal_reason",
            ),
        }
    if kind == "failure_epilogue":
        return {
            **base,
            "event": "failure_epilogue",
            "phase": phase,
            **_take(payload, "status", "reason", "attempted_return"),
        }
    if kind == "run_sealed":
        return {
            **base,
            "event": "terminal",
            "phase": phase,
            **_take(
                payload,
                "outcome",
                "reason",
                "failed_phase",
                "final_safety_state",
            ),
        }
    return None


def _take(payload: dict[str, Any], *keys: str) -> dict[str, object]:
    return {key: payload[key] for key in keys if key in payload}
