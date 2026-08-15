"""Immutable per-activation tuning behind one small interface.

``RunTuning`` is the only interface used by HTTP, voice, soak, persistence, and
stage execution.  It hides default resolution, target-specific confidence
policy, schema generation, conversion, and cross-field validation.  Safety
ownership, watchdogs, exact-zero disarm, camera generation continuity, and the
absolute hardware envelopes deliberately remain outside this module.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import ClassVar

from .config import env_bool
from .fruits import fruit_policy


def _number(
    values: Mapping[str, object], name: str, default: float, low: float, high: float
) -> float:
    raw = values.get(name, default)
    if isinstance(raw, bool):
        raise ValueError(f"{name} must be a number")  # noqa: TRY004 - public validation
    try:
        result = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if not low <= result <= high:
        raise ValueError(f"{name} must stay within {low}..{high}")
    return result


def _integer(
    values: Mapping[str, object], name: str, default: int, low: int, high: int
) -> int:
    raw = values.get(name, default)
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError(f"{name} must be an integer")  # noqa: TRY004 - public validation
    if not low <= raw <= high:
        raise ValueError(f"{name} must stay within {low}..{high}")
    return raw


def _boolean(values: Mapping[str, object], name: str, default: bool) -> bool:
    raw = values.get(name, default)
    if not isinstance(raw, bool):
        # Public activation validation consistently reports ValueError as HTTP
        # conflict; a TypeError would bypass that one validation seam.
        raise ValueError(f"{name} must be a boolean")  # noqa: TRY004
    return raw


def _optional_number(
    values: Mapping[str, object],
    name: str,
    default: float | None,
    low: float,
    high: float,
) -> float | None:
    if name not in values:
        return default
    if values[name] is None:
        return None
    return _number(values, name, 0.0, low, high)


def _group(
    payload: Mapping[str, object], name: str, allowed: set[str]
) -> Mapping[str, object]:
    raw = payload.get(name, {})
    if not isinstance(raw, Mapping):
        raise ValueError(f"{name} tuning must be an object")  # noqa: TRY004
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown {name} tuning: {', '.join(sorted(unknown))}")
    return raw


@dataclass(frozen=True)
class SearchTuning:
    yaw_rps: float = 0.40
    sweep_rad: float = 2.0 * math.pi
    timeout_s: float = 45.0
    focus_yaw_rps: float = 0.40
    progressive_focus_yaw_enabled: bool = True
    focus_yaw_step_rps: float = 0.10
    focus_minimum_yaw_rps: float = 0.40
    focus_near_center_ratio: float = 0.20
    focus_missing_grace_s: float = 0.50
    bearing_routing_enabled: bool = False


@dataclass(frozen=True)
class RecognitionTuning:
    focus_confidence: float | None
    lock_confidence: float
    tracking_confidence: float
    required_frames: int = 3


@dataclass(frozen=True)
class CenteringTuning:
    lock_tolerance_ratio: float = 0.05
    outer_corridor_ratio: float = 0.20
    approach_yaw_rps: float = 0.30
    recenter_yaw_rps: float = 0.50


@dataclass(frozen=True)
class ApproachTuning:
    forward_mps: float = 1.0
    timeout_s: float = 20.0
    duplicate_hold_s: float = 0.250
    source_maximum_age_s: float = 0.350
    detection_maximum_age_s: float = 0.250
    slow_inference_grace_s: float = 0.50


@dataclass(frozen=True)
class ArrivalTuning:
    near_bottom_ratio: float = 0.90
    disappearance_bottom_ratio: float = 0.80
    near_center_ratio: float = 0.72
    near_confirmations: int = 3
    loss_confirmations: int = 2
    loss_grace_s: float = 0.75
    final_push_mps: float = 0.60
    final_push_duration_s: float = 1.0


@dataclass(frozen=True)
class HomeTuning:
    align_yaw_rps: float = 0.80
    align_tolerance_deg: float = 5.0
    align_timeout_s: float = 30.0
    return_forward_mps: float = 1.0
    arrival_tolerance_m: float = 0.50
    heading_gate_deg: float = 20.0
    return_yaw_deadband_deg: float = 5.0
    return_minimum_yaw_rps: float = 0.50
    return_yaw_rps: float = 0.50
    minimum_progress_m: float = 0.03
    stall_timeout_s: float = 5.0
    return_timeout_s: float = 30.0
    settle_interval_s: float = 0.30
    settled_sample_count: int = 4
    settled_maximum_spread_m: float = 0.03
    settled_sample_timeout_s: float = 1.0
    settled_retry_count: int = 1

    @classmethod
    def from_env(cls) -> HomeTuning:
        values: dict[str, object] = {
            "align_yaw_rps": os.environ.get(
                "BORDER_COLLIE_HOME_ALIGN_YAW_RPS", "0.80"
            ),
            "arrival_tolerance_m": os.environ.get(
                "BORDER_COLLIE_HOME_ARRIVAL_TOLERANCE_M", "0.50"
            ),
            "stall_timeout_s": os.environ.get(
                "BORDER_COLLIE_HOME_STALL_TIMEOUT_S", "5.0"
            ),
            "settle_interval_s": os.environ.get(
                "BORDER_COLLIE_HOME_SETTLE_INTERVAL_S", "0.30"
            ),
            "settled_sample_count": int(
                os.environ.get("BORDER_COLLIE_HOME_SETTLED_SAMPLE_COUNT", "4")
            ),
            "settled_maximum_spread_m": os.environ.get(
                "BORDER_COLLIE_HOME_SETTLED_MAX_SPREAD_M", "0.03"
            ),
            "settled_sample_timeout_s": os.environ.get(
                "BORDER_COLLIE_HOME_SETTLED_SAMPLE_TIMEOUT_S", "1.0"
            ),
            "settled_retry_count": int(
                os.environ.get("BORDER_COLLIE_HOME_SETTLED_RETRY_COUNT", "1")
            ),
        }
        return cls(
            align_yaw_rps=_number(values, "align_yaw_rps", 0.80, 0.50, 0.80),
            arrival_tolerance_m=_number(
                values, "arrival_tolerance_m", 0.50, 0.05, 0.50
            ),
            stall_timeout_s=_number(
                values, "stall_timeout_s", 5.0, 0.5, 5.0
            ),
            settle_interval_s=_number(
                values, "settle_interval_s", 0.30, 0.0, 2.0
            ),
            settled_sample_count=_integer(
                values, "settled_sample_count", 4, 3, 5
            ),
            settled_maximum_spread_m=_number(
                values, "settled_maximum_spread_m", 0.03, 0.005, 0.05
            ),
            settled_sample_timeout_s=_number(
                values, "settled_sample_timeout_s", 1.0, 0.25, 3.0
            ),
            settled_retry_count=_integer(
                values, "settled_retry_count", 1, 0, 1
            ),
        )


_FRUIT_CONFIDENCE_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "apple": {"focus": (0.40, 0.70), "lock": (0.40, 0.70), "tracking": (0.10, 0.70)},
    "banana": {"focus": (0.20, 0.70), "lock": (0.20, 0.70), "tracking": (0.20, 0.70)},
    "pear": {"focus": (0.65, 0.85), "lock": (0.65, 0.85), "tracking": (0.55, 0.85)},
}


@dataclass(frozen=True)
class RunTuning:
    """One validated, target-bound policy snapshot for exactly one run."""

    target_fruit: str
    search: SearchTuning
    recognition: RecognitionTuning
    centering: CenteringTuning
    approach: ApproachTuning
    arrival: ArrivalTuning
    home: HomeTuning

    GROUPS: ClassVar[tuple[str, ...]] = (
        "search",
        "recognition",
        "centering",
        "approach",
        "arrival",
        "home",
    )

    @classmethod
    def defaults(cls, target_fruit: str) -> RunTuning:
        target = target_fruit.casefold().strip()
        policy = fruit_policy(target)
        # Environment remains a deployment-default adapter only. Once this
        # snapshot exists, no later environment change can affect the run.
        from .guidance import GuidanceConfig

        guidance = GuidanceConfig.from_env()
        return cls(
            target_fruit=target,
            search=SearchTuning(
                yaw_rps=guidance.search_yaw_rps,
                sweep_rad=guidance.search_sweep_rad,
                timeout_s=_number(
                    {
                        "timeout_s": os.environ.get(
                            "BORDER_COLLIE_GUIDANCE_SEARCH_TIMEOUT_S", "45"
                        )
                    },
                    "timeout_s",
                    45.0,
                    5.0,
                    60.0,
                ),
                focus_yaw_rps=guidance.focus_yaw_rps,
                progressive_focus_yaw_enabled=(
                    guidance.progressive_focus_yaw_enabled
                ),
                focus_yaw_step_rps=guidance.focus_yaw_step_rps,
                focus_minimum_yaw_rps=guidance.focus_minimum_yaw_rps,
                focus_near_center_ratio=guidance.focus_near_center_ratio,
                focus_missing_grace_s=guidance.focus_missing_grace_s,
                bearing_routing_enabled=env_bool(
                    "BORDER_COLLIE_BEARING_ROUTING_ENABLED", False
                ),
            ),
            recognition=RecognitionTuning(
                focus_confidence=policy.focus_confidence,
                lock_confidence=policy.acquisition_confidence,
                tracking_confidence=policy.close_range_tracking_confidence,
                required_frames=guidance.center_confirmations,
            ),
            centering=CenteringTuning(
                lock_tolerance_ratio=guidance.center_tolerance_ratio,
                outer_corridor_ratio=guidance.outer_corridor_ratio,
                approach_yaw_rps=guidance.approach_yaw_rps,
                recenter_yaw_rps=guidance.recenter_yaw_rps,
            ),
            approach=ApproachTuning(
                forward_mps=guidance.approach_forward_mps,
                duplicate_hold_s=guidance.duplicate_hold_s,
                source_maximum_age_s=guidance.source_maximum_age_s,
                detection_maximum_age_s=guidance.detection_maximum_age_s,
                slow_inference_grace_s=guidance.slow_inference_grace_s,
            ),
            arrival=ArrivalTuning(
                near_bottom_ratio=guidance.near_bottom_ratio,
                disappearance_bottom_ratio=guidance.disappearance_bottom_ratio,
                near_center_ratio=guidance.near_center_ratio,
                near_confirmations=guidance.near_confirmations,
                loss_confirmations=guidance.near_loss_confirmations,
                loss_grace_s=guidance.near_loss_grace_s,
                final_push_mps=guidance.final_push_mps,
                final_push_duration_s=guidance.final_push_duration_s,
            ),
            home=HomeTuning.from_env(),
        )

    @classmethod
    def from_payload(
        cls, target_fruit: str, payload: Mapping[str, object] | None
    ) -> RunTuning:
        target = target_fruit.casefold().strip()
        defaults = cls.defaults(target)
        raw: Mapping[str, object] = payload or {}
        if not isinstance(raw, Mapping):
            raise ValueError("run tuning must be an object")  # noqa: TRY004
        legacy_keys = {
            "search_yaw_rps",
            "focus_confidence",
            "lock_confidence",
            "center_confirmations",
            "center_tolerance_ratio",
        }
        if raw and set(raw) <= legacy_keys:
            raw = {
                "search": {
                    "yaw_rps": raw.get(
                        "search_yaw_rps", cls.defaults(target).search.yaw_rps
                    )
                },
                "recognition": {
                    key: value
                    for key, value in {
                        "focus_confidence": raw.get("focus_confidence"),
                        "lock_confidence": raw.get("lock_confidence"),
                        "required_frames": raw.get("center_confirmations"),
                    }.items()
                    if value is not None
                },
                "centering": {
                    "lock_tolerance_ratio": raw.get(
                        "center_tolerance_ratio",
                        cls.defaults(target).centering.lock_tolerance_ratio,
                    )
                },
            }
        unknown = set(raw) - ({"target_fruit"} | set(cls.GROUPS))
        if unknown:
            raise ValueError(f"unknown run tuning: {', '.join(sorted(unknown))}")
        payload_target = raw.get("target_fruit")
        if payload_target is not None and (
            not isinstance(payload_target, str)
            or payload_target.casefold().strip() != target
        ):
            raise ValueError("run tuning Target Fruit does not match the Demo Run")

        search = _group(
            raw,
            "search",
            {
                "yaw_rps",
                "sweep_rad",
                "timeout_s",
                "focus_yaw_rps",
                "progressive_focus_yaw_enabled",
                "focus_yaw_step_rps",
                "focus_minimum_yaw_rps",
                "focus_near_center_ratio",
                "focus_missing_grace_s",
                "bearing_routing_enabled",
            },
        )
        recognition = _group(
            raw,
            "recognition",
            {
                "focus_confidence",
                "lock_confidence",
                "tracking_confidence",
                "required_frames",
            },
        )
        centering = _group(
            raw,
            "centering",
            {
                "lock_tolerance_ratio",
                "outer_corridor_ratio",
                "approach_yaw_rps",
                "recenter_yaw_rps",
            },
        )
        approach = _group(
            raw,
            "approach",
            {
                "forward_mps",
                "timeout_s",
                "duplicate_hold_s",
                "source_maximum_age_s",
                "detection_maximum_age_s",
                "slow_inference_grace_s",
            },
        )
        arrival = _group(
            raw,
            "arrival",
            {
                "near_bottom_ratio",
                "disappearance_bottom_ratio",
                "near_center_ratio",
                "near_confirmations",
                "loss_confirmations",
                "loss_grace_s",
                "final_push_mps",
                "final_push_duration_s",
            },
        )
        home = _group(
            raw,
            "home",
            {
                "align_yaw_rps",
                "align_tolerance_deg",
                "align_timeout_s",
                "return_forward_mps",
                "arrival_tolerance_m",
                "heading_gate_deg",
                "return_yaw_deadband_deg",
                "return_minimum_yaw_rps",
                "return_yaw_rps",
                "minimum_progress_m",
                "stall_timeout_s",
                "return_timeout_s",
                "settle_interval_s",
                "settled_sample_count",
                "settled_maximum_spread_m",
                "settled_sample_timeout_s",
                "settled_retry_count",
            },
        )

        ranges = _FRUIT_CONFIDENCE_RANGES[target]
        result = cls(
            target_fruit=target,
            search=SearchTuning(
                yaw_rps=_number(search, "yaw_rps", defaults.search.yaw_rps, 0.40, 0.80),
                sweep_rad=_number(
                    search,
                    "sweep_rad",
                    defaults.search.sweep_rad,
                    math.radians(45),
                    2 * math.pi,
                ),
                timeout_s=_number(
                    search, "timeout_s", defaults.search.timeout_s, 5.0, 60.0
                ),
                focus_yaw_rps=_number(
                    search,
                    "focus_yaw_rps",
                    defaults.search.focus_yaw_rps,
                    0.40,
                    0.80,
                ),
                progressive_focus_yaw_enabled=_boolean(
                    search,
                    "progressive_focus_yaw_enabled",
                    defaults.search.progressive_focus_yaw_enabled,
                ),
                focus_yaw_step_rps=_number(
                    search,
                    "focus_yaw_step_rps",
                    defaults.search.focus_yaw_step_rps,
                    0.05,
                    0.20,
                ),
                focus_minimum_yaw_rps=_number(
                    search,
                    "focus_minimum_yaw_rps",
                    defaults.search.focus_minimum_yaw_rps,
                    0.40,
                    0.80,
                ),
                focus_near_center_ratio=_number(
                    search,
                    "focus_near_center_ratio",
                    defaults.search.focus_near_center_ratio,
                    0.06,
                    0.35,
                ),
                focus_missing_grace_s=_number(
                    search,
                    "focus_missing_grace_s",
                    defaults.search.focus_missing_grace_s,
                    0.10,
                    1.0,
                ),
                bearing_routing_enabled=_boolean(
                    search,
                    "bearing_routing_enabled",
                    defaults.search.bearing_routing_enabled,
                ),
            ),
            recognition=RecognitionTuning(
                focus_confidence=_optional_number(
                    recognition,
                    "focus_confidence",
                    defaults.recognition.focus_confidence,
                    *ranges["focus"],
                ),
                lock_confidence=_number(
                    recognition,
                    "lock_confidence",
                    defaults.recognition.lock_confidence,
                    *ranges["lock"],
                ),
                tracking_confidence=_number(
                    recognition,
                    "tracking_confidence",
                    defaults.recognition.tracking_confidence,
                    *ranges["tracking"],
                ),
                required_frames=_integer(
                    recognition,
                    "required_frames",
                    defaults.recognition.required_frames,
                    2,
                    8,
                ),
            ),
            centering=CenteringTuning(
                lock_tolerance_ratio=_number(
                    centering,
                    "lock_tolerance_ratio",
                    defaults.centering.lock_tolerance_ratio,
                    0.05,
                    0.20,
                ),
                outer_corridor_ratio=_number(
                    centering,
                    "outer_corridor_ratio",
                    defaults.centering.outer_corridor_ratio,
                    0.12,
                    0.35,
                ),
                approach_yaw_rps=_number(
                    centering,
                    "approach_yaw_rps",
                    defaults.centering.approach_yaw_rps,
                    0.10,
                    0.80,
                ),
                recenter_yaw_rps=_number(
                    centering,
                    "recenter_yaw_rps",
                    defaults.centering.recenter_yaw_rps,
                    0.50,
                    0.80,
                ),
            ),
            approach=ApproachTuning(
                forward_mps=_number(
                    approach, "forward_mps", defaults.approach.forward_mps, 0.50, 1.0
                ),
                timeout_s=_number(
                    approach, "timeout_s", defaults.approach.timeout_s, 5.0, 60.0
                ),
                duplicate_hold_s=_number(
                    approach,
                    "duplicate_hold_s",
                    defaults.approach.duplicate_hold_s,
                    0.05,
                    0.250,
                ),
                source_maximum_age_s=_number(
                    approach,
                    "source_maximum_age_s",
                    defaults.approach.source_maximum_age_s,
                    0.05,
                    0.350,
                ),
                detection_maximum_age_s=_number(
                    approach,
                    "detection_maximum_age_s",
                    defaults.approach.detection_maximum_age_s,
                    0.05,
                    0.250,
                ),
                slow_inference_grace_s=_number(
                    approach,
                    "slow_inference_grace_s",
                    defaults.approach.slow_inference_grace_s,
                    0.250,
                    1.0,
                ),
            ),
            arrival=ArrivalTuning(
                near_bottom_ratio=_number(
                    arrival,
                    "near_bottom_ratio",
                    defaults.arrival.near_bottom_ratio,
                    0.60,
                    0.98,
                ),
                disappearance_bottom_ratio=_number(
                    arrival,
                    "disappearance_bottom_ratio",
                    defaults.arrival.disappearance_bottom_ratio,
                    0.60,
                    0.95,
                ),
                near_center_ratio=_number(
                    arrival,
                    "near_center_ratio",
                    defaults.arrival.near_center_ratio,
                    0.50,
                    0.95,
                ),
                near_confirmations=_integer(
                    arrival,
                    "near_confirmations",
                    defaults.arrival.near_confirmations,
                    2,
                    8,
                ),
                loss_confirmations=_integer(
                    arrival,
                    "loss_confirmations",
                    defaults.arrival.loss_confirmations,
                    2,
                    5,
                ),
                loss_grace_s=_number(
                    arrival, "loss_grace_s", defaults.arrival.loss_grace_s, 0.10, 1.5
                ),
                final_push_mps=_number(
                    arrival,
                    "final_push_mps",
                    defaults.arrival.final_push_mps,
                    0.50,
                    1.0,
                ),
                final_push_duration_s=_number(
                    arrival,
                    "final_push_duration_s",
                    defaults.arrival.final_push_duration_s,
                    0.0,
                    1.50,
                ),
            ),
            home=HomeTuning(
                align_yaw_rps=_number(
                    home, "align_yaw_rps", defaults.home.align_yaw_rps, 0.50, 0.80
                ),
                align_tolerance_deg=_number(
                    home,
                    "align_tolerance_deg",
                    defaults.home.align_tolerance_deg,
                    3.0,
                    15.0,
                ),
                align_timeout_s=_number(
                    home, "align_timeout_s", defaults.home.align_timeout_s, 10.0, 60.0
                ),
                return_forward_mps=_number(
                    home,
                    "return_forward_mps",
                    defaults.home.return_forward_mps,
                    0.50,
                    1.0,
                ),
                arrival_tolerance_m=_number(
                    home,
                    "arrival_tolerance_m",
                    defaults.home.arrival_tolerance_m,
                    0.05,
                    0.50,
                ),
                heading_gate_deg=_number(
                    home, "heading_gate_deg", defaults.home.heading_gate_deg, 10.0, 45.0
                ),
                return_yaw_deadband_deg=_number(
                    home,
                    "return_yaw_deadband_deg",
                    defaults.home.return_yaw_deadband_deg,
                    3.0,
                    15.0,
                ),
                return_minimum_yaw_rps=_number(
                    home,
                    "return_minimum_yaw_rps",
                    defaults.home.return_minimum_yaw_rps,
                    0.50,
                    0.80,
                ),
                return_yaw_rps=_number(
                    home, "return_yaw_rps", defaults.home.return_yaw_rps, 0.50, 0.80
                ),
                minimum_progress_m=_number(
                    home,
                    "minimum_progress_m",
                    defaults.home.minimum_progress_m,
                    0.01,
                    0.20,
                ),
                stall_timeout_s=_number(
                    home, "stall_timeout_s", defaults.home.stall_timeout_s, 0.5, 5.0
                ),
                return_timeout_s=_number(
                    home, "return_timeout_s", defaults.home.return_timeout_s, 10.0, 60.0
                ),
                settle_interval_s=_number(
                    home,
                    "settle_interval_s",
                    defaults.home.settle_interval_s,
                    0.0,
                    2.0,
                ),
                settled_sample_count=_integer(
                    home,
                    "settled_sample_count",
                    defaults.home.settled_sample_count,
                    3,
                    5,
                ),
                settled_maximum_spread_m=_number(
                    home,
                    "settled_maximum_spread_m",
                    defaults.home.settled_maximum_spread_m,
                    0.005,
                    0.05,
                ),
                settled_sample_timeout_s=_number(
                    home,
                    "settled_sample_timeout_s",
                    defaults.home.settled_sample_timeout_s,
                    0.25,
                    3.0,
                ),
                settled_retry_count=_integer(
                    home,
                    "settled_retry_count",
                    defaults.home.settled_retry_count,
                    0,
                    1,
                ),
            ),
        )
        result._validate_relationships()
        return result

    def _validate_relationships(self) -> None:
        focus = self.recognition.focus_confidence
        if focus is not None and focus < self.recognition.lock_confidence:
            raise ValueError(
                "focus confidence must be greater than or equal to lock confidence"
            )
        if self.centering.lock_tolerance_ratio >= self.centering.outer_corridor_ratio:
            raise ValueError("lock tolerance must be smaller than the outer corridor")
        if self.approach.detection_maximum_age_s > self.approach.source_maximum_age_s:
            raise ValueError("detection age cannot exceed source age")
        if self.search.focus_yaw_rps > self.search.yaw_rps:
            raise ValueError("focus yaw cannot exceed broad search yaw")
        if self.search.focus_minimum_yaw_rps > self.search.focus_yaw_rps:
            raise ValueError("minimum focus yaw cannot exceed focused alignment yaw")
        if (
            self.search.focus_near_center_ratio
            <= self.centering.lock_tolerance_ratio
        ):
            raise ValueError(
                "near-center focus band must exceed the lock tolerance"
            )
        if (
            self.approach.slow_inference_grace_s
            < self.approach.detection_maximum_age_s
        ):
            raise ValueError("slow inference grace cannot be below detection freshness")
        if self.approach.duplicate_hold_s > self.approach.detection_maximum_age_s:
            raise ValueError("duplicate hold cannot exceed detection freshness")
        if self.home.return_minimum_yaw_rps > self.home.return_yaw_rps:
            raise ValueError("minimum moving Home yaw cannot exceed its maximum")
        if self.home.return_yaw_deadband_deg >= self.home.heading_gate_deg:
            raise ValueError("moving Home yaw deadband must be smaller than its gate")
        if 0.0 < self.arrival.final_push_duration_s < 0.10:
            raise ValueError(
                "final push duration must be 0 (disabled) or at least 0.10 seconds"
            )
        if self.arrival.disappearance_bottom_ratio >= self.arrival.near_bottom_ratio:
            raise ValueError(
                "disappearance bottom threshold must be below direct Arrival threshold"
            )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def search_experiment_dict(self) -> dict[str, object]:
        """Compatibility projection for existing yaw scorecards."""
        return {
            "target_fruit": self.target_fruit,
            "search_yaw_rps": self.search.yaw_rps,
            "focus_confidence": self.recognition.focus_confidence,
            "lock_confidence": self.recognition.lock_confidence,
            "center_confirmations": self.recognition.required_frames,
            "center_tolerance_ratio": self.centering.lock_tolerance_ratio,
        }

    @classmethod
    def contract(cls) -> dict[str, object]:
        """Server-owned UI schema: labels, units, bounds, defaults, constraints."""
        fields: dict[str, list[dict[str, object]]] = {
            "search": [
                _field("yaw_rps", "Search yaw", "rad/s", 0.40, 0.80, 0.05),
                _field(
                    "sweep_rad",
                    "Search sweep",
                    "rad",
                    math.radians(45),
                    2 * math.pi,
                    0.1,
                ),
                _field("timeout_s", "Search timeout", "s", 5, 60, 1),
                _field(
                    "focus_yaw_rps",
                    "Focused alignment yaw",
                    "rad/s",
                    0.40,
                    0.80,
                    0.05,
                    safety="Yaw-only; never authorizes forward motion.",
                ),
                _boolean_field(
                    "progressive_focus_yaw_enabled",
                    "Slow focused yaw on each pass",
                    True,
                    safety=(
                        "Yaw-only. Disable to restore the fixed focused-yaw profile."
                    ),
                ),
                _field(
                    "focus_yaw_step_rps",
                    "Focused yaw reduction per pass",
                    "rad/s",
                    0.05,
                    0.20,
                    0.05,
                    safety="Applies only to fresh qualified off-center observations.",
                ),
                _field(
                    "focus_minimum_yaw_rps",
                    "Minimum focused yaw",
                    "rad/s",
                    0.40,
                    0.80,
                    0.05,
                    safety=(
                        "Never authorizes forward motion and cannot go below the "
                        "verified factory-avoidance turning floor."
                    ),
                ),
                _field(
                    "focus_near_center_ratio",
                    "Near-center focus band",
                    "frame ratio",
                    0.06,
                    0.35,
                    0.01,
                    safety=(
                        "Uses the verified minimum yaw for one bounded correction, "
                        "then holds zero for one fresh frame."
                    ),
                ),
                _field(
                    "focus_missing_grace_s",
                    "Focused missing-evidence grace",
                    "s",
                    0.10,
                    1.0,
                    0.05,
                    safety="Retains only yaw direction; lock counters do not advance.",
                ),
                _boolean_field(
                    "bearing_routing_enabled",
                    "Use mapped fruit bearing",
                    False,
                    safety=(
                        "Controls yaw routing only; mapping stays active and "
                        "selected-target guidance still owns translation and Arrival."
                    ),
                ),
            ],
            "recognition": [
                _field(
                    "focus_confidence",
                    "Focus confidence",
                    "ratio",
                    0,
                    1,
                    0.01,
                    target_specific=True,
                ),
                _field(
                    "lock_confidence",
                    "Lock confidence",
                    "ratio",
                    0,
                    1,
                    0.01,
                    target_specific=True,
                ),
                _field(
                    "tracking_confidence",
                    "Tracking confidence",
                    "ratio",
                    0,
                    1,
                    0.01,
                    target_specific=True,
                ),
                _field("required_frames", "Fresh centered frames", "frames", 2, 8, 1),
            ],
            "centering": [
                _field(
                    "lock_tolerance_ratio",
                    "Lock center tolerance",
                    "frame ratio",
                    0.05,
                    0.20,
                    0.01,
                ),
                _field(
                    "outer_corridor_ratio",
                    "Approach outer corridor",
                    "frame ratio",
                    0.12,
                    0.35,
                    0.01,
                ),
                _field(
                    "approach_yaw_rps",
                    "Moving correction yaw",
                    "rad/s",
                    0.10,
                    0.80,
                    0.05,
                ),
                _field(
                    "recenter_yaw_rps",
                    "In-place recenter yaw",
                    "rad/s",
                    0.50,
                    0.80,
                    0.05,
                ),
            ],
            "approach": [
                _field("forward_mps", "Approach speed", "m/s", 0.50, 1.0, 0.05),
                _field("timeout_s", "Approach timeout", "s", 5, 60, 1),
                _field(
                    "duplicate_hold_s", "Duplicate-frame hold", "s", 0.05, 0.250, 0.01
                ),
                _field(
                    "source_maximum_age_s",
                    "Source freshness ceiling",
                    "s",
                    0.05,
                    0.350,
                    0.01,
                    safety="May only tighten the hard 0.350 s stop ceiling.",
                ),
                _field(
                    "detection_maximum_age_s",
                    "Detection freshness ceiling",
                    "s",
                    0.05,
                    0.250,
                    0.01,
                    safety="May only tighten the hard 0.250 s stop ceiling.",
                ),
                _field(
                    "slow_inference_grace_s",
                    "Slow inference terminal grace",
                    "s",
                    0.250,
                    1.0,
                    0.05,
                    safety=(
                        "Motion remains exact zero after the 0.250 s freshness "
                        "ceiling; this only delays terminal failure."
                    ),
                ),
            ],
            "arrival": [
                _field(
                    "near_bottom_ratio",
                    "Direct Arrival bottom threshold",
                    "frame ratio",
                    0.60,
                    0.98,
                    0.01,
                ),
                _field(
                    "disappearance_bottom_ratio",
                    "Disappearance Arrival bottom threshold",
                    "frame ratio",
                    0.60,
                    0.95,
                    0.01,
                    safety=(
                        "Only the immediately following fresh missing frame qualifies; "
                        "must stay below the direct Arrival threshold."
                    ),
                ),
                _field(
                    "near_center_ratio",
                    "Near center threshold",
                    "frame ratio",
                    0.50,
                    0.95,
                    0.01,
                ),
                _field("near_confirmations", "Near confirmations", "frames", 2, 8, 1),
                _field("loss_confirmations", "Loss confirmations", "frames", 2, 5, 1),
                _field("loss_grace_s", "Near-loss grace", "s", 0.10, 1.5, 0.05),
                _field("final_push_mps", "Final push speed", "m/s", 0.50, 1.0, 0.05),
                _field(
                    "final_push_duration_s",
                    "Final push duration (0 disables)",
                    "s",
                    0.0,
                    1.50,
                    0.10,
                    safety=(
                        "Zero disables the push; non-zero values must be at least "
                        "0.10 s and remain bounded by the server."
                    ),
                ),
            ],
            "home": [
                _field(
                    "align_yaw_rps", "Home alignment yaw", "rad/s", 0.50, 0.80, 0.05
                ),
                _field(
                    "align_tolerance_deg", "Home alignment tolerance", "deg", 3, 15, 1
                ),
                _field("align_timeout_s", "Home alignment timeout", "s", 10, 60, 1),
                _field(
                    "return_forward_mps", "Home return speed", "m/s", 0.50, 1.0, 0.05
                ),
                _field(
                    "arrival_tolerance_m",
                    "Home arrival tolerance",
                    "m",
                    0.05,
                    0.50,
                    0.01,
                ),
                _field(
                    "heading_gate_deg", "Moving Home heading gate", "deg", 10, 45, 1
                ),
                _field(
                    "return_yaw_deadband_deg",
                    "Moving Home yaw deadband",
                    "deg",
                    3,
                    15,
                    1,
                    safety=(
                        "Yaw is exactly zero inside this band; outside it the "
                        "verified minimum turning signal applies."
                    ),
                ),
                _field(
                    "return_minimum_yaw_rps",
                    "Minimum moving Home yaw",
                    "rad/s",
                    0.50,
                    0.80,
                    0.05,
                    safety=(
                        "Never lower than the physically verified 0.50 rad/s "
                        "turning signal."
                    ),
                ),
                _field(
                    "return_yaw_rps",
                    "Maximum moving Home yaw",
                    "rad/s",
                    0.50,
                    0.80,
                    0.05,
                ),
                _field(
                    "minimum_progress_m", "Minimum Home progress", "m", 0.01, 0.20, 0.01
                ),
                _field("stall_timeout_s", "Home stall timeout", "s", 0.5, 5.0, 0.1),
                _field("return_timeout_s", "Home return timeout", "s", 10, 60, 1),
                _field(
                    "settle_interval_s",
                    "Post-stop settling interval",
                    "s",
                    0.0,
                    2.0,
                    0.05,
                    safety="Motion is exact-zero and disarmed throughout this interval.",
                ),
                _field(
                    "settled_sample_count",
                    "Settled Home samples",
                    "samples",
                    3,
                    5,
                    1,
                    safety="Every sample must be fresh, advancing, and inside Home.",
                ),
                _field(
                    "settled_maximum_spread_m",
                    "Settled Home maximum spread",
                    "m",
                    0.005,
                    0.05,
                    0.005,
                    safety="Does not widen the 0.10 m Home position gate.",
                ),
                _field(
                    "settled_sample_timeout_s",
                    "Settled Home sample timeout",
                    "s",
                    0.25,
                    3.0,
                    0.05,
                    safety="Timeout or non-advancing evidence fails closed.",
                ),
                _field(
                    "settled_retry_count",
                    "Position-only Home retries",
                    "retries",
                    0,
                    1,
                    1,
                    safety="At most one correction; heading restoration stays disabled.",
                ),
            ],
        }
        return {
            "schema_version": 1,
            "groups": fields,
            "fruits": {
                fruit: {
                    "defaults": cls.defaults(fruit).to_dict(),
                    "confidence_ranges": {
                        key: list(value) for key, value in ranges.items()
                    },
                }
                for fruit, ranges in _FRUIT_CONFIDENCE_RANGES.items()
            },
            "hard_invariants": [
                "single motion owner and control lease",
                "command watchdog and absolute motion envelopes",
                "exact-zero disarm",
                "stale, wrong-generation, wrong-label, and camera-failure stop",
                "inter-run Home clearance",
            ],
        }


def _field(
    name: str,
    label: str,
    unit: str,
    minimum: float,
    maximum: float,
    step: float,
    *,
    target_specific: bool = False,
    safety: str = "Bounded by the server before motion.",
) -> dict[str, object]:
    return {
        "name": name,
        "label": label,
        "unit": unit,
        "minimum": minimum,
        "maximum": maximum,
        "step": step,
        "target_specific": target_specific,
        "safety": safety,
    }


def _boolean_field(
    name: str,
    label: str,
    default: bool,
    *,
    safety: str,
) -> dict[str, object]:
    return {
        "name": name,
        "label": label,
        "unit": "boolean",
        "type": "boolean",
        "default": default,
        "target_specific": False,
        "safety": safety,
    }
