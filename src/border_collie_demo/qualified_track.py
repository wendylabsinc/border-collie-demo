"""Short-form temporal gate for motion-qualified fruit tracking.

This module deliberately answers one question: may the current target evidence
continue to drive the existing approach controller?  The mature tracker branch
adds richer motion recommendations and history scoring behind a similar seam.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


class TrackIdentityChanged(RuntimeError):
    pass


@dataclass(frozen=True)
class TrackDecision:
    ready: bool
    target_visible: bool
    requested_ready: bool
    acquisition_candidate: bool
    close_range_continuation: bool
    confidence: float | None
    center_x_ratio: float | None
    center_y_ratio: float | None
    bottom_ratio: float | None


class QualifiedTrackGate:
    """Maintain the minimum temporal state needed for safe approach continuity."""

    def __init__(
        self,
        *,
        target_fruit: str,
        acquisition_confidence: float,
        acquisition_confirmations: int,
        close_range_confidence: float,
        close_range_minimum_bottom_ratio: float,
        maximum_center_delta_ratio: float,
        maximum_vertical_retreat_ratio: float,
    ) -> None:
        self.target = target_fruit.casefold().strip()
        self.acquisition_confidence = float(acquisition_confidence)
        self.acquisition_confirmations = int(acquisition_confirmations)
        self.close_range_confidence = float(close_range_confidence)
        self.close_range_minimum_bottom_ratio = float(
            close_range_minimum_bottom_ratio
        )
        self.maximum_center_delta_ratio = float(maximum_center_delta_ratio)
        self.maximum_vertical_retreat_ratio = float(maximum_vertical_retreat_ratio)
        if not self.target or self.acquisition_confirmations < 1:
            raise ValueError("qualified track target and confirmations are required")
        self._confirmations = 0
        self._acquired = False
        self._last_geometry: tuple[float, float, float] | None = None
        self.minimum_observed_confidence: float | None = None
        self.close_range_continuation_samples = 0

    def observe(self, status: dict[str, object]) -> TrackDecision:
        detection = status.get("detection")
        detection = detection if isinstance(detection, dict) else None
        label = (
            str(detection.get("label") or "").casefold() if detection else ""
        )
        confidence = _number(detection.get("confidence")) if detection else None
        center_x = _number(detection.get("center_x_ratio")) if detection else None
        center_y = _number(detection.get("center_y_ratio")) if detection else None
        bottom = _number(detection.get("bottom_ratio")) if detection else None
        target_ready = bool(status.get("target_ready"))
        if target_ready and detection is not None and label != self.target:
            raise TrackIdentityChanged(
                f"qualified {self.target} track changed identity to {label or 'unknown'}"
            )

        target_visible = detection is not None and label == self.target
        requested_ready = target_ready and target_visible
        acquisition_candidate = bool(
            target_visible
            and confidence is not None
            and confidence >= self.acquisition_confidence
        )
        close_range_continuation = bool(
            self._acquired
            and target_visible
            and confidence is not None
            and confidence >= self.close_range_confidence
            and center_x is not None
            and center_y is not None
            and bottom is not None
            and self._last_geometry is not None
            and max(bottom, self._last_geometry[2])
            >= self.close_range_minimum_bottom_ratio
            and abs(center_x - self._last_geometry[0])
            <= self.maximum_center_delta_ratio
            and center_y
            >= self._last_geometry[1] - self.maximum_vertical_retreat_ratio
            and bottom
            >= self._last_geometry[2] - self.maximum_vertical_retreat_ratio
        )
        if requested_ready:
            self._confirmations = self.acquisition_confirmations
        elif acquisition_candidate:
            self._confirmations += 1
        elif close_range_continuation:
            self._confirmations = max(
                self._confirmations,
                self.acquisition_confirmations,
            )
        else:
            self._confirmations = 0
        ready = bool(
            requested_ready
            or (
                acquisition_candidate
                and self._confirmations >= self.acquisition_confirmations
            )
            or close_range_continuation
        )
        if ready and None not in (center_x, center_y, bottom):
            self._acquired = True
            self._last_geometry = (float(center_x), float(center_y), float(bottom))
            if confidence is not None:
                self.minimum_observed_confidence = (
                    confidence
                    if self.minimum_observed_confidence is None
                    else min(self.minimum_observed_confidence, confidence)
                )
            if close_range_continuation and not (
                requested_ready or acquisition_candidate
            ):
                self.close_range_continuation_samples += 1
        return TrackDecision(
            ready=ready,
            target_visible=target_visible,
            requested_ready=requested_ready,
            acquisition_candidate=acquisition_candidate,
            close_range_continuation=close_range_continuation,
            confidence=confidence,
            center_x_ratio=center_x,
            center_y_ratio=center_y,
            bottom_ratio=bottom,
        )
