"""Persistent mission-lifetime fruit identity and motion qualification.

YOLO detections are observations of a track, never the track itself.  The
tracker deliberately keeps identity, geometry, freshness, and motion authority
as separate facts so search and approach can share one history without making
a weak frame look like a new fruit.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from .fruits import fruit_policy


class FruitTrackState(StrEnum):
    UNSEEN = "unseen"
    CANDIDATE = "candidate"
    LOCKED = "locked"
    DEGRADED = "degraded"
    LOST = "lost"
    LOCKED_OFF_AXIS = "locked_off_axis"


@dataclass(frozen=True)
class PersistentFruitTrackerConfig:
    target_fruit: str
    acquisition_confidence: float
    maintenance_confidence: float
    acquisition_confirmations: int
    maximum_evidence_age_s: float = 0.250
    degraded_grace_s: float = 0.250
    center_corridor_ratio: float = 0.20
    confidence_ema_alpha: float = 0.60
    crop_minimum_iou: float = 0.10

    def __post_init__(self) -> None:
        if not self.target_fruit.casefold().strip():
            raise ValueError("target_fruit must be non-empty")
        ratios = (
            self.acquisition_confidence,
            self.maintenance_confidence,
            self.center_corridor_ratio,
            self.confidence_ema_alpha,
            self.crop_minimum_iou,
        )
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in ratios):
            raise ValueError("fruit tracker ratios must be finite and in [0, 1]")
        if self.maintenance_confidence > self.acquisition_confidence:
            raise ValueError("maintenance confidence cannot exceed acquisition")
        if self.acquisition_confirmations < 1:
            raise ValueError("acquisition_confirmations must be positive")
        if (
            not math.isfinite(self.maximum_evidence_age_s)
            or self.maximum_evidence_age_s <= 0.0
            or not math.isfinite(self.degraded_grace_s)
            or self.degraded_grace_s <= 0.0
            or self.degraded_grace_s > self.maximum_evidence_age_s
        ):
            raise ValueError("tracker time limits must be finite and bounded")


@dataclass(frozen=True)
class FruitTrackReport:
    state: FruitTrackState
    state_before: FruitTrackState
    same_identity: bool
    identity_confidence: float | None
    filtered_confidence: float | None
    evidence_age_s: float | None
    geometry_safe: bool
    motion_authorized: bool
    alignment_authorized: bool
    hold_authorized: bool
    arrival_eligible: bool
    arrival_counter: int
    reason: str
    identity_route: str | None
    geometry_route: str | None
    source_pts: int | None
    generation: str | None
    acquisition_epoch: int
    degraded_failures: int
    degraded_elapsed_s: float
    full_frame_detection: Mapping[str, object] | None
    crop_detection: Mapping[str, object] | None

    def to_evidence(self) -> dict[str, object]:
        return {
            "track_state_before": self.state_before.value,
            "track_state_after": self.state.value,
            "same_identity": self.same_identity,
            "identity_confidence": self.identity_confidence,
            "filtered_confidence": self.filtered_confidence,
            "evidence_age_s": self.evidence_age_s,
            "geometry_safe": self.geometry_safe,
            "motion_authorized": self.motion_authorized,
            "alignment_authorized": self.alignment_authorized,
            "hold_authorized": self.hold_authorized,
            "arrival_eligible": self.arrival_eligible,
            "arrival_counter": self.arrival_counter,
            "tracking_reason": self.reason,
            "identity_route": self.identity_route,
            "geometry_route": self.geometry_route,
            "source_pts": self.source_pts,
            "generation": self.generation,
            "acquisition_epoch": self.acquisition_epoch,
            "degraded_failures": self.degraded_failures,
            "degraded_elapsed_s": self.degraded_elapsed_s,
            "full_frame_detection": (
                None
                if self.full_frame_detection is None
                else dict(self.full_frame_detection)
            ),
            "crop_detection": (
                None if self.crop_detection is None else dict(self.crop_detection)
            ),
        }


@dataclass(frozen=True)
class _Detection:
    raw: Mapping[str, object]
    label: str
    confidence: float
    bbox: tuple[float, float, float, float]
    center_x: float
    center_y: float
    bottom: float
    area: float
    age_s: float
    source_pts: int
    generation: str
    route: str


class PersistentFruitTracker:
    """Consume full-frame plus optional crop evidence for one Demo Run."""

    def __init__(self, config: PersistentFruitTrackerConfig) -> None:
        self.config = config
        self._state = FruitTrackState.UNSEEN
        self._generation: str | None = None
        self._candidate_samples = 0
        self._acquisition_epoch = 0
        self._last_source_pts: int | None = None
        self._last_good_at: float | None = None
        self._degraded_started_at: float | None = None
        self._degraded_failures = 0
        self._filtered_confidence: float | None = None
        self._arrival_counter = 0
        self._last_geometry: _Detection | None = None

    @property
    def state(self) -> FruitTrackState:
        return self._state

    @classmethod
    def for_fruit(
        cls,
        target_fruit: str,
        *,
        acquisition_confirmations: int,
    ) -> PersistentFruitTracker:
        policy = fruit_policy(target_fruit)
        # The persistent track owns identity hysteresis.  The existing fruit
        # policy remains the maintenance floor until each lower value has
        # physical evidence; EMA and the one-frame degraded state remove the
        # raw-frame threshold oscillation without silently weakening it.
        return cls(
            PersistentFruitTrackerConfig(
                target_fruit=target_fruit.casefold().strip(),
                acquisition_confidence=policy.acquisition_confidence,
                maintenance_confidence=policy.close_range_tracking_confidence,
                acquisition_confirmations=acquisition_confirmations,
            )
        )

    def observe(
        self,
        status: Mapping[str, object],
        *,
        now_s: float,
    ) -> FruitTrackReport:
        if not math.isfinite(now_s):
            raise ValueError("observation time must be finite")
        before = self._state
        observations = status.get("observations")
        full_raw: object = None
        crop_raw: object = None
        if isinstance(observations, Mapping):
            full_raw = observations.get("full_frame")
            crop_raw = observations.get("crop")
        else:
            # Backward-compatible adapter for saved v28 evidence. The media
            # sidecar now publishes the explicit observations shape.
            full_raw = status.get("detection")
        full = self._parse_detection(full_raw)
        crop = self._parse_detection(crop_raw)
        generation = status.get("generation")
        generation_value = generation if isinstance(generation, str) else None

        if status.get("camera_healthy") is not True:
            return self._hard_loss(
                before,
                "camera_unhealthy",
                full,
                crop,
                generation_value,
            )
        if self._generation is not None and generation_value != self._generation:
            return self._hard_loss(
                before,
                "generation_changed",
                full,
                crop,
                generation_value,
            )
        if full is not None and full.generation != generation_value:
            return self._hard_loss(
                before,
                "generation_mismatch",
                full,
                crop,
                generation_value,
            )
        if full is not None and full.age_s > self.config.maximum_evidence_age_s:
            return self._hard_loss(
                before,
                "stale_full_frame_observation",
                full,
                crop,
                generation_value,
            )
        if full is not None and full.label != self.config.target_fruit.casefold():
            return self._hard_loss(
                before,
                "confirmed_wrong_identity",
                full,
                crop,
                generation_value,
            )

        source = status.get("source")
        source_pts = source.get("pts") if isinstance(source, Mapping) else None
        current_pts = source_pts if isinstance(source_pts, int) else None
        if full is not None:
            current_pts = full.source_pts
        duplicate = current_pts is not None and current_pts == self._last_source_pts
        if duplicate:
            return self._degrade(
                before,
                "duplicate_frame",
                now_s,
                full,
                crop,
                generation_value,
                count_failure=False,
            )

        if full is None:
            return self._miss(
                before,
                now_s,
                full,
                crop,
                generation_value,
                current_pts,
            )

        if not self._geometry_possible(full):
            return self._hard_loss(
                before,
                "impossible_geometry",
                full,
                crop,
                generation_value,
            )

        self._generation = generation_value
        self._last_source_pts = full.source_pts
        same_identity = full.label == self.config.target_fruit.casefold()
        geometry = self._refined_geometry(full, crop)
        effective_confidence = self._refined_confidence(full, crop)
        maintenance_qualified = (
            effective_confidence >= self.config.maintenance_confidence
        )
        acquisition_qualified = (
            full.confidence >= self.config.acquisition_confidence
        )

        if self._state in {FruitTrackState.UNSEEN, FruitTrackState.CANDIDATE, FruitTrackState.LOST}:
            if not acquisition_qualified:
                self._candidate_samples = 0
                self._state = FruitTrackState.UNSEEN
                return self._report(
                    before,
                    "full_frame_below_acquisition",
                    full,
                    crop,
                    geometry,
                    same_identity=False,
                    motion_authorized=False,
                    alignment_authorized=False,
                )
            self._candidate_samples += 1
            self._state = FruitTrackState.CANDIDATE
            self._update_confidence(effective_confidence)
            self._last_geometry = geometry
            self._last_good_at = now_s
            if self._candidate_samples < self.config.acquisition_confirmations:
                return self._report(
                    before,
                    "confirming_identity",
                    full,
                    crop,
                    geometry,
                    same_identity=True,
                    motion_authorized=False,
                    alignment_authorized=True,
                )
            self._acquisition_epoch += 1
            self._degraded_started_at = None
            self._degraded_failures = 0
            return self._lock_report(before, full, crop, geometry, now_s)

        if not maintenance_qualified:
            return self._degrade(
                before,
                "weak_full_frame_observation",
                now_s,
                full,
                crop,
                generation_value,
                count_failure=True,
            )

        self._candidate_samples = max(
            self._candidate_samples,
            self.config.acquisition_confirmations,
        )
        self._degraded_started_at = None
        self._degraded_failures = 0
        self._update_confidence(effective_confidence)
        self._last_geometry = geometry
        self._last_good_at = now_s
        return self._lock_report(before, full, crop, geometry, now_s)

    def _lock_report(
        self,
        before: FruitTrackState,
        full: _Detection,
        crop: _Detection | None,
        geometry: _Detection,
        now_s: float,
    ) -> FruitTrackReport:
        off_axis = (
            abs(geometry.center_x - 0.5) > self.config.center_corridor_ratio
        )
        self._state = (
            FruitTrackState.LOCKED_OFF_AXIS if off_axis else FruitTrackState.LOCKED
        )
        self._arrival_counter = self._arrival_counter + 1 if not off_axis else 0
        self._last_good_at = now_s
        return self._report(
            before,
            "locked_off_axis" if off_axis else "locked",
            full,
            crop,
            geometry,
            same_identity=True,
            motion_authorized=not off_axis,
            alignment_authorized=off_axis,
            arrival_eligible=not off_axis,
        )

    def _miss(
        self,
        before: FruitTrackState,
        now_s: float,
        full: _Detection | None,
        crop: _Detection | None,
        generation: str | None,
        source_pts: int | None,
    ) -> FruitTrackReport:
        if source_pts is not None:
            self._last_source_pts = source_pts
        if self._state in {FruitTrackState.UNSEEN, FruitTrackState.CANDIDATE, FruitTrackState.LOST}:
            self._candidate_samples = 0
            self._state = FruitTrackState.UNSEEN
            return self._report(
                before,
                "full_frame_missing",
                full,
                crop,
                self._last_geometry,
                same_identity=False,
                motion_authorized=False,
                alignment_authorized=False,
            )
        return self._degrade(
            before,
            "missing_full_frame_observation",
            now_s,
            full,
            crop,
            generation,
            count_failure=True,
        )

    def _degrade(
        self,
        before: FruitTrackState,
        reason: str,
        now_s: float,
        full: _Detection | None,
        crop: _Detection | None,
        generation: str | None,
        *,
        count_failure: bool,
    ) -> FruitTrackReport:
        if self._state in {FruitTrackState.UNSEEN, FruitTrackState.CANDIDATE, FruitTrackState.LOST}:
            self._state = FruitTrackState.UNSEEN
            return self._report(
                before,
                reason,
                full,
                crop,
                self._last_geometry,
                same_identity=False,
                motion_authorized=False,
                alignment_authorized=False,
            )
        if self._degraded_started_at is None:
            self._degraded_started_at = now_s
        if count_failure:
            self._degraded_failures += 1
        elapsed = max(0.0, now_s - self._degraded_started_at)
        if self._degraded_failures >= 2 or elapsed > self.config.degraded_grace_s:
            self._state = FruitTrackState.LOST
            self._arrival_counter = 0
            return self._report(
                before,
                "confirmed_full_frame_loss",
                full,
                crop,
                self._last_geometry,
                same_identity=False,
                motion_authorized=False,
                alignment_authorized=False,
            )
        self._state = FruitTrackState.DEGRADED
        return self._report(
            before,
            reason,
            full,
            crop,
            self._last_geometry,
            same_identity=True,
            motion_authorized=False,
            alignment_authorized=False,
            hold_authorized=(
                self._last_good_at is not None
                and now_s - self._last_good_at <= self.config.degraded_grace_s
            ),
        )

    def _hard_loss(
        self,
        before: FruitTrackState,
        reason: str,
        full: _Detection | None,
        crop: _Detection | None,
        generation: str | None,
    ) -> FruitTrackReport:
        self._state = FruitTrackState.LOST
        self._arrival_counter = 0
        return self._report(
            before,
            reason,
            full,
            crop,
            self._last_geometry,
            same_identity=False,
            motion_authorized=False,
            alignment_authorized=False,
        )

    def _report(
        self,
        before: FruitTrackState,
        reason: str,
        full: _Detection | None,
        crop: _Detection | None,
        geometry: _Detection | None,
        *,
        same_identity: bool,
        motion_authorized: bool,
        alignment_authorized: bool,
        hold_authorized: bool = False,
        arrival_eligible: bool = False,
    ) -> FruitTrackReport:
        degraded_elapsed = 0.0
        if self._degraded_started_at is not None and self._last_good_at is not None:
            degraded_elapsed = max(0.0, self._degraded_started_at - self._last_good_at)
        return FruitTrackReport(
            state=self._state,
            state_before=before,
            same_identity=same_identity,
            identity_confidence=(None if full is None else full.confidence),
            filtered_confidence=self._filtered_confidence,
            evidence_age_s=(None if full is None else full.age_s),
            geometry_safe=bool(
                geometry is not None
                and abs(geometry.center_x - 0.5) <= self.config.center_corridor_ratio
            ),
            motion_authorized=motion_authorized,
            alignment_authorized=alignment_authorized,
            hold_authorized=hold_authorized,
            arrival_eligible=arrival_eligible,
            arrival_counter=self._arrival_counter,
            reason=reason,
            identity_route=(None if full is None else "full_frame"),
            geometry_route=(None if geometry is None else geometry.route),
            source_pts=(None if full is None else full.source_pts),
            generation=(
                full.generation
                if full is not None
                else self._generation
            ),
            acquisition_epoch=self._acquisition_epoch,
            degraded_failures=self._degraded_failures,
            degraded_elapsed_s=degraded_elapsed,
            full_frame_detection=(None if full is None else full.raw),
            crop_detection=(None if crop is None else crop.raw),
        )

    def _update_confidence(self, confidence: float) -> None:
        self._filtered_confidence = (
            confidence
            if self._filtered_confidence is None
            else self.config.confidence_ema_alpha * confidence
            + (1.0 - self.config.confidence_ema_alpha)
            * self._filtered_confidence
        )

    def _refined_confidence(
        self,
        full: _Detection,
        crop: _Detection | None,
    ) -> float:
        if self._crop_agrees(full, crop):
            assert crop is not None
            return max(full.confidence, crop.confidence)
        return full.confidence

    def _refined_geometry(
        self,
        full: _Detection,
        crop: _Detection | None,
    ) -> _Detection:
        if self._crop_agrees(full, crop):
            assert crop is not None
            return crop
        return full

    def _crop_agrees(self, full: _Detection, crop: _Detection | None) -> bool:
        return bool(
            crop is not None
            and crop.label == full.label
            and crop.generation == full.generation
            and crop.source_pts == full.source_pts
            and _iou(full.bbox, crop.bbox) >= self.config.crop_minimum_iou
        )

    @staticmethod
    def _geometry_possible(detection: _Detection) -> bool:
        return bool(
            0.0 <= detection.center_x <= 1.0
            and 0.0 <= detection.center_y <= detection.bottom <= 1.0
            and 0.0 < detection.area <= 1.0
        )

    @staticmethod
    def _parse_detection(raw: object) -> _Detection | None:
        if not isinstance(raw, Mapping) or not raw:
            return None
        label = str(raw.get("label") or "").casefold().strip()
        confidence = _number(raw.get("confidence"))
        center_x = _number(raw.get("center_x_ratio"))
        center_y = _number(raw.get("center_y_ratio"))
        bottom = _number(raw.get("bottom_ratio"))
        area = _number(raw.get("bbox_area_ratio"))
        age_s = _number(raw.get("age_s"))
        source_pts = raw.get("source_pts")
        generation = raw.get("generation")
        bbox_raw = raw.get("bbox_xyxy")
        if not (
            label
            and None not in (confidence, center_x, center_y, bottom, area, age_s)
            and isinstance(source_pts, int)
            and not isinstance(source_pts, bool)
            and isinstance(generation, str)
            and generation
        ):
            return None
        assert confidence is not None
        assert center_x is not None and center_y is not None and bottom is not None
        assert area is not None and age_s is not None
        if isinstance(bbox_raw, (list, tuple)) and len(bbox_raw) == 4:
            bbox_values = tuple(_number(value) for value in bbox_raw)
            if any(value is None for value in bbox_values):
                return None
            bbox = tuple(float(value) for value in bbox_values if value is not None)
        else:
            # Normalized saved/test evidence can predate raw box retention.
            # This approximation is used only for temporal overlap; current
            # media evidence always includes the authoritative pixel box.
            side = math.sqrt(max(area, 1e-9))
            bbox = (
                center_x - side / 2.0,
                center_y - side / 2.0,
                center_x + side / 2.0,
                center_y + side / 2.0,
            )
        return _Detection(
            raw=raw,
            label=label,
            confidence=confidence,
            bbox=bbox,
            center_x=center_x,
            center_y=center_y,
            bottom=bottom,
            area=area,
            age_s=age_s,
            source_pts=source_pts,
            generation=generation,
            route=str(raw.get("route") or "unknown"),
        )


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0
