"""Resident fruit-model routing behind one normalized prediction interface."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from media.fruit_color import classify_bbox_color, classify_replacement_candidate


@dataclass(frozen=True)
class FruitCandidate:
    confidence: float
    bbox_xyxy: tuple[int, int, int, int]


@dataclass(frozen=True)
class RoutedPrediction:
    candidate: FruitCandidate | None
    inference_passes: int
    route: dict[str, object]


def _best_candidate(
    results: object,
    *,
    x_offset: int = 0,
    y_offset: int = 0,
) -> FruitCandidate | None:
    if not isinstance(results, (list, tuple)) or not results:
        return None
    boxes = getattr(results[0], "boxes", None)
    if boxes is None or len(boxes) == 0:
        return None
    confidences = boxes.conf.detach().cpu().tolist()
    best_index = max(range(len(confidences)), key=confidences.__getitem__)
    confidence = float(confidences[best_index])
    coordinates = boxes.xyxy[best_index].detach().cpu().tolist()
    x1, y1, x2, y2 = (round(float(value)) for value in coordinates)
    return FruitCandidate(
        confidence=confidence,
        bbox_xyxy=(
            x1 + x_offset,
            y1 + y_offset,
            x2 + x_offset,
            y2 + y_offset,
        ),
    )


def _best_labeled_candidate(
    results: object,
    *,
    labels_by_id: dict[int, str],
) -> tuple[FruitCandidate, str] | None:
    if not isinstance(results, (list, tuple)) or not results:
        return None
    boxes = getattr(results[0], "boxes", None)
    if boxes is None or len(boxes) == 0:
        return None
    confidences = boxes.conf.detach().cpu().tolist()
    class_ids = boxes.cls.detach().cpu().tolist()
    best_index = max(range(len(confidences)), key=confidences.__getitem__)
    class_id = int(class_ids[best_index])
    label = labels_by_id.get(class_id)
    if label is None:
        return None
    coordinates = boxes.xyxy[best_index].detach().cpu().tolist()
    x1, y1, x2, y2 = (round(float(value)) for value in coordinates)
    return FruitCandidate(float(confidences[best_index]), (x1, y1, x2, y2)), label


def _bbox_iou(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


class FruitModelRouter:
    """Route one requested fruit through resident general/specialist models."""

    def __init__(
        self,
        *,
        general_model: Any,
        general_class_ids: dict[str, int],
        banana_specialist_model: Any | None = None,
        banana_specialist_class_id: int | None = None,
        banana_minimum_confidence: float = 0.55,
        banana_minimum_agreement_iou: float = 0.10,
        mango_model: Any | None = None,
        mango_class_ids: dict[str, int] | None = None,
        mango_minimum_raw_confidence: float = 0.08,
        mango_minimum_color_confidence: float = 0.80,
    ) -> None:
        for name, value in (
            ("banana minimum confidence", banana_minimum_confidence),
            ("banana minimum agreement IoU", banana_minimum_agreement_iou),
            ("mango minimum color confidence", mango_minimum_color_confidence),
            ("mango minimum raw confidence", mango_minimum_raw_confidence),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if (banana_specialist_model is None) != (
            banana_specialist_class_id is None
        ):
            raise ValueError(
                "banana specialist model and class ID must be configured together"
            )
        if (mango_model is None) != (mango_class_ids is None):
            raise ValueError("mango model and class IDs must be configured together")
        self._general_model = general_model
        self._general_class_ids = dict(general_class_ids)
        self._banana_specialist_model = banana_specialist_model
        self._banana_specialist_class_id = banana_specialist_class_id
        self._banana_minimum_confidence = banana_minimum_confidence
        self._banana_minimum_agreement_iou = banana_minimum_agreement_iou
        self._mango_model = mango_model
        self._mango_class_ids = dict(mango_class_ids or {})
        self._mango_minimum_raw_confidence = mango_minimum_raw_confidence
        self._mango_minimum_color_confidence = mango_minimum_color_confidence

    @property
    def banana_specialist_enabled(self) -> bool:
        return self._banana_specialist_model is not None

    def status(self) -> dict[str, object]:
        return {
            "mode": "general_with_banana_specialist"
            if self.banana_specialist_enabled
            else "general_only",
            "banana_specialist_loaded": self.banana_specialist_enabled,
            "banana_minimum_confidence": self._banana_minimum_confidence,
            "banana_minimum_agreement_iou": self._banana_minimum_agreement_iou,
            "mango_derived_route_loaded": self._mango_model is not None,
            "mango_raw_labels": list(self._mango_class_ids),
            "mango_minimum_raw_confidence": self._mango_minimum_raw_confidence,
            "mango_minimum_color_confidence": self._mango_minimum_color_confidence,
        }

    def predict(
        self,
        *,
        source: Any,
        target_fruit: str,
        device: int | str,
        confidence_floor: float = 0.01,
        x_offset: int = 0,
        y_offset: int = 0,
    ) -> RoutedPrediction:
        normalized_target = target_fruit.casefold().strip()
        if normalized_target == "mango":
            if self._mango_model is None or not self._mango_class_ids:
                raise ValueError("mango derived route is unavailable")
            raw_results = self._mango_model.predict(
                source=source,
                conf=confidence_floor,
                classes=list(self._mango_class_ids.values()),
                device=device,
                verbose=False,
            )
            labeled = _best_labeled_candidate(
                raw_results,
                labels_by_id={value: key for key, value in self._mango_class_ids.items()},
            )
            if labeled is None:
                return RoutedPrediction(
                    candidate=None,
                    inference_passes=1,
                    route={
                        "mode": "mango_derived",
                        "triggered": False,
                        "confirmed": False,
                        "raw_label": None,
                        "raw_confidence": None,
                        "raw_bbox_xyxy": None,
                        "derived_identity": "unknown",
                        "derived_confidence": 0.0,
                    },
                )
            raw_local, raw_label = labeled
            color = classify_bbox_color(source, raw_local.bbox_xyxy)
            shape = getattr(source, "shape", None)
            source_height = (
                int(shape[0])
                if isinstance(shape, (list, tuple)) and len(shape) >= 2
                else None
            )
            source_width = (
                int(shape[1])
                if isinstance(shape, (list, tuple)) and len(shape) >= 2
                else None
            )
            identity = classify_replacement_candidate(
                raw_label,
                color,
                raw_local.bbox_xyxy,
                source_width=source_width,
                source_height=source_height,
            )
            derived_confidence = float(identity["confidence"])
            confirmed = bool(
                identity["identity"] == "mango"
                and raw_local.confidence > self._mango_minimum_raw_confidence
                and derived_confidence >= self._mango_minimum_color_confidence
            )
            global_bbox = tuple(
                value + (x_offset if index % 2 == 0 else y_offset)
                for index, value in enumerate(raw_local.bbox_xyxy)
            )
            candidate = (
                FruitCandidate(raw_local.confidence, global_bbox) if confirmed else None
            )
            return RoutedPrediction(
                candidate=candidate,
                inference_passes=1,
                route={
                    "mode": "mango_derived",
                    "triggered": True,
                    "confirmed": confirmed,
                    "raw_label": raw_label,
                    "raw_confidence": raw_local.confidence,
                    "raw_bbox_xyxy": list(global_bbox),
                    "derived_identity": identity["identity"],
                    "derived_confidence": derived_confidence,
                },
            )
        try:
            general_class_id = self._general_class_ids[normalized_target]
        except KeyError as exc:
            raise ValueError(f"general model does not support {target_fruit}") from exc
        general_results = self._general_model.predict(
            source=source,
            conf=confidence_floor,
            classes=[general_class_id],
            device=device,
            verbose=False,
        )
        general = _best_candidate(
            general_results,
            x_offset=x_offset,
            y_offset=y_offset,
        )
        if normalized_target != "banana" or not self.banana_specialist_enabled:
            return RoutedPrediction(
                candidate=general,
                inference_passes=1,
                route={
                    "mode": "general",
                    "triggered": False,
                    "confirmed": general is not None,
                },
            )

        empty_route = {
            "mode": "banana_specialist",
            "triggered": False,
            "confirmed": False,
            "general_confidence": None,
            "specialist_confidence": None,
            "agreement_iou": None,
        }
        if general is None:
            return RoutedPrediction(
                candidate=None,
                inference_passes=1,
                route=empty_route,
            )

        assert self._banana_specialist_model is not None
        assert self._banana_specialist_class_id is not None
        specialist_results = self._banana_specialist_model.predict(
            source=source,
            conf=confidence_floor,
            classes=[self._banana_specialist_class_id],
            device=device,
            verbose=False,
        )
        specialist = _best_candidate(
            specialist_results,
            x_offset=x_offset,
            y_offset=y_offset,
        )
        agreement_iou = (
            None
            if specialist is None
            else _bbox_iou(general.bbox_xyxy, specialist.bbox_xyxy)
        )
        confirmed = bool(
            specialist is not None
            and specialist.confidence >= self._banana_minimum_confidence
            and agreement_iou is not None
            and agreement_iou >= self._banana_minimum_agreement_iou
        )
        return RoutedPrediction(
            candidate=specialist if confirmed else None,
            inference_passes=2,
            route={
                "mode": "banana_specialist",
                "triggered": True,
                "confirmed": confirmed,
                "general_confidence": general.confidence,
                "specialist_confidence": (
                    None if specialist is None else specialist.confidence
                ),
                "agreement_iou": agreement_iou,
            },
        )
