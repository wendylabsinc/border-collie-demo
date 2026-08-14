"""Resident fruit-model routing behind one normalized prediction interface."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class FruitCandidate:
    confidence: float
    bbox_xyxy: tuple[int, int, int, int]


@dataclass(frozen=True)
class RoutedPrediction:
    candidate: FruitCandidate | None
    inference_passes: int
    route: dict[str, object]
    observations: dict[str, FruitCandidate] = field(default_factory=dict)


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


def _best_candidates_by_class(
    results: object,
    *,
    class_names: dict[int, str],
    x_offset: int = 0,
    y_offset: int = 0,
) -> dict[str, FruitCandidate]:
    if not isinstance(results, (list, tuple)) or not results:
        return {}
    boxes = getattr(results[0], "boxes", None)
    if boxes is None or len(boxes) == 0:
        return {}
    confidences = boxes.conf.detach().cpu().tolist()
    classes = boxes.cls.detach().cpu().tolist()
    best: dict[str, FruitCandidate] = {}
    for index, (raw_confidence, raw_class) in enumerate(
        zip(confidences, classes, strict=True)
    ):
        label = class_names.get(int(raw_class))
        if label is None:
            continue
        confidence = float(raw_confidence)
        existing = best.get(label)
        if existing is not None and existing.confidence >= confidence:
            continue
        coordinates = boxes.xyxy[index].detach().cpu().tolist()
        x1, y1, x2, y2 = (round(float(value)) for value in coordinates)
        best[label] = FruitCandidate(
            confidence=confidence,
            bbox_xyxy=(
                x1 + x_offset,
                y1 + y_offset,
                x2 + x_offset,
                y2 + y_offset,
            ),
        )
    return best


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
    ) -> None:
        for name, value in (
            ("banana minimum confidence", banana_minimum_confidence),
            ("banana minimum agreement IoU", banana_minimum_agreement_iou),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if (banana_specialist_model is None) != (
            banana_specialist_class_id is None
        ):
            raise ValueError(
                "banana specialist model and class ID must be configured together"
            )
        self._general_model = general_model
        self._general_class_ids = dict(general_class_ids)
        self._banana_specialist_model = banana_specialist_model
        self._banana_specialist_class_id = banana_specialist_class_id
        self._banana_minimum_confidence = banana_minimum_confidence
        self._banana_minimum_agreement_iou = banana_minimum_agreement_iou

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
        if normalized_target not in self._general_class_ids:
            raise ValueError(f"general model does not support {target_fruit}")
        general_results = self._general_model.predict(
            source=source,
            conf=confidence_floor,
            device=device,
            verbose=False,
        )
        observations = _best_candidates_by_class(
            general_results,
            class_names={value: key for key, value in self._general_class_ids.items()},
            x_offset=x_offset,
            y_offset=y_offset,
        )
        general = observations.get(normalized_target)
        if normalized_target != "banana" or not self.banana_specialist_enabled:
            return RoutedPrediction(
                candidate=general,
                inference_passes=1,
                route={
                    "mode": "general",
                    "triggered": False,
                    "confirmed": general is not None,
                },
                observations=observations,
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
                observations=observations,
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
            observations=observations,
        )
