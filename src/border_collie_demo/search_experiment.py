"""Per-run search tuning and result analysis behind one small interface."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SearchExperimentTuning:
    """Bounded values that may vary between supervised Demo Runs."""

    search_yaw_rps: float = 0.40
    apple_focus_confidence: float = 0.50
    apple_acquisition_confidence: float = 0.40
    center_confirmations: int = 3
    center_tolerance_ratio: float = 0.08

    def __post_init__(self) -> None:
        numeric = (
            self.search_yaw_rps,
            self.apple_focus_confidence,
            self.apple_acquisition_confidence,
            self.center_tolerance_ratio,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("search experiment values must be finite")
        if not 0.40 <= self.search_yaw_rps <= 0.80:
            raise ValueError("search_yaw_rps must stay within 0.40..0.80 rad/s")
        if not 0.50 <= self.apple_focus_confidence <= 0.70:
            raise ValueError("Apple focus confidence must stay within 0.50..0.70")
        if not 0.40 <= self.apple_acquisition_confidence <= 0.70:
            raise ValueError(
                "Apple acquisition confidence must stay within 0.40..0.70"
            )
        if self.apple_focus_confidence < self.apple_acquisition_confidence:
            raise ValueError(
                "Apple focus confidence must be greater than or equal to "
                "Apple acquisition confidence"
            )
        if not 2 <= self.center_confirmations <= 5:
            raise ValueError("center_confirmations must stay within 2..5 frames")
        if not 0.05 <= self.center_tolerance_ratio <= 0.12:
            raise ValueError(
                "center_tolerance_ratio must stay within 0.05..0.12 frame width"
            )

    @classmethod
    def defaults(cls) -> SearchExperimentTuning:
        return cls(
            search_yaw_rps=float(
                os.environ.get("BORDER_COLLIE_GUIDANCE_SEARCH_YAW_RPS", "0.40")
            ),
            apple_focus_confidence=float(
                os.environ.get("BORDER_COLLIE_APPLE_FOCUS_CONFIDENCE", "0.50")
            ),
            apple_acquisition_confidence=float(
                os.environ.get(
                    "BORDER_COLLIE_APPLE_ACQUISITION_CONFIDENCE", "0.40"
                )
            ),
            center_confirmations=int(
                os.environ.get("BORDER_COLLIE_GUIDANCE_CENTER_CONFIRMATIONS", "3")
            ),
            center_tolerance_ratio=float(
                os.environ.get(
                    "BORDER_COLLIE_GUIDANCE_CENTER_TOLERANCE_RATIO", "0.08"
                )
            ),
        )

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, object] | None,
    ) -> SearchExperimentTuning:
        defaults = cls.defaults().to_dict()
        if values is not None:
            defaults.update(values)
        try:
            return cls(
                search_yaw_rps=float(defaults["search_yaw_rps"]),
                apple_focus_confidence=float(
                    defaults["apple_focus_confidence"]
                ),
                apple_acquisition_confidence=float(
                    defaults["apple_acquisition_confidence"]
                ),
                center_confirmations=int(defaults["center_confirmations"]),
                center_tolerance_ratio=float(defaults["center_tolerance_ratio"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, ValueError) and str(exc).startswith(
                ("search_", "Apple ", "center_")
            ):
                raise
            raise ValueError("invalid search experiment tuning") from exc

    def to_dict(self) -> dict[str, float | int]:
        return {
            "search_yaw_rps": self.search_yaw_rps,
            "apple_focus_confidence": self.apple_focus_confidence,
            "apple_acquisition_confidence": self.apple_acquisition_confidence,
            "center_confirmations": self.center_confirmations,
            "center_tolerance_ratio": self.center_tolerance_ratio,
        }


def confidence_summary(
    samples: list[dict[str, object]],
) -> dict[str, object]:
    confidences = [
        float(value)
        for sample in samples
        if isinstance((value := sample.get("confidence")), (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    ]
    lock_confidence = next(
        (
            sample.get("confidence")
            for sample in reversed(samples)
            if sample.get("locked") is True
            and isinstance(sample.get("confidence"), (int, float))
        ),
        None,
    )
    return {
        "detected_frames": len(confidences),
        "minimum": min(confidences) if confidences else None,
        "maximum": max(confidences) if confidences else None,
        "average": sum(confidences) / len(confidences) if confidences else None,
        "lock_confidence": lock_confidence,
    }


def search_experiment_scorecard(runs: list[dict[str, Any]]) -> dict[str, object]:
    cohorts: dict[tuple[str, float], dict[str, object]] = {}
    for run in runs:
        tuning = run.get("search_experiment")
        if not isinstance(tuning, dict):
            continue
        yaw = tuning.get("search_yaw_rps")
        fruit = run.get("target_fruit")
        if not isinstance(yaw, (int, float)) or not isinstance(fruit, str):
            continue
        key = (fruit, float(yaw))
        cohort = cohorts.setdefault(
            key,
            {
                "target_fruit": fruit,
                "search_yaw_rps": float(yaw),
                "runs": 0,
                "locks": 0,
                "successes": 0,
            },
        )
        cohort["runs"] = int(cohort["runs"]) + 1
        stage_results = run.get("stage_results")
        locked = isinstance(stage_results, dict) and isinstance(
            stage_results.get("turn_to_fruit"), dict
        )
        if locked:
            cohort["locks"] = int(cohort["locks"]) + 1
        if run.get("outcome") == "COMPLETED" and run.get("reason") == "SUCCESS":
            cohort["successes"] = int(cohort["successes"]) + 1
    values = []
    for cohort in cohorts.values():
        count = int(cohort["runs"])
        cohort["lock_rate_percent"] = round(100.0 * int(cohort["locks"]) / count, 1)
        cohort["success_rate_percent"] = round(
            100.0 * int(cohort["successes"]) / count,
            1,
        )
        values.append(cohort)
    return {
        "cohorts": sorted(
            values,
            key=lambda item: (str(item["target_fruit"]), item["search_yaw_rps"]),
        )
    }
