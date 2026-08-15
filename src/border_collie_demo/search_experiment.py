"""Per-run search tuning and result analysis behind one small interface."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .fruits import fruit_policy

_CONFIDENCE_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "apple": {
        "focus_confidence": (0.40, 0.70),
        "lock_confidence": (0.40, 0.70),
    },
    "banana": {
        "focus_confidence": (0.20, 0.70),
        "lock_confidence": (0.20, 0.70),
    },
    "pear": {
        "focus_confidence": (0.65, 0.85),
        "lock_confidence": (0.65, 0.85),
    },
}

_EVIDENCE_PATH = Path(__file__).with_name("search_experiment_evidence.json")


def search_experiment_evidence() -> dict[str, object]:
    """Return the checked-in physical experiment index shown in the UI."""

    return deepcopy(json.loads(_EVIDENCE_PATH.read_text(encoding="utf-8")))


def search_experiment_contract() -> dict[str, object]:
    """Return UI-safe defaults and bounds for every selectable Target Fruit."""

    fruits: dict[str, object] = {}
    for fruit, ranges in _CONFIDENCE_RANGES.items():
        policy = fruit_policy(fruit)
        fruits[fruit] = {
            "defaults": {
                "focus_confidence": (
                    policy.focus_confidence
                    if policy.focus_confidence is not None
                    else policy.acquisition_confidence
                ),
                "lock_confidence": policy.acquisition_confidence,
            },
            "ranges": {
                name: [minimum, maximum]
                for name, (minimum, maximum) in ranges.items()
            },
        }
    defaults = SearchExperimentTuning.defaults("pear")
    return {
        "defaults": {
            "search_yaw_rps": defaults.search_yaw_rps,
            "center_confirmations": defaults.center_confirmations,
            "center_tolerance_ratio": defaults.center_tolerance_ratio,
        },
        "ranges": {
            "search_yaw_rps": [0.40, 0.80],
            "center_confirmations": [2, 5],
            "center_tolerance_ratio": [0.05, 0.12],
        },
        "fruits": fruits,
    }


@dataclass(frozen=True)
class SearchExperimentTuning:
    """Bounded values that may vary between supervised Demo Runs."""

    target_fruit: str
    search_yaw_rps: float = 0.40
    focus_confidence: float | None = None
    lock_confidence: float | None = None
    center_confirmations: int = 3
    center_tolerance_ratio: float = 0.05

    def __post_init__(self) -> None:
        target = self.target_fruit.casefold().strip()
        if target not in _CONFIDENCE_RANGES:
            raise ValueError(f"unsupported Target Fruit: {self.target_fruit}")
        object.__setattr__(self, "target_fruit", target)

        policy = fruit_policy(target)
        if self.lock_confidence is None:
            object.__setattr__(
                self,
                "lock_confidence",
                policy.acquisition_confidence,
            )
        if self.focus_confidence is None and policy.focus_confidence is not None:
            object.__setattr__(
                self,
                "focus_confidence",
                policy.focus_confidence,
            )

        assert self.lock_confidence is not None
        numeric = (
            self.search_yaw_rps,
            self.lock_confidence,
            self.center_tolerance_ratio,
        )
        if self.focus_confidence is not None:
            numeric += (self.focus_confidence,)
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("search experiment values must be finite")
        if not 0.40 <= self.search_yaw_rps <= 0.80:
            raise ValueError("search_yaw_rps must stay within 0.40..0.80 rad/s")

        ranges = _CONFIDENCE_RANGES[target]
        lock_minimum, lock_maximum = ranges["lock_confidence"]
        if not lock_minimum <= self.lock_confidence <= lock_maximum:
            raise ValueError(
                f"{target.title()} lock confidence must stay within "
                f"{lock_minimum:.2f}..{lock_maximum:.2f}"
            )
        if self.focus_confidence is not None:
            focus_minimum, focus_maximum = ranges["focus_confidence"]
            if not focus_minimum <= self.focus_confidence <= focus_maximum:
                raise ValueError(
                    f"{target.title()} focus confidence must stay within "
                    f"{focus_minimum:.2f}..{focus_maximum:.2f}"
                )
            if self.focus_confidence < self.lock_confidence:
                raise ValueError(
                    f"{target.title()} focus confidence must be greater than or "
                    "equal to lock confidence"
                )
        if not 2 <= self.center_confirmations <= 5:
            raise ValueError("center_confirmations must stay within 2..5 frames")
        if not 0.05 <= self.center_tolerance_ratio <= 0.12:
            raise ValueError(
                "center_tolerance_ratio must stay within 0.05..0.12 frame width"
            )

    @classmethod
    def defaults(cls, target_fruit: str) -> SearchExperimentTuning:
        policy = fruit_policy(target_fruit)
        return cls(
            target_fruit=target_fruit,
            search_yaw_rps=float(
                os.environ.get("BORDER_COLLIE_GUIDANCE_SEARCH_YAW_RPS", "0.40")
            ),
            focus_confidence=policy.focus_confidence,
            lock_confidence=policy.acquisition_confidence,
            center_confirmations=int(
                os.environ.get("BORDER_COLLIE_GUIDANCE_CENTER_CONFIRMATIONS", "3")
            ),
            center_tolerance_ratio=float(
                os.environ.get(
                    "BORDER_COLLIE_GUIDANCE_CENTER_TOLERANCE_RATIO", "0.05"
                )
            ),
        )

    @classmethod
    def from_mapping(
        cls,
        target_fruit: str,
        values: Mapping[str, object] | None,
    ) -> SearchExperimentTuning:
        defaults = cls.defaults(target_fruit).to_dict()
        if values is not None:
            persisted_target = values.get("target_fruit")
            if (
                persisted_target is not None
                and str(persisted_target).casefold().strip()
                != target_fruit.casefold().strip()
            ):
                raise ValueError(
                    "search experiment Target Fruit does not match the Demo Run"
                )
            defaults.update(values)
        try:
            raw_focus = defaults["focus_confidence"]
            return cls(
                target_fruit=target_fruit,
                search_yaw_rps=float(defaults["search_yaw_rps"]),
                focus_confidence=(
                    None if raw_focus is None else float(raw_focus)
                ),
                lock_confidence=float(defaults["lock_confidence"]),
                center_confirmations=int(defaults["center_confirmations"]),
                center_tolerance_ratio=float(defaults["center_tolerance_ratio"]),
            )
        except (KeyError, TypeError) as exc:
            raise ValueError("invalid search experiment tuning") from exc

    def to_dict(self) -> dict[str, object]:
        return {
            "target_fruit": self.target_fruit,
            "search_yaw_rps": self.search_yaw_rps,
            "focus_confidence": self.focus_confidence,
            "lock_confidence": self.lock_confidence,
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
    cohorts: dict[tuple[str, float, str], dict[str, object]] = {}
    non_terminal_runs = 0
    for run in runs:
        tuning = run.get("search_experiment")
        if not isinstance(tuning, dict):
            continue
        yaw = tuning.get("search_yaw_rps")
        fruit = run.get("target_fruit")
        tuning_fruit = tuning.get("target_fruit")
        if (
            not isinstance(yaw, (int, float))
            or not isinstance(fruit, str)
            or (tuning_fruit is not None and tuning_fruit != fruit)
        ):
            continue
        outcome = run.get("outcome")
        if outcome is None:
            non_terminal_runs += 1
            continue
        run_tuning = run.get("run_tuning")
        search_tuning = (
            run_tuning.get("search") if isinstance(run_tuning, dict) else None
        )
        routing_value = (
            search_tuning.get("bearing_routing_enabled")
            if isinstance(search_tuning, dict)
            else None
        )
        routing = (
            "ON"
            if routing_value is True
            else "OFF"
            if routing_value is False
            else "NOT_RECORDED"
        )
        key = (fruit, float(yaw), routing)
        cohort = cohorts.setdefault(
            key,
            {
                "target_fruit": fruit,
                "search_yaw_rps": float(yaw),
                "bearing_routing": routing,
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
    evidence = search_experiment_evidence()
    return {
        "schema_version": 2,
        "note": (
            "Only terminal runs are counted. Routing is NOT_RECORDED for runs "
            "created before the per-run routing flag was persisted."
        ),
        "non_terminal_runs_excluded": non_terminal_runs,
        "cohorts": sorted(
            values,
            key=lambda item: (
                str(item["target_fruit"]),
                item["search_yaw_rps"],
                str(item["bearing_routing"]),
            ),
        ),
        "experiments": evidence["experiments"],
    }
