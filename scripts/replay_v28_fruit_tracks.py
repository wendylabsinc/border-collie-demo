#!/usr/bin/env python3
"""Replay the five saved v28 runs through the persistent track interface.

The v28 aggregate sampled the API at operator cadence rather than retaining
every detector frame. Exact before counters therefore come from the saved
motion-command traces; the replay is explicitly labelled sparse and must not be
presented as a physical after measurement.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from border_collie_demo.fruit_track_replay import FruitTrackReplay
from border_collie_demo.persistent_fruit_tracker import PersistentFruitTracker


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUTS = (
    ROOT / "benchmarks/results/2026-08-12-v28-random-five.json",
    ROOT / "benchmarks/results/2026-08-12-v28-random-five-run-05.json",
)
DEFAULT_OUTPUT = (
    ROOT
    / "benchmarks/results/2026-08-12-v29-persistent-tracker-offline-replay.json"
)


def _commands(run: dict[str, Any]) -> list[dict[str, Any]]:
    commands = list((run.get("failure_details") or {}).get("motion_commands") or [])
    for stage in (run.get("stage_results") or {}).values():
        commands.extend((stage or {}).get("motion_commands") or [])
    return commands


def _sparse_events(run: dict[str, Any]) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for index, sample in enumerate(run.get("samples") or [], start=1):
        label = sample.get("detection_label")
        confidence = sample.get("confidence")
        bottom = sample.get("bbox_bottom_ratio")
        detection = None
        if isinstance(label, str) and isinstance(confidence, (int, float)):
            bottom_ratio = (
                float(bottom) if isinstance(bottom, (int, float)) else 0.65
            )
            center_y = min(bottom_ratio, max(0.0, bottom_ratio - 0.15))
            detection = {
                "label": label,
                "confidence": float(confidence),
                "bbox_xyxy": [0.40, max(0.0, center_y - 0.10), 0.60, bottom_ratio],
                "center_x_ratio": 0.50,
                "center_y_ratio": center_y,
                "bottom_ratio": bottom_ratio,
                "bbox_area_ratio": max(0.001, 0.20 * (bottom_ratio - center_y + 0.10)),
                "generation": "v28-saved-run",
                "source_pts": index,
                "source_time_base": "operator-sample",
                "age_s": 0.01,
                "route": "legacy_selected_detection",
            }
        events.append(
            {
                "kind": "perception_sample",
                # Preserve ordering while using a controller-like cadence. The
                # aggregate's original 1.5 s polling gaps are not frame cadence.
                "recorded_monotonic_s": index * 0.10,
                "payload": {
                    "phase": str(sample.get("phase") or "unknown"),
                    "target_fruit": run["target_fruit"],
                    "camera_healthy": sample.get("app_error") is None,
                    "generation": "v28-saved-run",
                    "source": {
                        "pts": index,
                        "time_base": "operator-sample",
                        "age_s": 0.01,
                    },
                    "observations": {"full_frame": detection, "crop": None},
                },
            }
        )
    return events


def replay_runs(paths: tuple[Path, ...]) -> dict[str, object]:
    runs: list[dict[str, Any]] = []
    for path in paths:
        runs.extend(json.loads(path.read_text())["runs"])
    results = []
    for run in runs:
        fruit = str(run["target_fruit"])
        baseline = Counter(
            str(command.get("reason") or "unknown") for command in _commands(run)
        )
        replay = FruitTrackReplay(
            PersistentFruitTracker.for_fruit(
                fruit,
                acquisition_confirmations=(3 if fruit == "apple" else 5),
            )
        ).replay(_sparse_events(run))
        results.append(
            {
                "run_id": run["run_id"],
                "target_fruit": fruit,
                "physical_outcome": run["reason"],
                "saved_api_samples": len(run.get("samples") or []),
                "before_command_counts": dict(sorted(baseline.items())),
                "sparse_persistent_replay": {
                    key: value
                    for key, value in replay.to_dict().items()
                    if key != "transitions"
                },
            }
        )
    return {
        "schema_version": 1,
        "release_candidate": "stage-camera-v29-persistent-fruit-track",
        "scope": "offline sparse replay of all five saved v28 physical records",
        "limitations": [
            "v28 retained operator-cadence API samples, not every detector frame",
            "exact after command counts require one new v29 physical run",
            "sparse replay cannot reconstruct crop/full-frame route separation that v28 did not record",
        ],
        "baseline_highlights": {
            "apple_tracking_confidence_low_zero_commands": 116,
            "pear_center_corridor_recenter_commands": 130,
            "banana_detection_missing_zero_commands": 6,
            "pear_candidate_alignment_hold_losses": 27,
        },
        "runs": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = replay_runs(DEFAULT_INPUTS)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
