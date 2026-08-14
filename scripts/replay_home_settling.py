#!/usr/bin/env python3
"""Replay saved post-stop Home sample windows without robot access."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from border_collie_demo.home_stability import (
    HomeSample,
    HomeStabilityConfig,
    HomeStabilityWindow,
    HomeVerificationError,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURES = ROOT / "benchmarks" / "replays" / "home-settling"


def replay(path: Path) -> dict[str, Any]:
    fixture = json.loads(path.read_text(encoding="utf-8"))
    epoch = str(fixture["odometry_epoch"])
    window = HomeStabilityWindow(
        HomeStabilityConfig(), expected_odometry_epoch=epoch
    )
    decision = None
    error = None
    try:
        for sequence, raw in enumerate(fixture["samples"], 1):
            decision = window.observe(
                HomeSample(
                    sequence=sequence,
                    x_m=float(raw["distance_m"]),
                    y_m=0.0,
                    yaw_rad=0.0,
                    captured_monotonic_s=float(raw["captured_monotonic_s"]),
                    age_s=float(raw["age_s"]),
                    odometry_epoch=epoch,
                )
            )
    except (HomeVerificationError, TypeError, ValueError) as exc:
        error = str(exc)
    observed_stable = bool(decision is not None and decision.stable and error is None)
    expected_stable = bool(fixture["expected_stable"])
    return {
        "fixture": path.name,
        "run_id": fixture["run_id"],
        "expected_stable": expected_stable,
        "observed_stable": observed_stable,
        "passed": observed_stable is expected_stable,
        "decision": decision.to_dict() if decision is not None else None,
        "error": error,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "fixtures",
        nargs="*",
        type=Path,
        help="JSON replay fixtures; defaults to all saved Home windows",
    )
    args = parser.parse_args(argv)
    paths = args.fixtures or sorted(DEFAULT_FIXTURES.glob("*.json"))
    results = [replay(path.resolve()) for path in paths]
    passed = bool(results) and all(result["passed"] for result in results)
    print(json.dumps({"passed": passed, "replays": results}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
