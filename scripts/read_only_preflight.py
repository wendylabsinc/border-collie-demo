"""Read-only live proof before or after a Border Collie deployment."""

from __future__ import annotations

import argparse
import json
import time
from typing import Any
from urllib.request import urlopen


def _json(url: str) -> dict[str, Any]:
    with urlopen(url, timeout=5.0) as response:
        return json.load(response)


def _zero_motion(status: dict[str, Any]) -> bool:
    motion = ((status.get("hardware") or {}).get("motion") or {})
    command = motion.get("last_command") or {}
    return bool(
        motion.get("armed") is False
        and command.get("forward_mps") == 0.0
        and command.get("yaw_rps") == 0.0
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://woof.local:8110")
    parser.add_argument("--expected-build-label", required=True)
    parser.add_argument(
        "--expected-fruits",
        nargs="+",
        default=["apple", "banana", "pear"],
    )
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    first_status = _json(f"{base}/api/status")
    first_camera = _json(f"{base}/api/fruits/preview")
    time.sleep(0.35)
    status = _json(f"{base}/api/status")
    camera = _json(f"{base}/api/fruits/preview")
    fruits = _json(f"{base}/api/fruits")

    hardware = status.get("hardware") or {}
    pose = hardware.get("pose") or {}
    mission = status.get("mission") or {}
    cohort = status.get("cohort") or {}
    first_source = first_camera.get("source") or {}
    source = camera.get("source") or {}
    checks = {
        "exact_build": status.get("build_label") == args.expected_build_label,
        "production_runtime": status.get("runtime_mode") == "production",
        "qualified_fruits": sorted(fruits.get("qualified_fruits") or [])
        == sorted(args.expected_fruits),
        "hardware_connected": hardware.get("configured") is True
        and hardware.get("clients_initialized") is True
        and hardware.get("connected") is True,
        "no_active_operation": hardware.get("active_operation") is None,
        "no_active_run": status.get("active_run_id") is None,
        "no_running_cohort": cohort.get("status") != "RUNNING",
        "activation_ready": (status.get("activation") or {}).get("ready") is True,
        "no_restart_latch": mission.get("restart_required") is not True,
        "exact_zero_disarm": _zero_motion(first_status) and _zero_motion(status),
        "fresh_pose": pose.get("healthy") is True
        and isinstance(pose.get("age_s"), (int, float))
        and pose["age_s"] <= 0.5,
        "camera_healthy": first_camera.get("camera_healthy") is True
        and camera.get("camera_healthy") is True,
        "camera_generation_stable": first_camera.get("generation")
        == camera.get("generation"),
        "camera_advancing": isinstance(first_source.get("pts"), int)
        and isinstance(source.get("pts"), int)
        and source["pts"] > first_source["pts"],
        "camera_fresh": isinstance(source.get("age_s"), (int, float))
        and source["age_s"] <= 0.35,
    }
    evidence = {
        "passed": all(checks.values()),
        "checks": checks,
        "build_label": status.get("build_label"),
        "qualified_fruits": fruits.get("qualified_fruits"),
        "pose_age_s": pose.get("age_s"),
        "camera_generation": camera.get("generation"),
        "camera_pts": [first_source.get("pts"), source.get("pts")],
    }
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0 if evidence["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
