#!/usr/bin/env python3
"""Replay Banana failures and fruit-bearing contracts without robot access."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPLAYS = (
    "tests/test_guidance.py::test_physical_banana_slow_inference_replay_stops_then_resumes_fresh_motion",
    "tests/test_guidance.py::test_physical_banana_search_replay_retains_fine_focus_through_brief_misses",
    "tests/test_hardware.py::test_near_home_heading_escape_reconciles_from_fresh_post_disarm_position",
    "tests/test_media_perception.py::test_sidecar_publishes_all_fruit_observations_but_keeps_selected_detection",
    "tests/test_guidance.py::test_unselected_all_fruit_observations_cannot_lock_or_authorize_motion",
    "tests/test_fruit_bearing_map.py",
    "tests/test_hardware.py::test_search_records_all_fruits_in_bearing_map_but_selected_target_drives_lock",
    "tests/test_production.py::test_subsequent_run_uses_mapped_shortest_turn_then_normal_camera_guidance",
    "tests/test_production.py::test_map_reuse_aligns_canonical_home_yaw_then_remeasures_before_routing",
    "tests/test_run_tuning.py::test_bearing_routing_default_is_env_backed_and_frozen_per_run",
    "tests/test_run_tuning.py::test_idempotency_includes_the_complete_tuning_snapshot",
)


def main() -> int:
    env = dict(os.environ)
    source_path = str(ROOT / "src")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        source_path if not existing else os.pathsep.join((source_path, existing))
    )
    command = [sys.executable, "-m", "pytest", "-q", *REPLAYS]
    print("Non-motion Banana reliability and fruit-bearing replay", flush=True)
    print("  " + "\n  ".join(REPLAYS), flush=True)
    return subprocess.run(command, cwd=ROOT, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
