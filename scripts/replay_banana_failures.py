#!/usr/bin/env python3
"""Replay three minimized Banana reliability failures without robot access."""

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
)


def main() -> int:
    env = dict(os.environ)
    source_path = str(ROOT / "src")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        source_path if not existing else os.pathsep.join((source_path, existing))
    )
    command = [sys.executable, "-m", "pytest", "-q", *REPLAYS]
    print("Non-motion Banana reliability replay", flush=True)
    print("  " + "\n  ".join(REPLAYS), flush=True)
    return subprocess.run(command, cwd=ROOT, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
