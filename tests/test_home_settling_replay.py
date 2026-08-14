from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_saved_physical_home_replays_and_stable_control() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/replay_home_settling.py")],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["passed"] is True
    by_run = {item["run_id"]: item for item in report["replays"]}
    assert by_run["3465b41f-c05c-45bd-a6e6-55526e39b9dc"]["observed_stable"] is False
    assert by_run["7fc8ef15-prior-inside-then-outside"]["observed_stable"] is False
    assert by_run["stable-control"]["observed_stable"] is True
