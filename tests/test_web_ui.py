from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from border_collie_demo.run_tuning import RunTuning

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "tests" / "web_ui_browser_harness.mjs"
CHROME = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")


def test_audience_ui_run_and_cohort_controls_in_a_real_browser(tmp_path: Path) -> None:
    """Drive the public DOM and HTTP seams without connecting to Woof."""

    node = shutil.which("node")
    if node is None or not CHROME.exists():
        pytest.skip("the local headless-browser acceptance runtime is unavailable")
    contract = tmp_path / "run-tuning-contract.json"
    contract.write_text(json.dumps(RunTuning.contract()), encoding="utf-8")
    environment = dict(os.environ)
    environment["CHROME_PATH"] = str(CHROME)

    completed = subprocess.run(
        [
            node,
            str(HARNESS),
            str(ROOT / "web" / "index.html"),
            str(contract),
        ],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, "\n".join(
        part for part in (completed.stdout, completed.stderr) if part
    )
    assert "browser UI contract: PASS" in completed.stdout
