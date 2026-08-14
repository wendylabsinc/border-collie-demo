from __future__ import annotations

import json
from uuid import uuid4

from border_collie_demo.black_box import RunBlackBox
from border_collie_demo.home_recording import (
    HomeRecordingHub,
    HomeRecordingStatus,
    SharedHomeEventJournal,
)


def test_home_recording_hub_keeps_one_versioned_shared_event_section(tmp_path) -> None:
    run_id = str(uuid4())
    primary = RunBlackBox(tmp_path / "runs")
    journal = SharedHomeEventJournal(tmp_path / "shared")
    hub = HomeRecordingHub(primary, journal)

    hub.record(
        run_id,
        "home_captured",
        phase="capture_home",
        payload={
            "x_m": 1.0,
            "y_m": 2.0,
            "yaw_rad": 0.5,
            "odometry_epoch": "odom-a",
        },
    )
    hub.record(
        run_id,
        "motion_command",
        phase="return_home",
        payload={"forward_mps": 1.0, "yaw_rps": 0.5, "motion_path": "factory"},
    )
    hub.record(
        run_id,
        "guidance_decision",
        phase="approach_fruit",
        payload={"confidence": 0.7},
    )
    hub.close()

    events = [
        json.loads(line)
        for line in (tmp_path / "shared" / f"{run_id}.ndjson")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [event["kind"] for event in events] == [
        "home_captured",
        "motion_command",
    ]
    assert {event["schema_version"] for event in events} == {1}
    assert events[1]["phase"] == "return_home"
    assert primary.read(run_id)[-1]["kind"] == "guidance_decision"


def test_shared_recorder_failure_never_prevents_primary_safety_evidence(tmp_path) -> None:
    run_id = str(uuid4())
    primary = RunBlackBox(tmp_path / "runs")
    # A file where the shared event directory must be makes journal writes fail.
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory", encoding="utf-8")
    hub = HomeRecordingHub(primary, SharedHomeEventJournal(blocked))

    event = hub.record(
        run_id,
        "home_verification_terminal",
        phase="return_home",
        payload={"stable": False, "terminal_reason": "outside_home_gate"},
    )
    hub.close()

    assert event["kind"] == "home_verification_terminal"
    assert primary.read(run_id)[-1]["payload"]["stable"] is False
    assert hub.status()["shared_available"] is False


def test_status_reports_passive_recorder_unavailable_without_raising(
    tmp_path,
    monkeypatch,
) -> None:
    hub = HomeRecordingHub(
        RunBlackBox(tmp_path / "runs"),
        SharedHomeEventJournal(tmp_path / "shared"),
    )

    def unavailable(*_args, **_kwargs):
        raise OSError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", unavailable)
    status = HomeRecordingStatus(
        hub,
        passive_status_url="http://127.0.0.1:8112/status",
    )()
    hub.close()

    assert status["shared_available"] is True
    assert status["passive"]["ready"] is False
    assert "connection refused" in status["passive"]["detail"]
