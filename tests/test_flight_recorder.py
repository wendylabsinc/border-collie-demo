from __future__ import annotations

import json

from border_collie_demo.flight_recorder import FlightRecorder, terminal_evidence_bundle


def test_flight_recorder_survives_restart_and_continues_hash_chain(tmp_path) -> None:
    recorder = FlightRecorder(tmp_path, segment_max_bytes=4096)
    first = recorder.record("run_activated", {"fruit": "pear"}, run_id="run-1")

    restarted = FlightRecorder(tmp_path, segment_max_bytes=4096)
    second = restarted.record("motion_command", {"forward_mps": 0.5}, run_id="run-1")

    assert second["sequence"] == first["sequence"] + 1
    assert second["previous_event_sha256"] == first["event_sha256"]


def test_flight_recorder_rotates_with_bounded_segments(tmp_path) -> None:
    recorder = FlightRecorder(
        tmp_path,
        segment_max_bytes=1024,
        maximum_segments=2,
    )
    for number in range(30):
        recorder.record("sample", {"number": number, "padding": "x" * 200})

    assert len(list(tmp_path.glob("segment-*.ndjson"))) <= 2
    assert recorder.status()["segments"] <= 3  # two rotated plus active


def test_terminal_bundle_keeps_flight_data_when_media_capture_fails(tmp_path) -> None:
    recorder = FlightRecorder(tmp_path)
    recorder.record("failure", {"reason": "CAMERA_FAILURE"}, run_id="run-1")

    def failed_media():
        raise TimeoutError("sidecar unavailable")

    artifacts = terminal_evidence_bundle(recorder, failed_media)
    names = {artifact.filename for artifact in artifacts}
    warning = next(
        artifact for artifact in artifacts if artifact.filename.endswith("warnings.json")
    )

    assert "flight-recorder.ndjson" in names
    assert "sidecar unavailable" in json.loads(warning.content)["warnings"][0]
