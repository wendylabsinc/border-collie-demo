import asyncio
import time
from time import monotonic

import pytest
from fastapi.testclient import TestClient

from border_collie_demo.api import create_app
from border_collie_demo.evidence import EvidenceArtifact
from border_collie_demo.mission import MissionMachine
from border_collie_demo.models import MissionPhase, RemoteInput
from border_collie_demo.orchestrator import SimulatedStageExecutor, StageFailure
from border_collie_demo.recovery import RECOVERY_CONFIRMATION


class ReadyHardwareBoundary:
    def __init__(self) -> None:
        self.stop_calls = 0

    async def start(self) -> None:
        pass

    async def close(self) -> list[str]:
        return []

    async def emergency_stop(self) -> list[str]:
        self.stop_calls += 1
        return []

    def capture_home(self) -> dict[str, object]:
        return {
            "x_m": 1.25,
            "y_m": -0.5,
            "yaw_rad": 0.75,
            "captured_monotonic_s": 123.0,
            "age_s": 0.04,
            "source": "rt/sportmodestate",
        }

    def status(self) -> dict[str, object]:
        return {
            "configured": True,
            "autonomy_enabled": True,
            "connected": True,
            "fault": None,
            "active_operation": None,
            "pose": {
                "healthy": True,
                "age_s": 0.04,
                "error": None,
                "pose": {"x_m": 1.25, "y_m": -0.42, "yaw_rad": 0.75},
            },
            "motion": {"initialized": True, "armed": False, "fault": None},
        }


class RecoveryHardwareBoundary(ReadyHardwareBoundary):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, dict[str, object], dict[str, object]]] = []
        self.trace_phase: str | None = None

    def start_motion_trace(self, phase: str) -> None:
        self.trace_phase = phase

    def motion_trace(self) -> list[dict[str, object]]:
        return [
            {
                "sequence": 1,
                "phase": self.trace_phase,
                "forward_mps": (
                    1.0 if self.trace_phase == "recovery_return_home" else 0.0
                ),
                "yaw_rps": 0.0,
                "reason": self.trace_phase,
            }
        ]

    async def turn_toward_home(self, home, **options):
        self.calls.append(("turn_toward_home", home, options))
        return {
            "home_distance_m": 1.4,
            "home_bearing_error_rad": 0.03,
            "motion_commands_sent": True,
        }

    async def return_home(self, home, **options):
        self.calls.append(("return_home", home, options))
        return {
            "home_distance_m": 0.08,
            "requested_forward_pulses": options["forward_pulse_count"],
            "replayed_forward_pulses": 4,
            "motion_commands_sent": True,
        }


def ready_camera_perception() -> dict[str, object]:
    return {
        "ready": True,
        "detail": "camera generation and pear detector passed preflight",
    }


def healthy_camera_without_pear() -> dict[str, object]:
    return {
        "ready": False,
        "camera_healthy": True,
        "target_ready": False,
        "detail": "qualifying pear detection is missing",
    }


def broken_camera_perception() -> dict[str, object]:
    raise RuntimeError("camera process unavailable")


def ready_app(runs_root):
    return create_app(
        runs_root=runs_root,
        hardware=ReadyHardwareBoundary(),
        camera_perception_status=ready_camera_perception,
    )


class PoseLostAtHomeBoundary(ReadyHardwareBoundary):
    def capture_home(self) -> dict[str, object]:
        raise RuntimeError("pose sample became stale")


def test_audience_page_includes_the_annotated_camera_feed() -> None:
    response = TestClient(create_app()).get("/")

    assert response.status_code == 200
    assert 'id="camera-feed"' in response.text
    assert "'/api/camera/frame.jpg'" in response.text
    assert ":8111/api/camera/frame.jpg" not in response.text
    assert "YOLO fruit model overlay" in response.text
    assert 'id="target-fruit"' in response.text
    assert '<option value="apple">Red apple</option>' in response.text
    assert "target_fruit: targetFruit.value" in response.text


def test_camera_preview_is_proxied_through_the_main_app() -> None:
    jpeg = b"\xff\xd8apple-preview\xff\xd9"

    response = TestClient(create_app(camera_frame=lambda: jpeg)).get(
        "/api/camera/frame.jpg"
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.headers["cache-control"] == "no-store"
    assert response.content == jpeg


def test_fruit_test_page_can_select_supported_fruit_without_motion() -> None:
    selected: list[str] = []
    app = create_app(
        select_perception_target=lambda fruit: (
            selected.append(fruit)
            or {
                "target_fruit": fruit,
                "supported_fruits": ["apple", "banana", "pear"],
            }
        )
    )

    with TestClient(app) as client:
        page = client.get("/fruit-test")
        response = client.post(
            "/api/fruits/preview",
            json={"target_fruit": "apple"},
        )

    assert page.status_code == 200
    assert "Camera only — Woof will not move" in page.text
    assert "const frameUrl = '/api/camera/frame.jpg';" in page.text
    assert ":8111/api/camera/frame.jpg" not in page.text
    assert "document.hidden" in page.text
    assert "scheduleRefresh(1500)" in page.text
    assert response.status_code == 200
    assert response.json() == {
        "target_fruit": "apple",
        "qualified_for_demo": True,
        "supported_fruits": ["apple", "banana", "pear"],
    }
    assert selected == ["apple"]


def test_fruit_list_qualifies_red_apple_pear_and_specialist_banana() -> None:
    response = TestClient(create_app()).get("/api/fruits")

    assert response.status_code == 200
    assert response.json() == {
        "supported_fruits": ["apple", "banana", "pear"],
        "qualified_fruits": ["apple", "banana", "pear"],
    }


def test_debug_page_offers_recorded_evidence_for_fieldmark_labeling() -> None:
    response = TestClient(create_app()).get("/debug")

    assert response.status_code == 200
    assert 'id="run-artifacts"' in response.text
    assert "Label raw frames in Fieldmark" in response.text
    assert "/api/results/" in response.text


def test_status_is_explicitly_non_operational() -> None:
    response = TestClient(create_app()).get("/api/status")

    assert response.status_code == 200
    body = response.json()
    assert body["hardware"]["configured"] is False
    assert body["hardware"]["connected"] is False
    assert body["hardware"]["can_pulse_forward"] is False
    assert body["activation"] == {
        "ready": False,
        "blockers": [
            {
                "name": "hardware_connected",
                "detail": "Go2 hardware is not connected",
            },
            {
                "name": "autonomy_enabled",
                "detail": "autonomous demo motion is disabled",
            },
            {
                "name": "fresh_pose",
                "detail": "fresh Go2 pose is unavailable",
            },
                {
                    "name": "camera_perception_ready",
                    "detail": "production camera/perception adapter is not connected",
                },
                {
                    "name": "metric_arrival_calibrated",
                    "detail": "stationary forward range calibration is required",
                },
        ],
    }


def test_activate_fails_closed_when_preflight_is_not_ready(tmp_path) -> None:
    with TestClient(create_app(runs_root=tmp_path)) as client:
        response = client.post(
            "/api/run",
            json={"target_fruit": "pear", "orientation_degrees": 137},
        )

        assert response.status_code == 201
        created = response.json()["run"]
        assert created["run_id"]
        assert created["target_fruit"] == "pear"
        assert created["outcome"] == "FAILED"
        assert created["reason"] == "PREFLIGHT_FAILURE"
        assert created["current_phase"] == "failed"
        assert created["failed_phase"] == "preflight"
        assert created["final_safety_state"] == "DISARMED_CONFIRMED"
        assert created["preflight"]["ready"] is False
        assert {
            check["name"]: check["ready"] for check in created["preflight"]["checks"]
        } == {
            "durable_run_storage": True,
            "hardware_connected": False,
            "autonomy_enabled": False,
            "fresh_pose": False,
            "motion_disarmed": True,
                "camera_perception_ready": False,
                "metric_arrival_calibrated": False,
            }
        assert client.get("/api/status").json()["active_run_id"] is None

        readback = client.get(f"/api/results/{created['run_id']}")
        assert readback.status_code == 200
        assert readback.json()["run"] == created


def test_activate_rejects_a_second_active_demo_run(tmp_path) -> None:
    with TestClient(
        create_app(
            runs_root=tmp_path,
            hardware=ReadyHardwareBoundary(),
            camera_perception_status=ready_camera_perception,
        )
    ) as client:
        first = client.post("/api/run", json={"target_fruit": "pear"})
        second = client.post("/api/run", json={"target_fruit": "pear"})

        assert first.status_code == 201
        assert first.json()["run"]["preflight"]["ready"] is True
        assert first.json()["run"]["preflight"]["camera_perception"] == (
            ready_camera_perception()
        )
        run = first.json()["run"]
        assert run["current_phase"] == "wait_for_command"
        assert run["home"] == {
            "x_m": 1.25,
            "y_m": -0.5,
            "yaw_rad": 0.75,
            "captured_monotonic_s": 123.0,
            "age_s": 0.04,
            "source": "rt/sportmodestate",
        }
        assert [event["reason"] for event in run["events"]][-3:] == [
            "CAPTURE_HOME_STARTED",
            "HOME_CAPTURED",
            "WAITING_FOR_COMMAND",
        ]
        assert second.status_code == 409
        assert second.json()["detail"] == "a Demo Run is already active"


def test_activate_reuses_an_idempotency_key_without_starting_twice(tmp_path) -> None:
    with TestClient(
        create_app(
            runs_root=tmp_path,
            hardware=ReadyHardwareBoundary(),
            camera_perception_status=ready_camera_perception,
        )
    ) as client:
        payload = {"target_fruit": "pear", "idempotency_key": "request-123"}
        first = client.post("/api/run", json=payload)
        second = client.post("/api/run", json=payload)

    assert first.status_code == 201
    assert second.status_code == 201
    assert second.json()["activation_reused"] is True
    assert second.json()["run"]["run_id"] == first.json()["run"]["run_id"]


def test_activation_allows_a_healthy_camera_before_pear_is_visible(tmp_path) -> None:
    with TestClient(
        create_app(
            runs_root=tmp_path,
            hardware=ReadyHardwareBoundary(),
            camera_perception_status=healthy_camera_without_pear,
        )
    ) as client:
        status = client.get("/api/status").json()

        assert status["activation"] == {"ready": True, "blockers": []}

        response = client.post("/api/run", json={"target_fruit": "pear"})
        run = response.json()["run"]
        camera_check = next(
            check
            for check in run["preflight"]["checks"]
            if check["name"] == "camera_perception_ready"
        )
        assert response.status_code == 201
        assert run["preflight"]["ready"] is True
        assert run["current_phase"] == "wait_for_command"
        assert camera_check == {
            "name": "camera_perception_ready",
            "ready": True,
            "detail": "camera source is healthy and advancing",
        }


def test_activate_demo_completes_every_stage_with_simulated_adapters(tmp_path) -> None:
    with TestClient(
        create_app(
            runs_root=tmp_path,
            hardware=ReadyHardwareBoundary(),
            camera_perception_status=ready_camera_perception,
            stage_executor=SimulatedStageExecutor(),
        )
    ) as client:
        response = client.post(
            "/api/run",
            json={"target_fruit": "pear", "orientation_degrees": 137},
        )
        run_id = response.json()["run"]["run_id"]

        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            run = client.get(f"/api/results/{run_id}").json()["run"]
            if run["outcome"] is not None:
                break
            time.sleep(0.01)

        assert run["outcome"] == "COMPLETED"
        assert run["reason"] == "SUCCESS"
        assert run["current_phase"] == "complete"
        assert run["final_safety_state"] == "DISARMED_CONFIRMED"
        assert run["record_type"] == "success_summary"
        assert run["key_values"]["completed_stages"] == [
            "orient_for_run",
            "turn_to_fruit",
            "find_fruit",
            "approach_fruit",
            "sit_and_bark",
            "stand",
            "turn_toward_home",
            "return_home",
            "restore_heading",
        ]
        assert run["orientation_degrees"] == 137.0
        assert run["key_values"]["measured_orientation_change_rad"] == pytest.approx(
            2.391101
        )
        assert run["key_values"]["home_distance_m"] == 0.08
        assert run["key_values"]["outbound_forward_pulses"] == 7
        assert run["key_values"]["final_push_mps"] == 0.55
        assert run["key_values"]["close_range_mps"] == 0.55
        assert run["key_values"]["final_push_duration_s"] == 1.0
        assert run["key_values"]["bark_played"] is True
        assert run["key_values"]["requested_return_pulses"] == 7
        assert run["key_values"]["replayed_return_pulses"] == 7
        assert run["key_values"]["heading_error_rad"] == 0.04
        assert "stage_results" not in run
        assert "events" not in run
        assert sorted(path.name for path in (tmp_path / run_id).iterdir()) == [
            "result.json"
        ]


def test_demo_run_carries_outbound_forward_pulses_into_return_playback(
    tmp_path,
) -> None:
    class PulsePlaybackStages(SimulatedStageExecutor):
        async def execute(self, phase, context):
            if phase is MissionPhase.APPROACH_FRUIT:
                return {
                    "arrival_confirmed": True,
                    "forward_pulse_count": 7,
                    "motion_commands_sent": False,
                }
            if phase is MissionPhase.RETURN_HOME:
                return {
                    "requested_forward_pulses": context.outbound_forward_pulses,
                    "replayed_forward_pulses": context.outbound_forward_pulses,
                    "home_distance_m": 0.08,
                    "motion_commands_sent": False,
                }
            return await super().execute(phase, context)

    with TestClient(
        create_app(
            runs_root=tmp_path,
            hardware=ReadyHardwareBoundary(),
            camera_perception_status=ready_camera_perception,
            stage_executor=PulsePlaybackStages(),
        )
    ) as client:
        run_id = client.post("/api/run", json={"target_fruit": "pear"}).json()["run"][
            "run_id"
        ]
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            run = client.get(f"/api/results/{run_id}").json()["run"]
            if run["outcome"] is not None:
                break
            time.sleep(0.01)

    assert run["outcome"] == "COMPLETED"
    assert run["key_values"]["outbound_forward_pulses"] == 7
    assert run["key_values"]["requested_return_pulses"] == 7
    assert run["key_values"]["replayed_return_pulses"] == 7


def test_success_does_not_capture_terminal_frame_archive(tmp_path) -> None:
    captures = 0

    def capture_terminal_evidence() -> list[EvidenceArtifact]:
        nonlocal captures
        captures += 1
        return []

    with TestClient(
        create_app(
            runs_root=tmp_path,
            hardware=ReadyHardwareBoundary(),
            camera_perception_status=ready_camera_perception,
            stage_executor=SimulatedStageExecutor(),
            terminal_evidence=capture_terminal_evidence,
        )
    ) as client:
        run_id = client.post("/api/run", json={"target_fruit": "pear"}).json()[
            "run"
        ]["run_id"]
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            run = client.get(f"/api/results/{run_id}").json()["run"]
            if run["outcome"] is not None:
                break
            time.sleep(0.01)

    assert captures == 0
    assert run["record_type"] == "success_summary"


def test_camera_failure_identifies_find_fruit_as_the_broken_stage(tmp_path) -> None:
    with TestClient(
        create_app(
            runs_root=tmp_path,
            hardware=ReadyHardwareBoundary(),
            camera_perception_status=ready_camera_perception,
            stage_executor=SimulatedStageExecutor(
                fail_at="find_fruit",
                failure_reason="CAMERA_FAILURE",
                failure_message="source PTS stopped advancing",
            ),
        )
    ) as client:
        started = client.post("/api/run", json={"target_fruit": "pear"}).json()["run"]
        run_id = started["run_id"]

        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            run = client.get(f"/api/results/{run_id}").json()["run"]
            if run["outcome"] is not None:
                break
            time.sleep(0.01)

        assert run["outcome"] == "FAILED"
        assert run["reason"] == "CAMERA_FAILURE"
        assert run["failed_phase"] == "find_fruit"
        assert run["final_safety_state"] == "DISARMED_CONFIRMED"
        assert run["message"] == "source PTS stopped advancing"
        assert list(run["stage_results"]) == ["orient_for_run", "turn_to_fruit"]


def test_failed_search_persists_downloadable_fieldmark_evidence(tmp_path) -> None:
    class DistantPearStages(SimulatedStageExecutor):
        async def execute(self, phase, context):
            if phase is MissionPhase.TURN_TO_FRUIT:
                raise StageFailure(
                    "TARGET_RECOGNITION_FAILURE",
                    "pear was not found in the bounded search sweep",
                    details={
                        "recognition": {
                            "samples": 42,
                            "pear_candidate_samples": 27,
                            "maximum_confidence": 0.019,
                            "maximum_bbox_area_ratio": 0.001,
                        }
                    },
                )
            return await super().execute(phase, context)

    def capture_terminal_evidence() -> list[EvidenceArtifact]:
        return [
            EvidenceArtifact(
                filename="evidence.zip",
                content_type="application/zip",
                content=b"PK\x03\x04raw-fieldmark-frames",
            ),
            EvidenceArtifact(
                filename="terminal.jpg",
                content_type="image/jpeg",
                content=b"\xff\xd8annotated-terminal\xff\xd9",
            ),
        ]

    with TestClient(
        create_app(
            runs_root=tmp_path,
            hardware=ReadyHardwareBoundary(),
            camera_perception_status=ready_camera_perception,
            stage_executor=DistantPearStages(),
            terminal_evidence=capture_terminal_evidence,
        )
    ) as client:
        run_id = client.post("/api/run", json={"target_fruit": "pear"}).json()["run"][
            "run_id"
        ]
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            run = client.get(f"/api/results/{run_id}").json()["run"]
            if run["outcome"] is not None:
                break
            time.sleep(0.01)

        archive = client.get(f"/api/results/{run_id}/artifacts/evidence.zip")
        unreferenced = client.get(f"/api/results/{run_id}/artifacts/not-recorded.zip")
        diagnostic = client.get("/api/diagnostics/stages").json()

    assert run["reason"] == "TARGET_RECOGNITION_FAILURE"
    assert run["failure_details"] == {
        "recognition": {
            "samples": 42,
            "pear_candidate_samples": 27,
            "maximum_confidence": 0.019,
            "maximum_bbox_area_ratio": 0.001,
        }
    }
    assert [artifact["filename"] for artifact in run["artifacts"]] == [
        "flight-recorder.ndjson",
        "evidence.zip",
        "terminal.jpg",
    ]
    assert archive.status_code == 200
    assert archive.headers["content-type"] == "application/zip"
    assert archive.content == b"PK\x03\x04raw-fieldmark-frames"
    assert unreferenced.status_code == 404
    assert [
        artifact["filename"] for artifact in diagnostic["latest_run"]["artifacts"]
    ] == ["flight-recorder.ndjson", "evidence.zip", "terminal.jpg"]


def test_failed_run_recovery_uses_saved_home_and_failed_approach_trace(
    tmp_path,
) -> None:
    class FailedApproachStages(SimulatedStageExecutor):
        async def execute(self, phase, context):
            if phase is MissionPhase.APPROACH_FRUIT:
                raise StageFailure(
                    "ARRIVAL_FAILURE",
                    "qualified pear Arrival timed out",
                    details={
                        "motion_commands": [
                            {
                                "sequence": sequence,
                                "phase": "approach_fruit",
                                "forward_mps": 1.0,
                                "yaw_rps": 0.0,
                                "reason": "approach_target",
                            }
                            for sequence in range(1, 4)
                        ]
                    },
                )
            return await super().execute(phase, context)

    hardware = RecoveryHardwareBoundary()
    with TestClient(
        create_app(
            runs_root=tmp_path,
            hardware=hardware,
            camera_perception_status=ready_camera_perception,
            stage_executor=FailedApproachStages(),
        )
    ) as client:
        run_id = client.post("/api/run", json={"target_fruit": "pear"}).json()[
            "run"
        ]["run_id"]
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            failed = client.get(f"/api/results/{run_id}").json()["run"]
            if failed["outcome"] is not None:
                break
            time.sleep(0.01)

        wrong = client.post(
            f"/api/results/{run_id}/recover-home",
            json={"confirmation": "recover"},
        )
        accepted = client.post(
            f"/api/results/{run_id}/recover-home",
            json={"confirmation": RECOVERY_CONFIRMATION},
        )
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            recovered_run = client.get(f"/api/results/{run_id}").json()["run"]
            attempt = recovered_run["recovery_attempts"][0]
            if attempt["outcome"] is not None:
                break
            time.sleep(0.01)
        duplicate = client.post(
            f"/api/results/{run_id}/recover-home",
            json={"confirmation": RECOVERY_CONFIRMATION},
        )
        status = client.get("/api/status").json()

    assert wrong.status_code == 409
    assert accepted.status_code == 202
    assert accepted.json()["outbound_forward_pulses"] == 3
    assert accepted.json()["maximum_forward_pulses"] == 5
    assert accepted.json()["forward_pulse_source"] == (
        "failure_details.motion_commands"
    )
    assert duplicate.status_code == 409
    assert recovered_run["outcome"] == "FAILED"
    assert recovered_run["reason"] == "ARRIVAL_FAILURE"
    assert "events" in recovered_run
    assert "stage_results" in recovered_run
    assert "failure_details" in recovered_run
    assert (tmp_path / run_id / "events.ndjson").is_file()
    assert attempt["outcome"] == "COMPLETED"
    assert attempt["reason"] == "HOME_POSITION_RECOVERED"
    assert attempt["final_safety_state"] == "DISARMED_CONFIRMED"
    assert [step["step"] for step in attempt["steps"]] == [
        "preflight",
        "turn_toward_home",
        "return_home",
    ]
    assert attempt["steps"][2]["evidence"]["home_distance_m"] == 0.08
    assert attempt["steps"][0]["evidence"]["outbound_forward_pulses"] == 3
    assert attempt["steps"][0]["evidence"]["maximum_forward_pulses"] == 5
    assert attempt["steps"][2]["evidence"]["requested_forward_pulses"] == 5
    assert attempt["steps"][2]["evidence"]["motion_commands"][0][
        "phase"
    ] == "recovery_return_home"
    assert hardware.calls[0][0] == "turn_toward_home"
    assert hardware.calls[0][1] == failed["home"]
    assert hardware.calls[1][0] == "return_home"
    assert hardware.calls[1][2]["forward_pulse_count"] == 5
    assert status["active_recovery"] is None
    assert status["hardware"]["motion"]["armed"] is False


def test_active_recovery_blocks_activation_and_operator_stop_seals_it(
    tmp_path,
) -> None:
    class FailedApproachStages(SimulatedStageExecutor):
        async def execute(self, phase, context):
            if phase is MissionPhase.APPROACH_FRUIT:
                raise StageFailure(
                    "ARRIVAL_FAILURE",
                    "arrival timed out",
                    details={
                        "motion_commands": [
                            {
                                "phase": "approach_fruit",
                                "forward_mps": 1.0,
                            }
                        ]
                    },
                )
            return await super().execute(phase, context)

    class SlowRecoveryHardware(RecoveryHardwareBoundary):
        async def turn_toward_home(self, home, **options):
            await asyncio.sleep(1.0)
            return await super().turn_toward_home(home, **options)

    hardware = SlowRecoveryHardware()
    with TestClient(
        create_app(
            runs_root=tmp_path,
            hardware=hardware,
            camera_perception_status=ready_camera_perception,
            stage_executor=FailedApproachStages(),
        )
    ) as client:
        run_id = client.post("/api/run", json={"target_fruit": "pear"}).json()[
            "run"
        ]["run_id"]
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            failed = client.get(f"/api/results/{run_id}").json()["run"]
            if failed["outcome"] is not None:
                break
            time.sleep(0.01)

        accepted = client.post(
            f"/api/results/{run_id}/recover-home",
            json={"confirmation": RECOVERY_CONFIRMATION},
        )
        active = client.get("/api/status").json()
        blocked = client.post("/api/run", json={"target_fruit": "apple"})
        stopped = client.post("/api/stop")
        recovered_run = client.get(f"/api/results/{run_id}").json()["run"]
        final_status = client.get("/api/status").json()

    assert accepted.status_code == 202
    assert active["active_recovery"]["run_id"] == run_id
    assert active["activation"]["ready"] is False
    assert blocked.status_code == 409
    assert stopped.status_code == 200
    assert recovered_run["recovery_attempts"][0]["outcome"] == "STOPPED"
    assert recovered_run["recovery_attempts"][0]["reason"] == "OPERATOR_STOP"
    assert recovered_run["recovery_attempts"][0]["final_safety_state"] == (
        "DISARMED_CONFIRMED"
    )
    assert final_status["active_recovery"] is None


def test_stop_during_a_stage_cancels_the_demo_without_late_resume(tmp_path) -> None:
    with TestClient(
        create_app(
            runs_root=tmp_path,
            hardware=ReadyHardwareBoundary(),
            camera_perception_status=ready_camera_perception,
            stage_executor=SimulatedStageExecutor(
                delay_at="turn_to_fruit",
                delay_s=0.25,
            ),
        )
    ) as client:
        run_id = client.post("/api/run", json={"target_fruit": "pear"}).json()["run"][
            "run_id"
        ]

        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            run = client.get(f"/api/results/{run_id}").json()["run"]
            if run["current_phase"] == "turn_to_fruit":
                break
            time.sleep(0.01)

        stopped = client.post("/api/stop")
        time.sleep(0.30)
        run = client.get(f"/api/results/{run_id}").json()["run"]

        assert stopped.status_code == 200
        assert run["outcome"] == "STOPPED"
        assert run["reason"] == "OPERATOR_STOP"
        assert run["current_phase"] == "stopped"
        assert run["failed_phase"] is None
        assert client.get("/api/status").json()["mission"]["phase"] == "stopped"


@pytest.mark.parametrize(
    ("failed_phase", "reason"),
    [
        ("turn_to_fruit", "TARGET_RECOGNITION_FAILURE"),
        ("find_fruit", "TARGET_RECOGNITION_FAILURE"),
        ("approach_fruit", "ARRIVAL_FAILURE"),
        ("sit_and_bark", "ACTION_FAILURE"),
        ("stand", "ACTION_FAILURE"),
        ("turn_toward_home", "RETURN_HOME_FAILURE"),
        ("return_home", "RETURN_HOME_FAILURE"),
        ("restore_heading", "RETURN_HOME_FAILURE"),
    ],
)
def test_each_stage_has_a_distinct_default_failure_result(
    tmp_path,
    failed_phase: str,
    reason: str,
) -> None:
    with TestClient(
        create_app(
            runs_root=tmp_path,
            hardware=ReadyHardwareBoundary(),
            camera_perception_status=ready_camera_perception,
            stage_executor=SimulatedStageExecutor(fail_at=failed_phase),
        )
    ) as client:
        run_id = client.post("/api/run", json={"target_fruit": "pear"}).json()["run"][
            "run_id"
        ]

        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            run = client.get(f"/api/results/{run_id}").json()["run"]
            if run["outcome"] is not None:
                break
            time.sleep(0.01)

        assert run["outcome"] == "FAILED"
        assert run["reason"] == reason
        assert run["failed_phase"] == failed_phase
        assert run["final_safety_state"] == "DISARMED_CONFIRMED"


def test_diagnostics_identifies_completed_failed_and_unreached_stages(
    tmp_path,
) -> None:
    with TestClient(
        create_app(
            runs_root=tmp_path,
            hardware=ReadyHardwareBoundary(),
            camera_perception_status=ready_camera_perception,
            stage_executor=SimulatedStageExecutor(fail_at="return_home"),
        )
    ) as client:
        run_id = client.post("/api/run", json={"target_fruit": "pear"}).json()["run"][
            "run_id"
        ]

        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            run = client.get(f"/api/results/{run_id}").json()["run"]
            if run["outcome"] is not None:
                break
            time.sleep(0.01)

        response = client.get("/api/diagnostics/stages")

        assert response.status_code == 200
        body = response.json()
        latest = body["latest_run"]
        assert latest["run_id"] == run_id
        assert latest["outcome"] == "FAILED"
        assert latest["reason"] == "RETURN_HOME_FAILURE"
        assert latest["failed_phase"] == "return_home"
        assert latest["failure_details"] is None
        assert latest["evidence_capture"] == {"available": True}
        assert [artifact["filename"] for artifact in latest["artifacts"]] == [
            "flight-recorder.ndjson",
            "evidence-capture-warnings.json",
        ]
        assert [stage["phase"] for stage in body["stages"]] == [
            "orient_for_run",
            "turn_to_fruit",
            "find_fruit",
            "approach_fruit",
            "sit_and_bark",
            "stand",
            "turn_toward_home",
            "return_home",
            "restore_heading",
        ]
        assert [stage["status"] for stage in body["stages"]] == [
            "COMPLETED",
            "COMPLETED",
            "COMPLETED",
            "COMPLETED",
            "COMPLETED",
            "COMPLETED",
            "COMPLETED",
            "FAILED",
            "NOT_RUN",
        ]
        assert body["stages"][6]["evidence"] == {
            "home_bearing_error_rad": 0.03,
            "motion_commands_sent": False,
        }
        assert body["stages"][7]["evidence"] is None


def test_home_capture_fails_closed_if_pose_freshness_is_lost(tmp_path) -> None:
    hardware = PoseLostAtHomeBoundary()
    with TestClient(
        create_app(
            runs_root=tmp_path,
            hardware=hardware,
            camera_perception_status=ready_camera_perception,
        )
    ) as client:
        response = client.post("/api/run", json={"target_fruit": "pear"})

        assert response.status_code == 201
        run = response.json()["run"]
        assert run["outcome"] == "FAILED"
        assert run["reason"] == "PREFLIGHT_FAILURE"
        assert run["failed_phase"] == "capture_home"
        assert run["final_safety_state"] == "DISARMED_CONFIRMED"
        assert "home" not in run
        assert "pose sample became stale" in run["message"]
        assert hardware.stop_calls == 1
        assert client.get("/api/status").json()["active_run_id"] is None


def test_camera_readiness_error_becomes_a_failed_preflight_check(tmp_path) -> None:
    with TestClient(
        create_app(
            runs_root=tmp_path,
            hardware=ReadyHardwareBoundary(),
            camera_perception_status=broken_camera_perception,
        )
    ) as client:
        response = client.post("/api/run", json={"target_fruit": "pear"})

        assert response.status_code == 201
        run = response.json()["run"]
        assert run["outcome"] == "FAILED"
        assert run["reason"] == "PREFLIGHT_FAILURE"
        camera_check = next(
            check
            for check in run["preflight"]["checks"]
            if check["name"] == "camera_perception_ready"
        )
        assert camera_check == {
            "name": "camera_perception_ready",
            "ready": False,
            "detail": "camera/perception readiness error: camera process unavailable",
        }


def test_stop_seals_the_active_demo_run_and_allows_another(tmp_path) -> None:
    with TestClient(ready_app(tmp_path)) as client:
        started = client.post("/api/run", json={"target_fruit": "pear"}).json()["run"]

        stopped = client.post("/api/stop")

        assert stopped.status_code == 200
        result = client.get(f"/api/results/{started['run_id']}").json()["run"]
        assert result["outcome"] == "STOPPED"
        assert result["reason"] == "OPERATOR_STOP"
        assert result["current_phase"] == "stopped"
        assert result["final_safety_state"] == "DISARMED_CONFIRMED"
        assert client.get("/api/status").json()["active_run_id"] is None
        assert client.post("/api/run", json={"target_fruit": "pear"}).status_code == 201


def test_startup_seals_an_interrupted_demo_run(tmp_path) -> None:
    with TestClient(ready_app(tmp_path)) as client:
        started = client.post("/api/run", json={"target_fruit": "pear"}).json()["run"]

    with TestClient(create_app(runs_root=tmp_path)) as restarted:
        recovered = restarted.get(f"/api/results/{started['run_id']}").json()["run"]

        assert recovered["outcome"] == "FAILED"
        assert recovered["reason"] == "PROCESS_INTERRUPTED"
        assert recovered["current_phase"] == "failed"
        assert recovered["final_safety_state"] == "STOP_REQUESTED_UNCONFIRMED"
        assert restarted.get("/api/status").json()["active_run_id"] is None


def test_results_list_returns_newest_demo_run_first(tmp_path) -> None:
    with TestClient(create_app(runs_root=tmp_path)) as client:
        first = client.post("/api/run", json={"target_fruit": "pear"}).json()["run"]
        client.post("/api/stop")
        second = client.post("/api/run", json={"target_fruit": "pear"}).json()["run"]

        response = client.get("/api/results")

        assert response.status_code == 200
        assert [run["run_id"] for run in response.json()["runs"]] == [
            second["run_id"],
            first["run_id"],
        ]


def test_activate_accepts_the_qualified_red_apple_target(tmp_path) -> None:
    selected: list[str] = []
    with TestClient(
        create_app(
            runs_root=tmp_path,
            select_perception_target=lambda fruit: (
                selected.append(fruit)
                or {"target_fruit": fruit, "supported_fruits": [fruit]}
            ),
        )
    ) as client:
        response = client.post("/api/run", json={"target_fruit": "apple"})

        assert response.status_code == 201
        assert response.json()["run"]["target_fruit"] == "apple"
        assert selected == ["apple"]


def test_activate_records_voice_as_the_activation_source(tmp_path) -> None:
    with TestClient(create_app(runs_root=tmp_path)) as client:
        response = client.post(
            "/api/run",
            json={"target_fruit": "pear", "activation_source": "voice"},
        )

        assert response.status_code == 201
        assert response.json()["run"]["activation_source"] == "voice"


def test_activate_accepts_specialist_gated_banana(tmp_path) -> None:
    with TestClient(create_app(runs_root=tmp_path)) as client:
        response = client.post("/api/run", json={"target_fruit": "banana"})

        assert response.status_code == 201
        assert response.json()["run"]["target_fruit"] == "banana"


@pytest.mark.parametrize("orientation_degrees", [-1, 360])
def test_activate_rejects_orientation_outside_one_revolution(
    tmp_path,
    orientation_degrees,
) -> None:
    with TestClient(create_app(runs_root=tmp_path)) as client:
        response = client.post(
            "/api/run",
            json={
                "target_fruit": "pear",
                "orientation_degrees": orientation_degrees,
            },
        )

    assert response.status_code == 422


def test_activate_rejects_an_unsupported_target_fruit(tmp_path) -> None:
    with TestClient(create_app(runs_root=tmp_path)) as client:
        response = client.post("/api/run", json={"target_fruit": "grape"})

        assert response.status_code == 422
        assert client.get("/api/results").json()["runs"] == []


def test_live_pulse_is_rejected_while_hardware_is_disabled() -> None:
    response = TestClient(create_app()).post(
        "/api/hardware/forward-pulse",
        json={"confirmation": "PATH CLEAR - MOVE WOOF FORWARD"},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "Go2 hardware is disabled"


def test_remote_takeover_latch_blocks_live_pulse_before_hardware() -> None:
    mission = MissionMachine()
    mission.remote_takeover(RemoteInput("unitree_remote", "left_stick", monotonic()))

    response = TestClient(create_app(mission=mission)).post(
        "/api/hardware/forward-pulse",
        json={"confirmation": "PATH CLEAR - MOVE WOOF FORWARD"},
    )

    assert response.status_code == 423
    assert "restart required" in response.json()["detail"]
