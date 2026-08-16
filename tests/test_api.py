import time
from time import monotonic

import pytest
from fastapi.testclient import TestClient

from border_collie_demo.api import create_app
from border_collie_demo.evidence import EvidenceArtifact
from border_collie_demo.mission import MissionMachine
from border_collie_demo.models import MissionPhase, RemoteInput
from border_collie_demo.orchestrator import SimulatedStageExecutor, StageFailure


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
            "connected": True,
            "fault": None,
            "active_operation": None,
            "pose": {
                "healthy": True,
                "age_s": 0.04,
                "error": None,
                "pose": {"x_m": 1.25, "y_m": -0.5},
            },
            "motion": {
                "armed": False,
                "last_command": {"forward_mps": 0.0, "yaw_rps": 0.0},
            },
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


class RecordingFailureEpilogue:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def recover(self, home, **context):
        self.calls.append({"home": home, **context})
        return {
            "status": "RETURNED_HOME",
            "reason": "bounded position-only return reached Home",
            "attempted_return": True,
            "exact_stop_confirmed": True,
            "terminal_home_measurement": {
                "home_distance_m": 0.04,
                "pose_age_s": 0.02,
                "pose_captured_monotonic_s": 42.0,
                "pose_source": "rt/sportmodestate",
            },
            "return_evidence": {"heading_restoration_skipped": True},
        }


def test_audience_page_includes_the_annotated_camera_feed() -> None:
    response = TestClient(create_app()).get("/")

    assert response.status_code == 200
    assert 'id="camera-feed"' in response.text
    assert "'/api/camera/frame.jpg'" in response.text
    assert ":8111/api/camera/frame.jpg" not in response.text
    assert "YOLO fruit model overlay" in response.text
    assert 'id="target-fruit"' in response.text
    assert '<option value="apple">Red apple</option>' in response.text
    assert '<option value="mango">Mango</option>' in response.text
    assert '<option value="banana">Banana</option>' not in response.text
    assert "Run Apple + Mango + Pear once" in response.text
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


def test_raw_camera_preview_is_proxied_for_client_side_model_boxes() -> None:
    jpeg = b"\xff\xd8raw-preview\xff\xd9"

    response = TestClient(create_app(raw_camera_frame=lambda: jpeg)).get(
        "/api/camera/raw.jpg"
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.headers["cache-control"] == "no-store"
    assert response.content == jpeg


def test_run_black_box_can_be_downloaded_while_the_run_is_active(tmp_path) -> None:
    with TestClient(ready_app(tmp_path)) as client:
        run = client.post(
            "/api/run",
            json={"target_fruit": "pear", "activation_id": "black-box-test"},
        ).json()["run"]

        response = client.get(f"/api/results/{run['run_id']}/black-box.ndjson")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    assert b'"kind":"run_started"' in response.content


def test_fruit_test_page_can_select_supported_fruit_without_motion() -> None:
    selected: list[str] = []
    app = create_app(
        camera_perception_status=lambda: {
            "target_fruit": "apple",
            "camera_healthy": True,
            "detection": {
                "label": "apple",
                "confidence": 0.73,
                "bbox_xyxy": [120.0, 240.0, 360.0, 700.0],
                "age_s": 0.04,
            },
        },
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
        status = client.get("/api/fruits/preview")

    assert page.status_code == 200
    assert "Camera only — Woof will not move" in page.text
    assert "const frameUrl = '/api/camera/raw.jpg';" in page.text
    assert '<option value="orange" data-model="coco">Orange (COCO)</option>' in page.text
    assert (
        '<option value="mango" data-model="coco" '
        'data-coco-label="bowl">Mango (COCO bowl)</option>' in page.text
    )
    assert "await configureCoco(true)" in page.text
    assert "item.label.toLowerCase() === 'bowl'" in page.text
    assert "proposal.latest_candidate_evidence" in page.text
    assert 'id="model-box"' in page.text
    assert "body.color_candidates" in page.text
    assert "saturated color confirmed Orange" in page.text
    assert ":8111/api/camera/frame.jpg" not in page.text
    assert "document.hidden" in page.text
    assert "scheduleRefresh(1500)" in page.text
    assert 'id="live-confidence"' in page.text
    assert "fruit.addEventListener('change', selectFruit);" in page.text
    assert response.status_code == 200
    assert response.json() == {
        "target_fruit": "apple",
        "qualified_for_demo": True,
        "supported_fruits": ["apple", "banana", "pear"],
    }
    assert status.status_code == 200
    assert status.json() == {
        "target_fruit": "apple",
        "camera_healthy": True,
        "detection": {
            "label": "apple",
            "confidence": 0.73,
            "bbox_xyxy": [120.0, 240.0, 360.0, 700.0],
            "age_s": 0.04,
        },
    }
    assert selected == ["apple"]


def test_fruit_test_page_controls_read_only_full_coco_confidence_test() -> None:
    changes: list[dict[str, object]] = []
    app = create_app(
        coco_test_status=lambda: {
            "enabled": True,
            "model": "yolo11n.pt",
            "class_count": 80,
            "frames_processed": 12,
            "classes": [],
        },
        configure_coco_test=lambda payload: changes.append(payload) or {
            "enabled": payload["enabled"],
            "model": "yolo11n.pt",
            "class_count": 80,
            "frames_processed": 0,
            "classes": [],
        },
    )

    with TestClient(app) as client:
        page = client.get("/fruit-test")
        status = client.get("/api/coco-test")
        enabled = client.post(
            "/api/coco-test",
            json={"enabled": True, "minimum_confidence": 0.05, "reset": True},
        )

    assert "COCO replacement tester" in page.text
    assert "all-frame score" in page.text
    assert status.json()["class_count"] == 80
    assert enabled.status_code == 200
    assert changes == [
        {"enabled": True, "minimum_confidence": 0.05, "reset": True}
    ]


def test_fruit_list_temporarily_replaces_banana_with_mango() -> None:
    response = TestClient(create_app()).get("/api/fruits")

    assert response.status_code == 200
    assert response.json() == {
        "supported_fruits": ["apple", "banana", "mango", "pear"],
        "qualified_fruits": ["apple", "mango", "pear"],
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
        ],
    }


def test_activate_fails_closed_when_preflight_is_not_ready(tmp_path) -> None:
    with TestClient(create_app(runs_root=tmp_path)) as client:
        response = client.post("/api/run", json={"target_fruit": "pear"})

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
        response = client.post("/api/run", json={"target_fruit": "pear"})
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
        assert list(run["stage_results"]) == [
            "turn_to_fruit",
            "find_fruit",
            "approach_fruit",
            "sit_and_bark",
            "stand",
            "turn_toward_home",
            "return_home",
            "restore_heading",
        ]
        assert run["stage_results"]["return_home"]["home_distance_m"] == 0.08
        assert run["stage_results"]["approach_fruit"]["forward_pulse_count"] == 7
        assert run["stage_results"]["approach_fruit"]["final_push_mps"] == 0.6
        assert run["stage_results"]["approach_fruit"]["final_push_duration_s"] == 1.0
        assert run["stage_results"]["sit_and_bark"]["down_hold_s"] == 5.0
        assert run["stage_results"]["return_home"]["requested_forward_pulses"] == 7
        assert run["stage_results"]["return_home"]["replayed_forward_pulses"] == 7
        assert run["stage_results"]["restore_heading"]["heading_error_rad"] == 0.04
        assert run["terminal_measurements"] == {
            "home_distance_m": 0.08,
            "heading_error_rad": None,
        }


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
    assert run["stage_results"]["approach_fruit"]["forward_pulse_count"] == 7
    assert run["stage_results"]["return_home"]["requested_forward_pulses"] == 7
    assert run["stage_results"]["return_home"]["replayed_forward_pulses"] == 7


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
        assert list(run["stage_results"]) == ["turn_to_fruit"]


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
        "evidence.zip",
        "terminal.jpg",
    ]
    assert archive.status_code == 200
    assert archive.headers["content-type"] == "application/zip"
    assert archive.content == b"PK\x03\x04raw-fieldmark-frames"
    assert unreferenced.status_code == 404
    assert [
        artifact["filename"] for artifact in diagnostic["latest_run"]["artifacts"]
    ] == ["evidence.zip", "terminal.jpg"]


def test_approach_timeout_diagnostics_are_persisted_in_the_run_result(tmp_path) -> None:
    diagnostics = {
        "recognition": {
            "guidance_reason": "tracking_confidence_below_floor",
            "approach_trace": [
                {
                    "sample": 1,
                    "confidence": 0.54,
                    "guidance_action": "stop",
                    "guidance_reason": "tracking_confidence_below_floor",
                    "resulting_command": {
                        "forward_mps": 0.0,
                        "yaw_rps": 0.0,
                        "reason": "tracking_confidence_below_floor",
                    },
                }
            ],
            "approach_summary": {
                "samples": 1,
                "action_counts": {"stop": 1},
                "reason_counts": {"tracking_confidence_below_floor": 1},
                "forward_decisions": 0,
                "stop_decisions": 1,
                "final_guidance_reason": "tracking_confidence_below_floor",
            },
        }
    }

    class TimedOutApproachStages(SimulatedStageExecutor):
        async def execute(self, phase, context):
            if phase is MissionPhase.APPROACH_FRUIT:
                raise StageFailure(
                    "ARRIVAL_FAILURE",
                    "pear camera guidance timed out",
                    details=diagnostics,
                )
            return await super().execute(phase, context)

    with TestClient(
        create_app(
            runs_root=tmp_path,
            hardware=ReadyHardwareBoundary(),
            camera_perception_status=ready_camera_perception,
            stage_executor=TimedOutApproachStages(),
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

    assert run["reason"] == "ARRIVAL_FAILURE"
    assert run["failed_phase"] == "approach_fruit"
    assert run["failure_details"] == diagnostics


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


def test_failed_run_remains_failed_after_one_successful_home_epilogue(
    tmp_path,
) -> None:
    epilogue = RecordingFailureEpilogue()
    with TestClient(
        create_app(
            runs_root=tmp_path,
            hardware=ReadyHardwareBoundary(),
            camera_perception_status=ready_camera_perception,
            stage_executor=SimulatedStageExecutor(fail_at="sit_and_bark"),
            failure_epilogue=epilogue,
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
    assert run["reason"] == "ACTION_FAILURE"
    assert run["failed_phase"] == "sit_and_bark"
    assert run["stage_results"]["approach_fruit"]["forward_pulse_count"] == 7
    assert run["failure_epilogue"]["status"] == "RETURNED_HOME"
    assert run["terminal_measurements"] == {
        "home_distance_m": 0.04,
        "heading_error_rad": None,
    }
    assert run["final_safety_state"] == "DISARMED_CONFIRMED"
    assert epilogue.calls == [
        {
            "home": {
                "x_m": 1.25,
                "y_m": -0.5,
                "yaw_rad": 0.75,
                "captured_monotonic_s": 123.0,
                "age_s": 0.04,
                "source": "rt/sportmodestate",
            },
            "original_reason": "ACTION_FAILURE",
            "failed_phase": "sit_and_bark",
            "takeover_latched": False,
        }
    ]


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
        assert body["latest_run"] == {
            "run_id": run_id,
            "outcome": "FAILED",
            "reason": "RETURN_HOME_FAILURE",
            "failed_phase": "return_home",
            "failure_details": None,
            "artifacts": [],
            "evidence_capture": {
                "available": False,
                "unavailable_reason": ("terminal evidence adapter is not configured"),
            },
        }
        assert [stage["phase"] for stage in body["stages"]] == [
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
            "FAILED",
            "NOT_RUN",
        ]
        assert body["stages"][5]["evidence"] == {
            "home_bearing_error_rad": 0.03,
            "motion_commands_sent": False,
        }
        assert body["stages"][6]["evidence"] is None


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


def test_orderly_shutdown_seals_an_interrupted_demo_run_after_disarm(tmp_path) -> None:
    with TestClient(ready_app(tmp_path)) as client:
        started = client.post("/api/run", json={"target_fruit": "pear"}).json()["run"]

    with TestClient(create_app(runs_root=tmp_path)) as restarted:
        recovered = restarted.get(f"/api/results/{started['run_id']}").json()["run"]

        assert recovered["outcome"] == "FAILED"
        assert recovered["reason"] == "PROCESS_INTERRUPTED"
        assert recovered["current_phase"] == "failed"
        assert recovered["final_safety_state"] == "DISARMED_CONFIRMED"
        assert restarted.get("/api/status").json()["active_run_id"] is None


def test_http_activation_id_replays_the_same_durable_run(tmp_path) -> None:
    with TestClient(ready_app(tmp_path)) as client:
        request = {
            "target_fruit": "pear",
            "activation_source": "audience_ui",
            "activation_id": "browser-click-123",
        }

        first = client.post("/api/run", json=request)
        replay = client.post("/api/run", json=request)

        assert first.status_code == 201
        assert replay.status_code == 201
        assert replay.json()["idempotent_replay"] is True
        assert replay.json()["run"] == first.json()["run"]
        assert replay.json()["run"]["activation_id"] == "browser-click-123"


@pytest.mark.parametrize(
    ("fruit", "focus", "lock"),
    [
        ("apple", 0.52, 0.42),
        ("pear", 0.70, 0.66),
        ("mango", 0.12, 0.10),
    ],
)
def test_run_activation_persists_selected_fruit_search_confidence_tuning(
    tmp_path,
    fruit: str,
    focus: float,
    lock: float,
) -> None:
    with TestClient(ready_app(tmp_path)) as client:
        response = client.post(
            "/api/run",
            json={
                "target_fruit": fruit,
                "activation_id": f"{fruit}-yaw-045",
                "tuning": {
                    "search_yaw_rps": 0.45,
                    "focus_confidence": focus,
                    "lock_confidence": lock,
                    "center_confirmations": 4,
                    "center_tolerance_ratio": 0.10,
                },
            },
        )

        assert response.status_code == 201
        assert response.json()["run"]["search_experiment"] == {
            "target_fruit": fruit,
            "search_yaw_rps": 0.45,
            "focus_confidence": focus,
            "lock_confidence": lock,
            "center_confirmations": 4,
            "center_tolerance_ratio": 0.10,
        }


def test_run_activation_rejects_search_experiment_outside_safety_bounds(
    tmp_path,
) -> None:
    with TestClient(ready_app(tmp_path)) as client:
        response = client.post(
            "/api/run",
            json={
                "target_fruit": "apple",
                "tuning": {"search_yaw_rps": 0.25},
            },
        )

        assert response.status_code == 409
        assert client.get("/api/results").json()["runs"] == []


@pytest.mark.parametrize(
    ("fruit", "tuning", "message"),
    [
        (
            "pear",
            {"focus_confidence": 0.70, "lock_confidence": 0.64},
            "lock_confidence",
        ),
        ("mango", {"focus_confidence": 0.07}, "focus_confidence"),
    ],
)
def test_run_activation_rejects_per_fruit_or_inverted_confidence_before_motion(
    tmp_path,
    fruit: str,
    tuning: dict[str, float],
    message: str,
) -> None:
    with TestClient(ready_app(tmp_path)) as client:
        response = client.post(
            "/api/run",
            json={"target_fruit": fruit, "tuning": tuning},
        )

        assert response.status_code == 409
        assert message in response.json()["detail"]
        assert client.get("/api/results").json()["runs"] == []


def test_activation_id_conflicts_when_selected_fruit_tuning_changes(tmp_path) -> None:
    with TestClient(ready_app(tmp_path)) as client:
        base = {
            "target_fruit": "mango",
            "activation_id": "mango-confidence-1",
            "tuning": {
                "focus_confidence": 0.12,
                "lock_confidence": 0.10,
            },
        }

        first = client.post("/api/run", json=base)
        replay = client.post("/api/run", json=base)
        conflict = client.post(
            "/api/run",
            json={
                **base,
                "tuning": {
                    "focus_confidence": 0.13,
                    "lock_confidence": 0.10,
                },
            },
        )

        assert first.status_code == 201
        assert replay.status_code == 201
        assert replay.json()["idempotent_replay"] is True
        assert conflict.status_code == 409
        assert "different Fruit Mission" in conflict.json()["detail"]


def test_run_activation_persists_one_run_final_push_tuning_and_conflicts_on_change(
    tmp_path,
) -> None:
    with TestClient(ready_app(tmp_path)) as client:
        request = {
            "target_fruit": "apple",
            "activation_id": "apple-push-060x040",
            "tuning": {
                "arrival": {
                    "final_push_mps": 0.60,
                    "final_push_duration_s": 0.40,
                }
            },
        }

        first = client.post("/api/run", json=request)
        replay = client.post("/api/run", json=request)
        conflict = client.post(
            "/api/run",
            json={
                **request,
                "tuning": {
                    "arrival": {
                        "final_push_mps": 0.60,
                        "final_push_duration_s": 0.50,
                    }
                },
            },
        )
        run_id = first.json()["run"]["run_id"]
        black_box = client.get(f"/api/results/{run_id}/black-box.ndjson")

        assert first.status_code == 201
        assert first.json()["run"]["run_tuning"]["arrival"]["final_push_mps"] == 0.60
        assert (
            first.json()["run"]["run_tuning"]["arrival"]["final_push_duration_s"]
            == 0.40
        )
        assert replay.status_code == 201
        assert replay.json()["idempotent_replay"] is True
        assert conflict.status_code == 409
        assert "different Fruit Mission" in conflict.json()["detail"]
        assert b'"final_push_mps":0.6' in black_box.content
        assert b'"final_push_duration_s":0.4' in black_box.content


@pytest.mark.parametrize(
    "arrival",
    [
        {"final_push_mps": 0.49, "final_push_duration_s": 0.40},
        {"final_push_mps": 0.60, "final_push_duration_s": 0.05},
        {"final_push_mps": 0.60, "final_push_duration_s": 1.51},
        {"final_push_mps": "NaN", "final_push_duration_s": 0.40},
        {
            "final_push_mps": 0.60,
            "final_push_duration_s": 0.40,
            "surprise": True,
        },
    ],
)
def test_run_activation_rejects_unsafe_or_unknown_final_push_before_motion(
    tmp_path,
    arrival: dict[str, object],
) -> None:
    with TestClient(ready_app(tmp_path)) as client:
        response = client.post(
            "/api/run",
            json={"target_fruit": "pear", "tuning": {"arrival": arrival}},
        )

        assert response.status_code == 409
        assert client.get("/api/results").json()["runs"] == []


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


def test_stage_ui_renders_server_owned_run_tuning_and_posts_snapshot(
    tmp_path,
) -> None:
    with TestClient(create_app(runs_root=tmp_path)) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert 'id="run-tuning-fields"' in response.text
    assert 'id="effective-tuning"' in response.text
    assert "renderTuning" in response.text
    assert "targetFruit.addEventListener('change'" in response.text
    assert 'id="search-trace"' in response.text
    assert "measured_yaw_rad" in response.text
    assert "confidence" in response.text
    assert "tuning: tuningPayload()" in response.text
    assert "fetch('/api/experiments/search')" in response.text
    # woof.local is served over plain HTTP, where Web Crypto UUID generation is
    # unavailable. The API already creates the durable activation ID.
    assert "crypto.randomUUID" not in response.text
    assert "activation_id:" not in response.text


def test_status_exposes_selected_fruit_confidence_defaults_and_ranges(tmp_path) -> None:
    with TestClient(create_app(runs_root=tmp_path)) as client:
        experiment = client.get("/api/status").json()["search_experiment"]

    assert experiment["fruits"] == {
        "apple": {
            "defaults": {"focus_confidence": 0.40, "lock_confidence": 0.40},
            "ranges": {
                "focus_confidence": [0.40, 0.70],
                "lock_confidence": [0.40, 0.70],
            },
        },
        "banana": {
            "defaults": {"focus_confidence": 0.20, "lock_confidence": 0.20},
            "ranges": {
                "focus_confidence": [0.20, 0.70],
                "lock_confidence": [0.20, 0.70],
            },
        },
        "mango": {
            "defaults": {"focus_confidence": 0.08, "lock_confidence": 0.08},
            "ranges": {
                "focus_confidence": [0.08, 0.20],
                "lock_confidence": [0.08, 0.20],
            },
        },
        "pear": {
            "defaults": {"focus_confidence": 0.65, "lock_confidence": 0.65},
            "ranges": {
                "focus_confidence": [0.65, 0.85],
                "lock_confidence": [0.65, 0.85],
            },
        },
    }


def test_activate_records_voice_as_the_activation_source(tmp_path) -> None:
    with TestClient(create_app(runs_root=tmp_path)) as client:
        response = client.post(
            "/api/run",
            json={"target_fruit": "pear", "activation_source": "voice"},
        )

        assert response.status_code == 201
        assert response.json()["run"]["activation_source"] == "voice"


def test_activate_accepts_derived_mango_target(tmp_path) -> None:
    with TestClient(create_app(runs_root=tmp_path)) as client:
        response = client.post("/api/run", json={"target_fruit": "mango"})

        assert response.status_code == 201
        assert response.json()["run"]["target_fruit"] == "mango"


def test_activate_rejects_temporarily_disabled_banana_before_a_run_exists(
    tmp_path,
) -> None:
    with TestClient(create_app(runs_root=tmp_path)) as client:
        response = client.post("/api/run", json={"target_fruit": "banana"})

        assert response.status_code == 409
        assert response.json()["detail"] == (
            "Banana Demo Runs are temporarily disabled; choose Apple, Mango, or Pear"
        )
        assert client.get("/api/results").json()["runs"] == []


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
