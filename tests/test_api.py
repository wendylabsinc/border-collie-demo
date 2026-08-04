import time
from time import monotonic

import pytest
from fastapi.testclient import TestClient

from border_collie_demo.api import create_app
from border_collie_demo.mission import MissionMachine
from border_collie_demo.models import MissionPhase, RemoteInput
from border_collie_demo.orchestrator import SimulatedStageExecutor


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
            "pose": {"healthy": True, "age_s": 0.04, "error": None},
            "motion": {"armed": False},
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
    assert ':8111/api/camera/frame.jpg' in response.text
    assert "YOLO pear model overlay" in response.text


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
            check["name"]: check["ready"]
            for check in created["preflight"]["checks"]
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
        assert run["stage_results"]["approach_fruit"]["final_push_mps"] == 0.3
        assert run["stage_results"]["approach_fruit"]["final_push_duration_s"] == 1.0
        assert run["stage_results"]["sit_and_bark"]["down_hold_s"] == 5.0
        assert run["stage_results"]["return_home"]["requested_forward_pulses"] == 7
        assert run["stage_results"]["return_home"]["replayed_forward_pulses"] == 7
        assert run["stage_results"]["restore_heading"]["heading_error_rad"] == 0.04
        assert run["terminal_measurements"] == {
            "home_distance_m": 0.08,
            "heading_error_rad": 0.04,
        }


def test_demo_run_carries_outbound_forward_pulses_into_return_playback(tmp_path) -> None:
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
        run_id = client.post(
            "/api/run", json={"target_fruit": "pear"}
        ).json()["run"]["run_id"]
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
        started = client.post(
            "/api/run", json={"target_fruit": "pear"}
        ).json()["run"]
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
        run_id = client.post(
            "/api/run", json={"target_fruit": "pear"}
        ).json()["run"]["run_id"]

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
        ("turn_to_fruit", "MOTION_FAILURE"),
        ("find_fruit", "TARGET_LOST"),
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
        run_id = client.post(
            "/api/run", json={"target_fruit": "pear"}
        ).json()["run"]["run_id"]

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
        run_id = client.post(
            "/api/run", json={"target_fruit": "pear"}
        ).json()["run"]["run_id"]

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
        started = client.post(
            "/api/run", json={"target_fruit": "pear"}
        ).json()["run"]

        stopped = client.post("/api/stop")

        assert stopped.status_code == 200
        result = client.get(f"/api/results/{started['run_id']}").json()["run"]
        assert result["outcome"] == "STOPPED"
        assert result["reason"] == "OPERATOR_STOP"
        assert result["current_phase"] == "stopped"
        assert result["final_safety_state"] == "DISARMED_CONFIRMED"
        assert client.get("/api/status").json()["active_run_id"] is None
        assert client.post(
            "/api/run", json={"target_fruit": "pear"}
        ).status_code == 201


def test_startup_seals_an_interrupted_demo_run(tmp_path) -> None:
    with TestClient(ready_app(tmp_path)) as client:
        started = client.post(
            "/api/run", json={"target_fruit": "pear"}
        ).json()["run"]

    with TestClient(create_app(runs_root=tmp_path)) as restarted:
        recovered = restarted.get(
            f"/api/results/{started['run_id']}"
        ).json()["run"]

        assert recovered["outcome"] == "FAILED"
        assert recovered["reason"] == "PROCESS_INTERRUPTED"
        assert recovered["current_phase"] == "failed"
        assert recovered["final_safety_state"] == "UNKNOWN"
        assert restarted.get("/api/status").json()["active_run_id"] is None


def test_results_list_returns_newest_demo_run_first(tmp_path) -> None:
    with TestClient(create_app(runs_root=tmp_path)) as client:
        first = client.post(
            "/api/run", json={"target_fruit": "pear"}
        ).json()["run"]
        client.post("/api/stop")
        second = client.post(
            "/api/run", json={"target_fruit": "pear"}
        ).json()["run"]

        response = client.get("/api/results")

        assert response.status_code == 200
        assert [run["run_id"] for run in response.json()["runs"]] == [
            second["run_id"],
            first["run_id"],
        ]


def test_activate_rejects_an_unqualified_target_fruit(tmp_path) -> None:
    with TestClient(create_app(runs_root=tmp_path)) as client:
        response = client.post(
            "/api/run", json={"target_fruit": "apple"}
        )

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
