from __future__ import annotations

import time

from fastapi.testclient import TestClient

from border_collie_demo.api import create_app
from border_collie_demo.orchestrator import SimulatedStageExecutor


class ReadyHardware:
    def __init__(self) -> None:
        self.started = False

    async def start(self) -> None:
        self.started = True

    async def close(self) -> list[str]:
        self.started = False
        return []

    async def emergency_stop(self) -> list[str]:
        return []

    def capture_home(self) -> dict[str, object]:
        return {
            "x_m": 1.0,
            "y_m": 2.0,
            "yaw_rad": 0.25,
            "captured_monotonic_s": 3.0,
            "age_s": 0.01,
            "source": "test",
        }

    def status(self) -> dict[str, object]:
        return {
            "configured": True,
            "connected": self.started,
            "fault": None,
            "active_operation": None,
            "pose": {
                "healthy": True,
                "age_s": 0.01,
                "error": None,
                "pose": {"x_m": 1.0, "y_m": 2.0},
            },
            "motion": {
                "armed": False,
                "last_command": {"forward_mps": 0.0, "yaw_rps": 0.0},
            },
        }


def ready_camera() -> dict[str, object]:
    return {"ready": True, "detail": "camera ready"}


def wait_for_cohort(client: TestClient, timeout_s: float = 2.0) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        cohort = client.get("/api/cohorts/active").json()["cohort"]
        if cohort and cohort["status"] != "RUNNING":
            return cohort
        time.sleep(0.01)
    raise AssertionError("cohort did not become terminal")


def test_api_runs_exact_fixed_fruit_cohort_and_persists_decisions(tmp_path) -> None:
    app = create_app(
        runs_root=tmp_path / "runs",
        cohorts_root=tmp_path / "cohorts",
        hardware=ReadyHardware(),
        camera_perception_status=ready_camera,
        stage_executor=SimulatedStageExecutor(),
    )

    with TestClient(app) as client:
        started = client.post(
            "/api/cohorts",
            json={
                "runs": 3,
                "randomized": False,
                "target_fruit": "banana",
                "seed": 81,
                "tolerated_failures": [],
            },
        )
        cohort = wait_for_cohort(client)
        fetched = client.get(f"/api/cohorts/{cohort['cohort_id']}")

    assert started.status_code == 201
    assert cohort["status"] == "COMPLETED"
    assert cohort["policy"]["runs"] == 3
    assert cohort["fruit_sequence"] == ["banana", "banana", "banana"]
    assert [item["target_fruit"] for item in cohort["runs"]] == [
        "banana",
        "banana",
        "banana",
    ]
    assert len({item["activation_id"] for item in cohort["runs"]}) == 3
    assert all(item["cohort_decision"] for item in cohort["runs"])
    assert fetched.status_code == 200
    assert fetched.json()["cohort"] == cohort


def test_api_random_cohort_persists_seeded_sequence(tmp_path) -> None:
    def run(seed: int, directory: str) -> list[str]:
        app = create_app(
            runs_root=tmp_path / directory / "runs",
            cohorts_root=tmp_path / directory / "cohorts",
            hardware=ReadyHardware(),
            camera_perception_status=ready_camera,
            stage_executor=SimulatedStageExecutor(),
        )
        with TestClient(app) as client:
            response = client.post(
                "/api/cohorts", json={"runs": 5, "randomized": True, "seed": seed}
            )
            assert response.status_code == 201
            return wait_for_cohort(client)["fruit_sequence"]

    assert run(919, "a") == run(919, "b")


def test_api_random_cohort_persists_subset_sequence_and_frozen_yaw_tuning(
    tmp_path,
) -> None:
    app = create_app(
        runs_root=tmp_path / "runs",
        cohorts_root=tmp_path / "cohorts",
        hardware=ReadyHardware(),
        camera_perception_status=ready_camera,
        stage_executor=SimulatedStageExecutor(),
    )
    request = {
        "runs": 4,
        "randomized": True,
        "fruit_subset": ["apple", "pear"],
        "seed": 919,
        "tuning": {
            "search": {"yaw_rps": 0.8},
            "home": {"align_yaw_rps": 0.8},
        },
    }

    with TestClient(app) as client:
        started = client.post("/api/cohorts", json=request)
        assert started.status_code == 201
        cohort = wait_for_cohort(client)

    assert cohort["policy"]["fruit_subset"] == ["apple", "pear"]
    assert cohort["selected_fruits"] == ["apple", "pear"]
    assert set(cohort["fruit_sequence"]) == {"apple", "pear"}
    assert cohort["tuning_template"] == request["tuning"]
    assert len(cohort["run_tuning_sequence"]) == 4
    for number, snapshot in enumerate(cohort["run_tuning_sequence"], start=1):
        assert snapshot["number"] == number
        assert snapshot["target_fruit"] == cohort["fruit_sequence"][number - 1]
        tuning = snapshot["run_tuning"]
        assert tuning["search"]["yaw_rps"] == 0.8
        assert tuning["home"]["align_yaw_rps"] == 0.8
        assert tuning["centering"]["focus_yaw_rps"] == 0.4
        assert tuning["centering"]["approach_yaw_rps"] == 0.3
        assert tuning["centering"]["recenter_yaw_rps"] == 0.5
    assert [item["run_tuning"] for item in cohort["runs"]] == [
        item["run_tuning"] for item in cohort["run_tuning_sequence"]
    ]


def test_api_rejects_empty_or_unknown_randomized_subset_before_activation(
    tmp_path,
) -> None:
    app = create_app(
        runs_root=tmp_path / "runs",
        cohorts_root=tmp_path / "cohorts",
        hardware=ReadyHardware(),
        camera_perception_status=ready_camera,
        stage_executor=SimulatedStageExecutor(),
    )
    with TestClient(app) as client:
        empty = client.post(
            "/api/cohorts",
            json={"runs": 2, "randomized": True, "fruit_subset": [], "seed": 1},
        )
        unknown = client.post(
            "/api/cohorts",
            json={
                "runs": 2,
                "randomized": True,
                "fruit_subset": ["mango"],
                "seed": 1,
            },
        )
        runs = client.get("/api/results").json()["runs"]

    assert empty.status_code in {409, 422}
    assert unknown.status_code in {409, 422}
    assert runs == []


def test_terminal_failure_policy_is_applied_before_home_clearance(tmp_path) -> None:
    app = create_app(
        runs_root=tmp_path / "runs",
        cohorts_root=tmp_path / "cohorts",
        hardware=ReadyHardware(),
        camera_perception_status=ready_camera,
        stage_executor=SimulatedStageExecutor(
            fail_at="sit_and_bark",
            failure_reason="ACTION_FAILURE",
            failure_message="speaker unavailable",
        ),
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/cohorts",
            json={
                "runs": 2,
                "randomized": False,
                "target_fruit": "pear",
                "seed": 8,
                "tolerated_failures": [{"reason": "ACTION_FAILURE"}],
            },
        )
        assert response.status_code == 201
        cohort = wait_for_cohort(client)

    assert cohort["status"] == "COMPLETED"
    assert len(cohort["runs"]) == 2
    assert all(
        item["cohort_decision"]["code"] == "FAILURE_TOLERATED"
        for item in cohort["runs"]
    )
    assert all(
        item["inter_run_clearance"]["safe_to_continue"] is True
        for item in cohort["runs"]
    )


def test_second_start_is_rejected_and_stop_is_observable(tmp_path) -> None:
    app = create_app(
        runs_root=tmp_path / "runs",
        cohorts_root=tmp_path / "cohorts",
        hardware=ReadyHardware(),
        camera_perception_status=ready_camera,
        stage_executor=SimulatedStageExecutor(
            delay_at="turn_to_fruit", delay_s=0.5
        ),
    )

    with TestClient(app) as client:
        first = client.post("/api/cohorts", json={"runs": 5, "seed": 12})
        replay = client.post("/api/cohorts", json={"runs": 5, "seed": 12})
        competing_run = client.post("/api/run", json={"target_fruit": "pear"})
        stopped = client.post("/api/cohorts/active/stop")
        cohort = wait_for_cohort(client)

    assert first.status_code == 201
    assert replay.status_code == 409
    assert competing_run.status_code == 409
    assert "cohort owns Demo Run activation" in competing_run.json()["detail"]
    assert stopped.status_code == 200
    assert cohort["status"] == "STOPPED"
    assert len(cohort["runs"]) <= 1


def test_cohort_request_defaults_are_explicit_and_validation_is_closed(tmp_path) -> None:
    app = create_app(
        runs_root=tmp_path / "runs",
        cohorts_root=tmp_path / "cohorts",
        hardware=ReadyHardware(),
        camera_perception_status=ready_camera,
        stage_executor=SimulatedStageExecutor(
            delay_at="turn_to_fruit", delay_s=0.5
        ),
    )

    with TestClient(app) as client:
        defaulted = client.post("/api/cohorts", json={"seed": 77})
        body = defaulted.json()["cohort"]
        invalid = client.post(
            "/api/cohorts",
            json={"runs": 2, "randomized": False, "seed": 1},
        )
        client.post("/api/cohorts/active/stop")

    assert defaulted.status_code == 201
    assert body["policy"]["runs"] == 5
    assert body["policy"]["randomized"] is True
    assert body["policy"]["fruit_subset"] is None
    assert body["selected_fruits"] == ["apple", "banana", "pear"]
    assert body["policy"]["tolerated_failures"] == []
    assert invalid.status_code in {409, 422}


def test_audience_ui_exposes_cohort_configuration_and_observation() -> None:
    page = TestClient(create_app()).get("/").text

    assert 'id="cohort-runs"' in page
    assert 'value="5"' in page
    assert 'id="cohort-randomized"' in page
    assert 'id="cohort-fixed-fruit"' in page
    assert 'data-cohort-fruit value="apple"' in page
    assert 'data-cohort-fruit value="banana"' in page
    assert 'data-cohort-fruit value="pear"' in page
    assert 'id="cohort-search-yaw-rps"' in page
    assert 'id="cohort-home-align-yaw-rps"' in page
    assert "fruit_subset: cohortRandomized.checked ? selectedCohortFruits() : null" in page
    assert "search: {yaw_rps: Number(cohortSearchYaw.value)}" in page
    assert "home: {align_yaw_rps: Number(cohortHomeAlignYaw.value)}" in page
    assert 'id="start-three-fruit"' in page
    assert "runs: 3" in page
    assert "Run Apple + Banana + Pear once" in page
    assert "Failures stop the cohort by default" in page
    assert "'/api/cohorts'" in page
    assert "'/api/cohorts/active'" in page
    assert "'/api/cohorts/active/stop'" in page
