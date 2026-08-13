from __future__ import annotations

import asyncio
import json

import pytest
from fastapi.testclient import TestClient
from test_api import ReadyHardwareBoundary, ready_camera_perception

from border_collie_demo.api import create_app
from border_collie_demo.guidance import GuidancePhase
from border_collie_demo.models import MissionPhase
from border_collie_demo.orchestrator import StageContext
from border_collie_demo.production import ProductionStageExecutor
from border_collie_demo.run_tuning import RunTuning


def test_defaults_preserve_kinda_good_behavior_for_every_fruit() -> None:
    apple = RunTuning.defaults("apple")
    pear = RunTuning.defaults("pear")
    banana = RunTuning.defaults("banana")

    assert apple.recognition.focus_confidence == 0.40
    assert apple.recognition.lock_confidence == 0.40
    assert apple.recognition.tracking_confidence == 0.10
    assert pear.recognition.focus_confidence is None
    assert pear.recognition.lock_confidence == 0.65
    assert pear.recognition.tracking_confidence == 0.55
    assert banana.recognition.focus_confidence is None
    assert banana.recognition.lock_confidence == 0.20
    assert pear.search.yaw_rps == 0.40
    assert pear.centering.lock_tolerance_ratio == 0.08
    assert pear.approach.forward_mps == 1.0
    assert pear.arrival.final_push_mps == 0.60
    assert pear.arrival.final_push_duration_s == 1.0
    assert pear.home.arrival_tolerance_m == 0.10


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"surprise": 1}, "unknown run tuning"),
        ({"search": {"surprise": 1}}, "unknown search tuning"),
        ({"search": {"yaw_rps": float("nan")}}, "finite"),
        ({"search": {"yaw_rps": 0.20}}, "within"),
        ({"recognition": {"required_frames": 3.0}}, "integer"),
        (
            {"centering": {"lock_tolerance_ratio": 0.20, "outer_corridor_ratio": 0.20}},
            "smaller",
        ),
        (
            {"approach": {"duplicate_hold_s": 0.25, "detection_maximum_age_s": 0.20}},
            "duplicate hold",
        ),
    ],
)
def test_invalid_tuning_is_rejected_at_the_one_public_seam(payload, message) -> None:
    with pytest.raises(ValueError, match=message):
        RunTuning.from_payload("pear", payload)


def test_target_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="Target Fruit"):
        RunTuning.from_payload("pear", {"target_fruit": "apple"})


def test_server_contract_documents_every_control() -> None:
    contract = RunTuning.contract()
    assert contract["schema_version"] == 1
    assert set(contract["groups"]) == set(RunTuning.GROUPS)
    for fields in contract["groups"].values():
        for field in fields:
            assert {
                "name",
                "label",
                "unit",
                "minimum",
                "maximum",
                "step",
                "safety",
            } <= set(field)
    assert "exact-zero disarm" in contract["hard_invariants"]


def test_activation_persists_exact_effective_tuning_in_result_and_black_box(
    tmp_path,
) -> None:
    app = create_app(
        runs_root=tmp_path,
        hardware=ReadyHardwareBoundary(),
        camera_perception_status=ready_camera_perception,
    )
    requested = {
        "target_fruit": "pear",
        "search": {"yaw_rps": 0.45},
        "recognition": {"tracking_confidence": 0.60},
        "centering": {"outer_corridor_ratio": 0.25},
        "approach": {"forward_mps": 0.75},
        "arrival": {"final_push_mps": 0.65, "final_push_duration_s": 0.40},
        "home": {"arrival_tolerance_m": 0.20},
    }
    with TestClient(app) as client:
        response = client.post(
            "/api/run",
            json={
                "target_fruit": "pear",
                "activation_id": "tuned-run",
                "tuning": requested,
            },
        )
        assert response.status_code == 201
        run = response.json()["run"]
        effective = RunTuning.from_payload("pear", requested).to_dict()
        assert run["run_tuning"] == effective
        trace = client.get(f"/api/results/{run['run_id']}/black-box.ndjson").text
        started = next(
            json.loads(line)
            for line in trace.splitlines()
            if '"kind":"run_started"' in line
        )
        assert started["payload"]["run_tuning"] == effective


def test_invalid_tuning_precedes_target_selection_or_run_creation(tmp_path) -> None:
    selected: list[str] = []
    app = create_app(
        runs_root=tmp_path,
        hardware=ReadyHardwareBoundary(),
        camera_perception_status=ready_camera_perception,
        select_perception_target=lambda fruit: selected.append(fruit) or {},
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/run",
            json={
                "target_fruit": "pear",
                "tuning": {"arrival": {"final_push_mps": 1.50}},
            },
        )
        assert response.status_code == 409
        assert selected == []
        assert client.get("/api/results").json() == {"runs": []}


def test_idempotency_includes_the_complete_tuning_snapshot(tmp_path) -> None:
    app = create_app(
        runs_root=tmp_path,
        hardware=ReadyHardwareBoundary(),
        camera_perception_status=ready_camera_perception,
    )
    with TestClient(app) as client:
        first = client.post(
            "/api/run",
            json={
                "target_fruit": "pear",
                "activation_id": "same",
                "tuning": {"arrival": {"final_push_mps": 0.60}},
            },
        )
        replay = client.post(
            "/api/run",
            json={
                "target_fruit": "pear",
                "activation_id": "same",
                "tuning": {"arrival": {"final_push_mps": 0.60}},
            },
        )
        conflict = client.post(
            "/api/run",
            json={
                "target_fruit": "pear",
                "activation_id": "same",
                "tuning": {"arrival": {"final_push_mps": 0.65}},
            },
        )
        assert first.status_code == 201
        assert replay.json()["idempotent_replay"] is True
        assert conflict.status_code == 409


def test_ui_builds_controls_from_server_schema_and_shows_effective_values() -> None:
    response = TestClient(create_app()).get("/")
    assert response.status_code == 200
    assert 'id="run-tuning-fields"' in response.text
    assert 'id="effective-tuning"' in response.text
    assert "current.run_tuning" in response.text
    assert "tuning: tuningPayload()" in response.text


def test_production_consumes_guidance_and_home_values_from_the_snapshot() -> None:
    class Hardware:
        def __init__(self) -> None:
            self.guidance = None
            self.home_calls: list[tuple[str, dict[str, object]]] = []

        async def guide_target(self, _reader, guidance, *, allow_forward, timeout_s):
            self.guidance = guidance
            guidance.phase = GuidancePhase.LOCKED
            return {
                "label": "pear",
                "timeout_s": timeout_s,
                "motion_commands_sent": False,
            }

        async def turn_toward_home(self, _home, **options):
            self.home_calls.append(("align", options))
            return {
                "home_bearing_error_rad": 0.0,
                "motion_path": "sport_yaw",
                "motion_commands_sent": False,
            }

        async def return_home_position(self, _home, **options):
            self.home_calls.append(("return", options))
            return {"motion_commands_sent": True}

    class Bark:
        async def bark(self):
            return {}

    tuning = RunTuning.from_payload(
        "pear",
        {
            "search": {"yaw_rps": 0.55, "timeout_s": 22.0},
            "recognition": {"tracking_confidence": 0.60, "required_frames": 4},
            "centering": {"outer_corridor_ratio": 0.25, "approach_yaw_rps": 0.40},
            "approach": {"forward_mps": 0.75},
            "arrival": {"final_push_mps": 0.65, "final_push_duration_s": 0.40},
            "home": {
                "align_yaw_rps": 0.60,
                "return_forward_mps": 0.75,
                "return_yaw_rps": 0.60,
                "arrival_tolerance_m": 0.20,
            },
        },
    )
    hardware = Hardware()
    executor = ProductionStageExecutor(hardware, dict, Bark())
    context = StageContext(
        run_id="tuned",
        target_fruit="pear",
        home={"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0},
        run_tuning=tuning.to_dict(),
    )

    async def scenario() -> None:
        await executor.execute(MissionPhase.TURN_TO_FRUIT, context)
        await executor.execute(MissionPhase.TURN_TOWARD_HOME, context)
        await executor.execute(MissionPhase.RETURN_HOME, context)

    asyncio.run(scenario())

    assert hardware.guidance.config.search_yaw_rps == 0.55
    assert hardware.guidance.config.center_confirmations == 4
    assert hardware.guidance.config.approach_forward_mps == 0.75
    assert hardware.guidance.config.outer_corridor_ratio == 0.25
    assert hardware.guidance.config.final_push_mps == 0.65
    assert hardware.guidance.policy.close_range_tracking_confidence == 0.60
    assert hardware.home_calls[0][1]["yaw_rps"] == 0.60
    assert hardware.home_calls[1][1]["forward_mps"] == 0.75
    assert hardware.home_calls[1][1]["arrival_tolerance_m"] == 0.20
