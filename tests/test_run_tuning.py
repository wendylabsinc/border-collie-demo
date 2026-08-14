from __future__ import annotations

import asyncio
import json
import math

import pytest
from fastapi.testclient import TestClient
from test_api import ReadyHardwareBoundary, ready_camera_perception

from border_collie_demo.api import create_app
from border_collie_demo.guidance import GuidancePhase
from border_collie_demo.models import MissionPhase
from border_collie_demo.orchestrator import StageContext
from border_collie_demo.production import ProductionStageExecutor
from border_collie_demo.run_tuning import RunTuning


def test_defaults_keep_target_confidence_and_share_the_search_contract() -> None:
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
    assert {fruit.search.focus_yaw_rps for fruit in (apple, pear, banana)} == {0.40}
    assert pear.search.focus_missing_grace_s == 0.50
    assert pear.search.bearing_routing_enabled is False
    assert pear.centering.lock_tolerance_ratio == 0.08
    assert pear.approach.forward_mps == 1.0
    assert pear.approach.slow_inference_grace_s == 0.50
    assert pear.arrival.near_bottom_ratio == 0.90
    assert pear.arrival.disappearance_bottom_ratio == 0.80
    assert pear.arrival.final_push_mps == 0.60
    assert pear.arrival.final_push_duration_s == 1.0
    assert pear.home.align_yaw_rps == 0.80
    assert pear.home.arrival_tolerance_m == 0.50
    assert pear.home.settle_interval_s == 0.30
    assert pear.home.settled_sample_count == 4
    assert pear.home.settled_maximum_spread_m == 0.03
    assert pear.home.settled_sample_timeout_s == 1.0
    assert pear.home.settled_retry_count == 1


def test_apple_40_percent_defaults_round_trip_through_activation_validation() -> None:
    defaults = RunTuning.defaults("apple")

    effective = RunTuning.from_payload("apple", defaults.to_dict())

    assert effective.recognition.focus_confidence == 0.40
    assert effective.recognition.lock_confidence == 0.40


def test_focus_yaw_default_allows_a_bounded_per_run_override() -> None:
    contract = RunTuning.contract()

    assert {
        fruit["defaults"]["search"]["focus_yaw_rps"]
        for fruit in contract["fruits"].values()
    } == {0.40}
    assert (
        RunTuning.from_payload(
            "banana", {"search": {"focus_yaw_rps": 0.20}}
        ).search.focus_yaw_rps
        == 0.20
    )
    with pytest.raises(ValueError, match="within"):
        RunTuning.from_payload("banana", {"search": {"focus_yaw_rps": 0.41}})


def test_bearing_routing_default_is_env_backed_and_frozen_per_run(
    monkeypatch,
) -> None:
    monkeypatch.setenv("BORDER_COLLIE_BEARING_ROUTING_ENABLED", "1")

    from_env = RunTuning.defaults("pear")
    overridden = RunTuning.from_payload(
        "pear", {"search": {"bearing_routing_enabled": False}}
    )

    assert from_env.search.bearing_routing_enabled is True
    assert overridden.search.bearing_routing_enabled is False
    assert overridden.to_dict()["search"]["bearing_routing_enabled"] is False


def test_home_turn_yaw_default_is_env_backed_without_changing_search(
    monkeypatch,
) -> None:
    monkeypatch.setenv("BORDER_COLLIE_HOME_ALIGN_YAW_RPS", "0.75")

    tuning = RunTuning.defaults("pear")

    assert tuning.home.align_yaw_rps == 0.75
    assert tuning.search.yaw_rps == 0.40
    assert tuning.search.focus_yaw_rps == 0.40


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"surprise": 1}, "unknown run tuning"),
        ({"search": {"surprise": 1}}, "unknown search tuning"),
        ({"search": {"yaw_rps": float("nan")}}, "finite"),
        ({"search": {"yaw_rps": 0.20}}, "within"),
        ({"recognition": {"required_frames": 3.0}}, "integer"),
        ({"search": {"bearing_routing_enabled": 1}}, "boolean"),
        (
            {"centering": {"lock_tolerance_ratio": 0.20, "outer_corridor_ratio": 0.20}},
            "smaller",
        ),
        (
            {"approach": {"duplicate_hold_s": 0.25, "detection_maximum_age_s": 0.20}},
            "duplicate hold",
        ),
        (
            {
                "arrival": {
                    "near_bottom_ratio": 0.80,
                    "disappearance_bottom_ratio": 0.80,
                }
            },
            "below direct Arrival",
        ),
        ({"home": {"return_minimum_yaw_rps": 0.45}}, "within"),
        (
            {
                "home": {
                    "return_minimum_yaw_rps": 0.65,
                    "return_yaw_rps": 0.60,
                }
            },
            "cannot exceed",
        ),
        (
            {
                "home": {
                    "return_yaw_deadband_deg": 15.0,
                    "heading_gate_deg": 15.0,
                }
            },
            "smaller",
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
            assert {"name", "label", "unit", "safety"} <= set(field)
            if field.get("type") == "boolean":
                assert isinstance(field["default"], bool)
            else:
                assert {"minimum", "maximum", "step"} <= set(field)
    assert "exact-zero disarm" in contract["hard_invariants"]
    search_fields = {field["name"]: field for field in contract["groups"]["search"]}
    assert search_fields["bearing_routing_enabled"] == {
        "name": "bearing_routing_enabled",
        "label": "Use mapped fruit bearing",
        "unit": "boolean",
        "type": "boolean",
        "default": False,
        "target_specific": False,
        "safety": (
            "Controls yaw routing only; mapping stays active and selected-target "
            "guidance still owns translation and Arrival."
        ),
    }


def test_home_return_minimum_yaw_is_a_validated_runtime_control() -> None:
    tuning = RunTuning.from_payload(
        "pear",
        {
            "home": {
                "return_minimum_yaw_rps": 0.55,
                "return_yaw_rps": 0.65,
            }
        },
    )

    assert tuning.home.return_minimum_yaw_rps == 0.55
    fields = {field["name"]: field for field in RunTuning.contract()["groups"]["home"]}
    assert fields["return_minimum_yaw_rps"] == {
        "name": "return_minimum_yaw_rps",
        "label": "Minimum moving Home yaw",
        "unit": "rad/s",
        "minimum": 0.5,
        "maximum": 0.8,
        "step": 0.05,
        "safety": "Never lower than the physically verified 0.50 rad/s turning signal.",
        "target_specific": False,
    }


def test_home_return_yaw_deadband_is_a_validated_runtime_control() -> None:
    tuning = RunTuning.from_payload(
        "pear",
        {"home": {"return_yaw_deadband_deg": 7.0}},
    )

    assert tuning.home.return_yaw_deadband_deg == 7.0
    fields = {field["name"]: field for field in RunTuning.contract()["groups"]["home"]}
    assert fields["return_yaw_deadband_deg"]["unit"] == "deg"
    assert fields["return_yaw_deadband_deg"]["minimum"] == 3
    assert fields["return_yaw_deadband_deg"]["maximum"] == 15


def test_settled_home_contract_is_env_backed_bounded_and_visible_to_the_ui(
    monkeypatch,
) -> None:
    monkeypatch.setenv("BORDER_COLLIE_HOME_ARRIVAL_TOLERANCE_M", "0.45")
    monkeypatch.setenv("BORDER_COLLIE_HOME_SETTLE_INTERVAL_S", "0.45")
    monkeypatch.setenv("BORDER_COLLIE_HOME_SETTLED_SAMPLE_COUNT", "5")
    monkeypatch.setenv("BORDER_COLLIE_HOME_SETTLED_MAX_SPREAD_M", "0.025")
    monkeypatch.setenv("BORDER_COLLIE_HOME_SETTLED_SAMPLE_TIMEOUT_S", "1.5")
    monkeypatch.setenv("BORDER_COLLIE_HOME_SETTLED_RETRY_COUNT", "0")

    tuning = RunTuning.defaults("pear")
    fields = {field["name"]: field for field in RunTuning.contract()["groups"]["home"]}

    assert tuning.home.arrival_tolerance_m == 0.45
    assert tuning.home.settle_interval_s == 0.45
    assert tuning.home.settled_sample_count == 5
    assert tuning.home.settled_maximum_spread_m == 0.025
    assert tuning.home.settled_sample_timeout_s == 1.5
    assert tuning.home.settled_retry_count == 0
    assert fields["settle_interval_s"]["unit"] == "s"
    assert fields["settled_sample_count"]["minimum"] == 3
    assert fields["settled_sample_count"]["maximum"] == 5
    assert fields["settled_maximum_spread_m"]["unit"] == "m"
    assert fields["settled_retry_count"]["maximum"] == 1
    assert fields["arrival_tolerance_m"]["unit"] == "m"
    assert fields["arrival_tolerance_m"]["maximum"] == 0.50

    with pytest.raises(ValueError, match="settled_sample_count"):
        RunTuning.from_payload("pear", {"home": {"settled_sample_count": 2}})
    with pytest.raises(ValueError, match="settled_retry_count"):
        RunTuning.from_payload("pear", {"home": {"settled_retry_count": 2}})
    monkeypatch.setenv("BORDER_COLLIE_HOME_ARRIVAL_TOLERANCE_M", "0.51")
    with pytest.raises(ValueError, match="arrival_tolerance_m"):
        RunTuning.defaults("pear")


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
        "arrival": {
            "disappearance_bottom_ratio": 0.82,
            "final_push_mps": 0.65,
            "final_push_duration_s": 0.40,
        },
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
        routing_conflict = client.post(
            "/api/run",
            json={
                "target_fruit": "pear",
                "activation_id": "same",
                "tuning": {
                    "arrival": {"final_push_mps": 0.60},
                    "search": {"bearing_routing_enabled": True},
                },
            },
        )
        assert first.status_code == 201
        assert replay.json()["idempotent_replay"] is True
        assert conflict.status_code == 409
        assert routing_conflict.status_code == 409


def test_ui_builds_controls_from_server_schema_and_shows_effective_values() -> None:
    response = TestClient(create_app()).get("/")
    assert response.status_code == 200
    assert 'id="run-tuning-fields"' in response.text
    assert 'id="effective-tuning"' in response.text
    assert "current.run_tuning" in response.text
    assert "tuning: tuningPayload()" in response.text
    assert "field.type === 'boolean'" in response.text


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
            "arrival": {
                "disappearance_bottom_ratio": 0.82,
                "final_push_mps": 0.65,
                "final_push_duration_s": 0.40,
            },
            "home": {
                "align_yaw_rps": 0.60,
                "return_forward_mps": 0.75,
                "return_yaw_deadband_deg": 7.0,
                "return_minimum_yaw_rps": 0.55,
                "return_yaw_rps": 0.65,
                "arrival_tolerance_m": 0.20,
                "settle_interval_s": 0.45,
                "settled_sample_count": 5,
                "settled_maximum_spread_m": 0.025,
                "settled_sample_timeout_s": 1.5,
                "settled_retry_count": 0,
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
    assert hardware.guidance.config.disappearance_bottom_ratio == 0.82
    assert hardware.guidance.policy.close_range_tracking_confidence == 0.60
    assert hardware.home_calls[0][1]["yaw_rps"] == 0.60
    assert hardware.home_calls[1][1]["forward_mps"] == 0.75
    assert hardware.home_calls[1][1]["arrival_tolerance_m"] == 0.20
    assert hardware.home_calls[1][1]["heading_tolerance_rad"] == pytest.approx(
        math.radians(7.0)
    )
    assert hardware.home_calls[1][1]["minimum_yaw_rps"] == 0.55
    assert hardware.home_calls[1][1]["maximum_yaw_rps"] == 0.65
    assert hardware.home_calls[1][1]["settle_interval_s"] == 0.45
    assert hardware.home_calls[1][1]["settled_sample_count"] == 5
    assert hardware.home_calls[1][1]["settled_maximum_spread_m"] == 0.025
    assert hardware.home_calls[1][1]["settled_sample_timeout_s"] == 1.5
    assert hardware.home_calls[1][1]["settled_retry_count"] == 0
