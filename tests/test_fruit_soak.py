import json
from collections import Counter
from pathlib import Path

import pytest

from scripts.fruit_soak import (
    HarnessAbort,
    TempSource,
    ThreadedTempSampler,
    aggregate_stage_telemetry,
    dongle_check,
    draw_fruit_sequence,
    draw_orientation_sequence,
    resolve_failed_run_recovery,
    run_session,
    summarize_network,
    summarize_run,
    verify_live_preflight,
    wait_for_ready,
    wait_for_recovery,
    wait_for_terminal,
)


class FakeClient:
    """Scripted API double: statuses, run results, and activations."""

    def __init__(
        self,
        statuses,
        results_by_id=None,
        qualified=("apple", "banana", "pear"),
        sidecar=None,
    ):
        self._statuses = list(statuses)
        self._results = dict(results_by_id or {})
        self._qualified = list(qualified)
        self._sidecar = sidecar
        self.activated: list[str] = []
        self.orientation_degrees: list[float] = []
        self.idempotency_keys: list[str | None] = []
        self.search_policies: list[str | None] = []
        self.stop_calls = 0
        self.recover_home_calls: list[str] = []

    def status(self):
        return self._statuses.pop(0) if len(self._statuses) > 1 else self._statuses[0]

    def timed_status(self):
        try:
            return self.status(), 12.0, None
        except Exception as exc:  # noqa: BLE001
            return None, 12.0, str(exc)

    def sidecar_status(self):
        if self._sidecar is None:
            return None, "sidecar down"
        if isinstance(self._sidecar, list):
            payload = (
                self._sidecar.pop(0)
                if len(self._sidecar) > 1
                else self._sidecar[0]
            )
            return payload, None
        return self._sidecar, None

    def fruits(self):
        return {"qualified_fruits": self._qualified}

    def activate(
        self,
        fruit,
        orientation_degrees=0.0,
        idempotency_key=None,
        search_policy=None,
    ):
        self.activated.append(fruit)
        self.orientation_degrees.append(orientation_degrees)
        self.idempotency_keys.append(idempotency_key)
        self.search_policies.append(search_policy)
        run_id = f"run-{len(self.activated)}"
        return {"run": {"run_id": run_id}}

    def result(self, run_id):
        payloads = self._results[run_id]
        return payloads.pop(0) if len(payloads) > 1 else payloads[0]

    def stop(self):
        self.stop_calls += 1
        return {}

    def recover_home(self, run_id):
        self.recover_home_calls.append(run_id)
        raise AssertionError("unexpected manual Home recovery request")

    def camera_frame(self):
        return b"\xff\xd8fake-jpeg-bytes\xff\xd9"


READY = {
    "build_label": "base-soak-v2-orientation (demo/base)",
    "mission": {"restart_required": False, "phase": "idle"},
    "activation": {"ready": True, "blockers": []},
    "active_run_id": None,
    "active_recovery": None,
    "hardware": {"motion": {"armed": False}},
}
LATCHED = {
    "build_label": "base-soak-v2-orientation (demo/base)",
    "mission": {"restart_required": True, "reason": "REMOTE_TAKEOVER"},
    "activation": {"ready": False, "blockers": []},
    "active_run_id": None,
}
SIDECAR = {
    "source": {"width": 1280, "height": 720},
    "detection": {
        "label": "pear",
        "confidence": 0.81,
        "bbox_xyxy": [500.0, 300.0, 700.0, 648.0],
        "inference_s": 0.06,
    },
}

LIVE_PREFLIGHT_READY = {
    "build_label": "stage-camera-v28-lie-down-evidence (codex/test)",
    "search_policy": "slow-sweep",
    "release": {
        "release_id": "stage-camera-v28-lie-down-evidence",
        "config_schema": 12,
        "service": "app",
    },
    "mission": {
        "restart_required": False,
        "phase": "idle",
        "remote_takeover_latched": False,
    },
    "activation": {"ready": True, "blockers": []},
    "active_run_id": None,
    "active_recovery": None,
    "hardware": {
        "connected": True,
        "fault": None,
        "active_operation": None,
        "pose": {"healthy": True, "age_s": 0.02},
        "motion": {
            "armed": False,
            "last_command": {"forward_mps": 0.0, "yaw_rps": 0.0},
            "guardian": {"active": False},
        },
    },
}
LIVE_PREFLIGHT_SIDECAR = {
    "release": {
        "release_id": "stage-camera-v28-lie-down-evidence",
        "config_schema": 12,
        "service": "media",
    },
    "generation": "generation-1",
    "source": {"pts": 12345},
    "supervision": {
        "ready": True,
        "restart_required": False,
        "last_frame_age_s": 0.03,
    },
    "bark_ready": True,
    "error": None,
}


def test_live_preflight_proves_readiness_without_activation() -> None:
    advanced = {
        **LIVE_PREFLIGHT_SIDECAR,
        "source": {"pts": 12420},
    }
    client = FakeClient(
        [LIVE_PREFLIGHT_READY],
        sidecar=[LIVE_PREFLIGHT_SIDECAR, advanced],
    )

    report = verify_live_preflight(
        client,
        expected_build_label="stage-camera-v28-lie-down-evidence (codex/test)",
        expected_search_policy="slow-sweep",
        expected_fruits=["apple", "banana", "pear"],
        sleep=lambda _: None,
    )

    assert report == {
        "ready": True,
        "build_label": "stage-camera-v28-lie-down-evidence (codex/test)",
        "release_id": "stage-camera-v28-lie-down-evidence",
        "config_schema": 12,
        "search_policy": "slow-sweep",
        "qualified_fruits": ["apple", "banana", "pear"],
        "camera_generation": "generation-1",
        "source_pts": 12420,
        "pose_age_s": 0.02,
        "last_frame_age_s": 0.03,
        "motion_disarmed": True,
        "camera_frame_bytes": len(b"\xff\xd8fake-jpeg-bytes\xff\xd9"),
    }
    assert client.activated == []


@pytest.mark.parametrize(
    ("status_update", "sidecar_update", "message"),
    [
        (
            {"hardware": {**LIVE_PREFLIGHT_READY["hardware"], "motion": {"armed": True}}},
            {},
            "motion is not disarmed",
        ),
        ({}, {"generation": None}, "camera generation is unavailable"),
    ],
)
def test_live_preflight_fails_closed(status_update, sidecar_update, message) -> None:
    status = {**LIVE_PREFLIGHT_READY, **status_update}
    sidecar = {**LIVE_PREFLIGHT_SIDECAR, **sidecar_update}
    client = FakeClient([status], sidecar=[sidecar, sidecar])

    with pytest.raises(HarnessAbort, match=message):
        verify_live_preflight(client, sleep=lambda _: None)

    assert client.activated == []


def test_live_preflight_rejects_frozen_camera_pts() -> None:
    client = FakeClient(
        [LIVE_PREFLIGHT_READY],
        sidecar=[LIVE_PREFLIGHT_SIDECAR, LIVE_PREFLIGHT_SIDECAR],
    )

    with pytest.raises(HarnessAbort, match="camera source PTS did not advance"):
        verify_live_preflight(client, sleep=lambda _: None)

    assert client.activated == []


def terminal(run_id, outcome="COMPLETED", home=0.05):
    return {
        "run": {
            "run_id": run_id,
            "outcome": outcome,
            "reason": "SUCCESS" if outcome == "COMPLETED" else outcome,
            "message": "done",
            "terminal_measurements": {"home_distance_m": home, "heading_error_rad": 0.01},
            "stage_results": {"approach_fruit": {"forward_pulse_count": 12}},
            "events": [
                {"phase": "turn_to_fruit", "seconds_since_start": 1.0},
                {"phase": "approach_fruit", "seconds_since_start": 5.5},
                {"phase": "complete", "seconds_since_start": 40.0},
            ],
        }
    }


def test_sequence_is_seeded_and_only_qualified():
    first = draw_fruit_sequence(["pear", "apple", "banana"], 10, seed=7)
    second = draw_fruit_sequence(["banana", "apple", "pear"], 10, seed=7)
    assert first == second
    assert set(first) <= {"pear", "apple", "banana"}
    counts = [first.count(fruit) for fruit in ("apple", "banana", "pear")]
    assert max(counts) - min(counts) <= 1
    assert min(counts) >= 3
    assert draw_fruit_sequence(["pear"], 3, seed=1) == ["pear", "pear", "pear"]


def test_sequence_requires_qualified_fruits():
    with pytest.raises(HarnessAbort):
        draw_fruit_sequence([], 10, seed=7)


def test_sequence_requires_a_positive_run_count():
    with pytest.raises(HarnessAbort, match="greater than zero"):
        draw_fruit_sequence(["pear"], 0, seed=7)


def test_orientation_sequence_is_seeded_and_covers_the_full_heading_range():
    first = draw_orientation_sequence(10, seed=20260810)
    second = draw_orientation_sequence(10, seed=20260810)

    assert first == second
    assert len(first) == 10
    assert all(isinstance(angle, int) and 0 <= angle < 360 for angle in first)
    assert len(set(first)) > 1
    assert first != draw_orientation_sequence(10, seed=20260811)


def test_wait_for_ready_aborts_on_restart_required():
    with pytest.raises(HarnessAbort, match="restart-required"):
        wait_for_ready(FakeClient([LATCHED]), sleep=lambda _: None)


def test_wait_for_ready_times_out_with_blockers():
    blocked = {
        "mission": {"restart_required": False},
        "activation": {
            "ready": False,
            "blockers": [{"name": "camera_perception_ready", "detail": "down"}],
        },
        "active_run_id": None,
    }
    clock_values = iter([0.0, 0.0, 100.0, 100.0])
    with pytest.raises(HarnessAbort, match="camera_perception_ready"):
        wait_for_ready(
            FakeClient([blocked]),
            sleep=lambda _: None,
            clock=lambda: next(clock_values),
        )


def test_wait_for_terminal_stops_robot_on_overrun():
    pending = {"run": {"run_id": "run-1", "outcome": None}}
    client = FakeClient([READY], results_by_id={"run-1": [pending]})
    clock_values = iter([0.0] * 4 + [500.0] * 4)
    _run, samples, note = wait_for_terminal(
        client,
        "run-1",
        target_fruit="pear",
        sleep=lambda _: None,
        clock=lambda: next(clock_values),
    )
    assert client.stop_calls == 1
    assert "harness stop" in note
    assert samples


def test_wait_for_terminal_samples_confidence_and_proximity():
    running = {"run": {"run_id": "run-1", "outcome": None}}
    approach = dict(READY, mission={"restart_required": False, "phase": "approach_fruit"})
    client = FakeClient(
        [approach, approach, approach],
        results_by_id={"run-1": [running, running, terminal("run-1")]},
        sidecar=SIDECAR,
    )
    _run, samples, note = wait_for_terminal(
        client, "run-1", target_fruit="pear", sleep=lambda _: None
    )
    assert note is None
    assert samples[0]["phase"] == "approach_fruit"
    assert samples[0]["confidence"] == 0.81
    assert samples[0]["target_matches"] is True
    assert samples[0]["bbox_bottom_ratio"] == 0.9  # 648 / 720


def test_wait_for_terminal_reconciles_a_transient_result_timeout():
    class TransientResultClient(FakeClient):
        def __init__(self):
            super().__init__(
                [READY],
                results_by_id={"run-1": [terminal("run-1")]},
                sidecar=SIDECAR,
            )
            self.calls = 0

        def result(self, run_id):
            self.calls += 1
            if self.calls == 1:
                raise TimeoutError("transient result timeout")
            return super().result(run_id)

    run, samples, note = wait_for_terminal(
        TransientResultClient(),
        "run-1",
        target_fruit="pear",
        sleep=lambda _: None,
    )

    assert run["outcome"] == "COMPLETED"
    assert note is None
    assert samples[0]["result_error"] == "transient result timeout"


def test_wait_for_recovery_reconciles_transient_poll_errors():
    class RecoveryPollClient:
        calls = 0
        stop_calls = 0

        def result(self, run_id):
            self.calls += 1
            if self.calls == 1:
                raise TimeoutError("link timeout")
            return {
                "run": {
                    "recovery_attempts": [
                        {
                            "recovery_id": "recovery-1",
                            "outcome": "COMPLETED",
                            "reason": "HOME_POSITION_RECOVERED",
                        }
                    ]
                }
            }

        def stop(self):
            self.stop_calls += 1

    attempt, errors = wait_for_recovery(
        RecoveryPollClient(),
        "run-1",
        "recovery-1",
        sleep=lambda _: None,
    )

    assert attempt["outcome"] == "COMPLETED"
    assert errors == ["link timeout"]


def test_recovery_resolution_follows_active_client_state_without_manual_post():
    failed = terminal("run-1", outcome="FAILED", home=0.4)
    pending = {
        "recovery_id": "automatic-active",
        "outcome": None,
        "reason": None,
    }
    recovering = json.loads(json.dumps(failed))
    recovering["run"]["recovery_attempts"] = [pending]
    complete = json.loads(json.dumps(failed))
    complete["run"]["recovery_attempts"] = [
        {
            **pending,
            "outcome": "COMPLETED",
            "reason": "HOME_POSITION_RECOVERED",
            "final_safety_state": "DISARMED_CONFIRMED",
            "final_evidence": {"home_distance_m": 0.08},
        }
    ]

    class ActiveRecoveryClient(FakeClient):
        status_calls = 0

        def status(self):
            self.status_calls += 1
            if self.status_calls == 1:
                return {
                    **READY,
                    "activation": {"ready": False, "blockers": []},
                    "active_recovery": {
                        "run_id": "run-1",
                        "recovery_id": "automatic-active",
                    },
                }
            return READY

    client = ActiveRecoveryClient(
        [READY],
        results_by_id={"run-1": [failed, recovering, complete]},
        sidecar=SIDECAR,
    )

    record, errors = resolve_failed_run_recovery(
        client,
        "run-1",
        request_manual=True,
        sleep=lambda _: None,
    )

    assert errors == []
    assert client.recover_home_calls == []
    assert record["outcome"] == "COMPLETED"
    assert record["safe_to_continue"] is True


def test_aggregate_stage_telemetry_groups_by_phase():
    samples = [
        {
            "phase": "approach_fruit",
            "app_latency_ms": 10.0,
            "target_matches": True,
            "confidence": 0.8,
            "bbox_bottom_ratio": 0.5,
            "temps": {"gpu": 50.0},
        },
        {
            "phase": "approach_fruit",
            "app_latency_ms": 30.0,
            "target_matches": True,
            "confidence": 0.6,
            "bbox_bottom_ratio": 0.7,
            "temps": {"gpu": 54.0},
        },
        {
            "phase": "return_home",
            "app_latency_ms": 20.0,
            "target_matches": False,
            "confidence": 0.9,
            "app_error": "timed out",
        },
    ]
    stages = aggregate_stage_telemetry(samples)
    approach = stages["approach_fruit"]
    assert approach["target_confidence"] == {
        "min": 0.6,
        "mean": 0.7,
        "max": 0.8,
        "samples": 2,
    }
    assert approach["bbox_bottom_ratio"]["max"] == 0.7
    assert approach["temps_c"]["gpu"]["max"] == 54.0
    home = stages["return_home"]
    assert home["target_confidence"] is None  # wrong-label confidence is excluded
    assert home["app_errors"] == 1


def test_summarize_network_flags_cutouts_and_latency():
    samples = [
        {"t_monotonic_s": 1.0, "app_latency_ms": 10.0},
        {"t_monotonic_s": 2.0, "app_latency_ms": 900.0},
        {"t_monotonic_s": 3.0, "app_latency_ms": 8000.0, "app_error": "timed out"},
    ]
    network = summarize_network(samples)
    assert network["cut_out"] is True
    assert network["error_count"] == 1
    assert network["errors"][0]["t_monotonic_s"] == 3.0
    assert network["latency_ms"]["samples"] == 2  # errored polls excluded from latency


def test_dongle_check_matches_across_usb_and_audio_sources():
    sources = {
        "usb_devices": ["DJI MIC MINI (2ca3:4011)", "802.11ac WLAN Adapter (0bda:0811)"],
        "audio_devices": [{"name": "hw:1,0", "description": "APE"}],
    }
    found = dongle_check(sources, "dji mic mini")
    assert found["checked"] is True
    assert found["visible"] is True
    assert found["usb_devices"] == sources["usb_devices"]
    missing = dongle_check({"usb_devices": ["xHCI Host Controller"]}, "dji mic mini")
    assert missing["visible"] is False
    disabled = dongle_check(None, "dji mic mini")
    assert disabled["checked"] is False
    all_errored = dongle_check({"usb_devices": {"error": "agent unreachable"}}, "dji")
    assert all_errored == {
        "checked": False,
        "visible": None,
        "detail": "agent unreachable",
    }
    partial = dongle_check(
        {
            "usb_devices": {"error": "agent flake"},
            "audio_devices": [{"description": "DJI MIC MINI"}],
        },
        "dji mic mini",
    )
    assert partial["visible"] is True  # matched via the source that worked
    assert partial["probe_errors"] == {"usb_devices": "agent flake"}


def test_threaded_sampler_sync_fallback_and_disabled():
    class FakeSource:
        enabled = True
        def read(self):
            return {"gpu-thermal": 51.5}, None
    sampler = ThreadedTempSampler(FakeSource())
    temps, error, age_s = sampler.latest()
    assert temps == {"gpu-thermal": 51.5}
    assert error is None
    assert age_s is not None
    disabled = ThreadedTempSampler(TempSource())
    assert disabled.enabled is False


def test_temp_source_disabled_by_default():
    source = TempSource()
    assert source.enabled is False
    assert source.read() == (None, None)


def test_session_records_every_run_and_build_label(tmp_path: Path):
    client = FakeClient(
        [READY],
        results_by_id={
            "run-1": [terminal("run-1")],
            "run-2": [terminal("run-2", outcome="FAILED", home=0.3)],
        },
        sidecar=SIDECAR,
    )
    output = tmp_path / "soak.json"
    session = run_session(
        client, runs=2, seed=7, output_path=output, sleep=lambda _: None, log=lambda *_: None
    )
    saved = json.loads(output.read_text())
    assert saved["build_label"] == "base-soak-v2-orientation (demo/base)"
    assert saved["seed"] == 7
    assert saved["fruit_sequence"] == client.activated
    assert saved["orientation_sequence_degrees"] == client.orientation_degrees
    assert [r["orientation_degrees"] for r in saved["runs"]] == client.orientation_degrees
    assert [r["outcome"] for r in saved["runs"]] == ["COMPLETED", "FAILED"]
    assert saved["runs"][1]["home_distance_m"] == 0.3
    assert saved["runs"][0]["stage_results"] == {"approach_fruit": {"forward_pulse_count": 12}}
    assert saved["runs"][0]["network"]["poll_count"] >= 1
    assert "stage_telemetry" in saved["runs"][0]
    assert saved["runs"][0]["stage_durations"]["turn_to_fruit"] == 4.5
    assert saved["runs"][0]["lighting_frame"]["bytes"] > 0
    assert Path(saved["runs"][0]["lighting_frame"]["path"]).exists()
    assert saved["scorecard"]["criteria"]["completion"]["passed"] is False  # run 2 FAILED
    assert saved["scorecard"]["recorded_only"] is True
    assert session["aborted"] is None


def test_session_downloads_lie_down_photo_beside_run_evidence(tmp_path: Path):
    jpeg = b"\xff\xd8lie-down-fruit-position\xff\xd9"
    completed = terminal("run-1")
    completed["run"]["snapshots"] = [
        {
            "kind": "lie_down",
            "context": "audience_action",
            "available": True,
            "filename": "lie-down-audience-action.jpg",
        }
    ]

    class ArtifactClient(FakeClient):
        def artifact(self, run_id, filename):
            assert run_id == "run-1"
            assert filename == "lie-down-audience-action.jpg"
            return jpeg

    client = ArtifactClient(
        [READY],
        results_by_id={"run-1": [completed]},
        qualified=("apple",),
        sidecar=SIDECAR,
    )
    output = tmp_path / "soak.json"

    run_session(
        client,
        runs=1,
        seed=9,
        output_path=output,
        sleep=lambda _: None,
        log=lambda *_: None,
    )

    record = json.loads(output.read_text())["runs"][0]
    photo = record["lie_down_frame"]
    assert photo["context"] == "audience_action"
    assert photo["filename"] == "lie-down-audience-action.jpg"
    assert photo["bytes"] == len(jpeg)
    assert Path(photo["path"]).read_bytes() == jpeg


def test_session_requires_and_records_expected_search_policy(tmp_path: Path):
    status = {**READY, "search_policy": "double-back"}
    client = FakeClient(
        [status],
        results_by_id={"run-1": [terminal("run-1")]},
        sidecar=SIDECAR,
    )

    session = run_session(
        client,
        runs=1,
        seed=3,
        output_path=tmp_path / "double-back.json",
        expected_search_policy="double-back",
        sleep=lambda _: None,
        log=lambda *_: None,
    )

    assert session["search_policy"] == "double-back"


def test_session_selects_one_immutable_search_policy_for_every_run(tmp_path: Path):
    status = {**READY, "search_policy": "slow-sweep"}
    client = FakeClient(
        [status],
        results_by_id={
            "run-1": [terminal("run-1")],
            "run-2": [terminal("run-2")],
        },
        sidecar=SIDECAR,
    )

    session = run_session(
        client,
        runs=2,
        seed=3,
        output_path=tmp_path / "fast-lock.json",
        search_policy="fast-lock",
        sleep=lambda _: None,
        log=lambda *_: None,
    )

    assert session["search_policy"] == "fast-lock"
    assert session["default_search_policy"] == "slow-sweep"
    assert client.search_policies == ["fast-lock", "fast-lock"]


def test_session_rejects_unexpected_search_policy_before_motion(tmp_path: Path):
    status = {**READY, "search_policy": "slow-sweep"}
    client = FakeClient([status], sidecar=SIDECAR)

    with pytest.raises(HarnessAbort, match="expected search policy"):
        run_session(
            client,
            runs=1,
            seed=3,
            output_path=tmp_path / "wrong-policy.json",
            expected_search_policy="fast-lock",
            sleep=lambda _: None,
            log=lambda *_: None,
        )

    assert client.activated == []


def test_session_can_run_the_regular_soak_without_orientation_turns(tmp_path: Path):
    client = FakeClient(
        [READY],
        results_by_id={
            "run-1": [terminal("run-1")],
            "run-2": [terminal("run-2")],
        },
        sidecar=SIDECAR,
    )

    session = run_session(
        client,
        runs=2,
        seed=17,
        output_path=tmp_path / "regular-soak.json",
        randomize_orientation=False,
        sleep=lambda _: None,
        log=lambda *_: None,
    )

    assert session["orientation_randomized"] is False
    assert session["orientation_sequence_degrees"] == [0, 0]
    assert client.orientation_degrees == [0, 0]


def test_session_recovers_a_failed_run_before_continuing(tmp_path: Path):
    recovery = {
        "recovery_id": "recovery-1",
        "outcome": "COMPLETED",
        "reason": "HOME_POSITION_RECOVERED",
        "final_safety_state": "DISARMED_CONFIRMED",
        "final_evidence": {"home_distance_m": 0.08},
    }
    failed = terminal("run-1", outcome="FAILED", home=0.3)
    recovered = json.loads(json.dumps(failed))
    recovered["run"]["recovery_attempts"] = [recovery]

    class RecoveryClient(FakeClient):
        def result(self, run_id):
            if run_id == "run-1" and not self.recover_home_calls:
                return failed
            return super().result(run_id)

        def recover_home(self, run_id):
            assert run_id == "run-1"
            self.recover_home_calls.append(run_id)
            return {"recovery": {"recovery_id": "recovery-1"}}

    client = RecoveryClient(
        [READY],
        results_by_id={
            "run-1": [failed, recovered],
            "run-2": [terminal("run-2")],
        },
        sidecar=SIDECAR,
    )

    session = run_session(
        client,
        runs=2,
        seed=17,
        output_path=tmp_path / "recovering-soak.json",
        recover_failures=True,
        sleep=lambda _: None,
        log=lambda *_: None,
    )

    assert session["failure_recovery_enabled"] is True
    assert client.recover_home_calls == ["run-1"]
    assert len(session["runs"]) == 2
    assert session["runs"][0]["outcome"] == "FAILED"
    assert session["runs"][0]["recovery"]["outcome"] == "COMPLETED"
    assert session["runs"][1]["outcome"] == "COMPLETED"
    assert session["aborted"] is None


def test_session_observes_automatic_recovery_before_continuing(tmp_path: Path):
    pending_recovery = {
        "recovery_id": "automatic-1",
        "outcome": None,
        "reason": None,
        "final_safety_state": None,
    }
    completed_recovery = {
        **pending_recovery,
        "outcome": "COMPLETED",
        "reason": "HOME_POSITION_RECOVERED",
        "final_safety_state": "DISARMED_CONFIRMED",
        "final_evidence": {"home_distance_m": 0.08},
    }
    failed = terminal("run-1", outcome="FAILED", home=0.3)
    recovering = json.loads(json.dumps(failed))
    recovering["run"]["recovery_attempts"] = [pending_recovery]
    recovered = json.loads(json.dumps(failed))
    recovered["run"]["recovery_attempts"] = [completed_recovery]
    active = {
        **READY,
        "activation": {"ready": False, "blockers": []},
        "active_recovery": {
            "run_id": "run-1",
            "recovery_id": "automatic-1",
        },
    }
    client = FakeClient(
        [READY, READY, active, READY, READY],
        results_by_id={
            "run-1": [failed, recovering, recovered],
            "run-2": [terminal("run-2")],
        },
        sidecar=SIDECAR,
    )

    session = run_session(
        client,
        runs=2,
        seed=17,
        output_path=tmp_path / "automatic-recovery-soak.json",
        recover_failures=True,
        sleep=lambda _: None,
        log=lambda *_: None,
    )

    assert client.recover_home_calls == []
    assert len(session["runs"]) == 2
    assert session["runs"][0]["mission_outcome"] == "FAILED"
    assert session["runs"][0]["recovery"]["outcome"] == "COMPLETED"
    assert session["runs"][0]["recovery"]["source"] == "automatic"
    assert session["runs"][0]["recovery"]["stage_home_margin_m"] == 0.5
    assert session["aborted"] is None


def test_session_waits_for_delayed_automatic_recovery(tmp_path: Path):
    pending = {
        "recovery_id": "automatic-delayed",
        "outcome": None,
        "reason": None,
    }
    complete = {
        **pending,
        "outcome": "COMPLETED",
        "reason": "HOME_POSITION_RECOVERED",
        "final_safety_state": "DISARMED_CONFIRMED",
        "final_evidence": {"home_distance_m": 0.09},
    }
    failed = terminal("run-1", outcome="FAILED", home=0.4)
    recovering = json.loads(json.dumps(failed))
    recovering["run"]["recovery_attempts"] = [pending]
    recovered = json.loads(json.dumps(failed))
    recovered["run"]["recovery_attempts"] = [complete]
    client = FakeClient(
        [READY],
        results_by_id={
            "run-1": [failed, failed, failed, recovering, recovered],
            "run-2": [terminal("run-2")],
        },
        sidecar=SIDECAR,
    )

    session = run_session(
        client,
        runs=2,
        seed=5,
        output_path=tmp_path / "delayed-auto.json",
        recover_failures=True,
        sleep=lambda _: None,
        log=lambda *_: None,
    )

    assert client.recover_home_calls == []
    assert session["runs"][0]["recovery"]["outcome"] == "COMPLETED"
    assert len(session["runs"]) == 2


@pytest.mark.parametrize(
    ("recovery_outcome", "recovery_reason"),
    [
        ("FAILED", "AUTOMATIC_RECOVERY_SKIPPED"),
        ("STOPPED", "OPERATOR_STOP"),
    ],
)
def test_session_records_unsuccessful_recovery_and_blocks_next_activation(
    tmp_path: Path,
    recovery_outcome: str,
    recovery_reason: str,
):
    failed = terminal("run-1", outcome="FAILED", home=0.7)
    settled = json.loads(json.dumps(failed))
    settled["run"]["recovery_attempts"] = [
        {
            "recovery_id": "automatic-1",
            "outcome": recovery_outcome,
            "reason": recovery_reason,
            "final_safety_state": "DISARMED_CONFIRMED",
            "final_evidence": {"home_distance_m": 0.7},
        }
    ]
    output = tmp_path / f"{recovery_reason}.json"
    client = FakeClient(
        [READY],
        results_by_id={"run-1": [failed, settled]},
        sidecar=SIDECAR,
    )

    session = run_session(
        client,
        runs=2,
        seed=5,
        output_path=output,
        recover_failures=True,
        sleep=lambda _: None,
        log=lambda *_: None,
    )

    saved = json.loads(output.read_text())
    assert client.activated == [saved["fruit_sequence"][0]]
    assert saved["runs"][0]["mission_outcome"] == "FAILED"
    assert saved["runs"][0]["recovery"]["outcome"] == recovery_outcome
    assert saved["runs"][0]["recovery"]["reason"] == recovery_reason
    assert saved["runs"][0]["recovery"]["safe_to_continue"] is False
    assert "next activation blocked" in session["aborted"]


def test_session_blocks_completed_recovery_outside_stage_home_margin(tmp_path: Path):
    failed = terminal("run-1", outcome="FAILED", home=0.8)
    recovered = json.loads(json.dumps(failed))
    recovered["run"]["recovery_attempts"] = [
        {
            "recovery_id": "automatic-1",
            "outcome": "COMPLETED",
            "reason": "HOME_POSITION_RECOVERED",
            "final_safety_state": "DISARMED_CONFIRMED",
            "final_evidence": {"home_distance_m": 0.51},
        }
    ]
    client = FakeClient(
        [READY],
        results_by_id={"run-1": [failed, recovered]},
        sidecar=SIDECAR,
    )

    session = run_session(
        client,
        runs=2,
        seed=5,
        output_path=tmp_path / "outside-margin.json",
        recover_failures=True,
        stage_home_margin_m=0.5,
        sleep=lambda _: None,
        log=lambda *_: None,
    )

    assert len(client.activated) == 1
    assert session["runs"][0]["recovery"]["outcome"] == "COMPLETED"
    assert session["runs"][0]["recovery"]["within_stage_home_margin"] is False
    assert session["runs"][0]["recovery"]["safe_to_continue"] is False


def test_session_waits_for_terminal_recovery_to_clear_client_state(tmp_path: Path):
    failed = terminal("run-1", outcome="FAILED", home=0.6)
    recovered = json.loads(json.dumps(failed))
    recovered["run"]["recovery_attempts"] = [
        {
            "recovery_id": "automatic-1",
            "outcome": "COMPLETED",
            "reason": "HOME_POSITION_RECOVERED",
            "final_safety_state": "DISARMED_CONFIRMED",
            "final_evidence": {"home_distance_m": 0.06},
        }
    ]
    still_active = {
        **READY,
        "activation": {"ready": False, "blockers": []},
        "active_recovery": {
            "run_id": "run-1",
            "recovery_id": "automatic-1",
        },
    }
    class DelayedClearClient(FakeClient):
        recovery_observed = False
        post_recovery_status_calls = 0

        def result(self, run_id):
            payload = super().result(run_id)
            if (payload.get("run") or {}).get("recovery_attempts"):
                self.recovery_observed = True
            return payload

        def status(self):
            if self.recovery_observed:
                self.post_recovery_status_calls += 1
                if self.post_recovery_status_calls <= 2:
                    return still_active
            return READY

    client = DelayedClearClient(
        [READY],
        results_by_id={
            "run-1": [failed, recovered],
            "run-2": [terminal("run-2")],
        },
        sidecar=SIDECAR,
    )

    session = run_session(
        client,
        runs=2,
        seed=5,
        output_path=tmp_path / "client-clearance.json",
        recover_failures=True,
        sleep=lambda _: None,
        log=lambda *_: None,
    )

    assert session["runs"][0]["recovery"]["client_recovery_cleared"] is True
    assert session["runs"][0]["recovery"]["client_motion_disarmed"] is True
    assert len(session["runs"]) == 2


def test_ambiguous_manual_recovery_request_is_reconciled_without_retry(
    tmp_path: Path,
):
    pending = {
        "recovery_id": "accepted-despite-409",
        "outcome": None,
        "reason": None,
    }
    complete = {
        **pending,
        "outcome": "COMPLETED",
        "reason": "HOME_POSITION_RECOVERED",
        "final_safety_state": "DISARMED_CONFIRMED",
        "final_evidence": {"home_distance_m": 0.07},
    }
    failed = terminal("run-1", outcome="FAILED", home=0.4)
    recovering = json.loads(json.dumps(failed))
    recovering["run"]["recovery_attempts"] = [pending]
    recovered = json.loads(json.dumps(failed))
    recovered["run"]["recovery_attempts"] = [complete]

    class AmbiguousRecoveryClient(FakeClient):
        recovery_result_calls = 0

        def result(self, run_id):
            if run_id == "run-1" and self.recover_home_calls:
                self.recovery_result_calls += 1
                return recovering if self.recovery_result_calls == 1 else recovered
            return super().result(run_id)

        def recover_home(self, run_id):
            self.recover_home_calls.append(run_id)
            raise RuntimeError("HTTP Error 409: recovery is already active")

    client = AmbiguousRecoveryClient(
        [READY],
        results_by_id={
            "run-1": [failed],
            "run-2": [terminal("run-2")],
        },
        sidecar=SIDECAR,
    )

    session = run_session(
        client,
        runs=2,
        seed=5,
        output_path=tmp_path / "ambiguous-409.json",
        recover_failures=True,
        sleep=lambda _: None,
        log=lambda *_: None,
    )

    assert client.recover_home_calls == ["run-1"]
    assert session["runs"][0]["recovery"]["outcome"] == "COMPLETED"
    assert session["runs"][0]["recovery"]["source"] == (
        "reconciled_after_ambiguous_manual_request"
    )
    assert len(session["runs"]) == 2


def test_unresolved_manual_recovery_request_blocks_without_retry(tmp_path: Path):
    failed = terminal("run-1", outcome="FAILED", home=0.4)

    class UnresolvedRecoveryClient(FakeClient):
        def recover_home(self, run_id):
            self.recover_home_calls.append(run_id)
            raise TimeoutError("request outcome unknown")

    client = UnresolvedRecoveryClient(
        [READY],
        results_by_id={"run-1": [failed]},
        sidecar=SIDECAR,
    )

    session = run_session(
        client,
        runs=2,
        seed=5,
        output_path=tmp_path / "unresolved-recovery.json",
        recover_failures=True,
        sleep=lambda _: None,
        log=lambda *_: None,
    )

    assert client.recover_home_calls == ["run-1"]
    assert len(client.activated) == 1
    assert session["runs"][0]["recovery"]["reason"] == (
        "RECOVERY_REQUEST_AMBIGUOUS"
    )
    assert session["runs"][0]["recovery"]["safe_to_continue"] is False


def test_session_aborts_and_persists_partial_on_latch(tmp_path: Path):
    client = FakeClient(
        [READY, READY, LATCHED],
        results_by_id={"run-1": [terminal("run-1")]},
        sidecar=SIDECAR,
    )
    output = tmp_path / "soak.json"
    session = run_session(
        client, runs=3, seed=7, output_path=output, sleep=lambda _: None, log=lambda *_: None
    )
    saved = json.loads(output.read_text())
    assert len(saved["runs"]) == 1
    assert "restart-required" in saved["aborted"]
    assert session["aborted"] == saved["aborted"]


def test_session_rejects_the_wrong_build_before_activation(tmp_path: Path):
    client = FakeClient([READY], sidecar=SIDECAR)
    with pytest.raises(HarnessAbort, match="no run was activated"):
        run_session(
            client,
            runs=1,
            seed=7,
            output_path=tmp_path / "soak.json",
            expected_build_label="edge (demo/edge)",
            sleep=lambda _: None,
            log=lambda *_: None,
        )
    assert client.activated == []


def test_session_rejects_the_wrong_qualified_fruits_before_activation(tmp_path: Path):
    client = FakeClient([READY], qualified=("pear",), sidecar=SIDECAR)
    with pytest.raises(HarnessAbort, match="expected qualified fruits"):
        run_session(
            client,
            runs=10,
            seed=7,
            output_path=tmp_path / "soak.json",
            expected_build_label="base-soak-v2-orientation (demo/base)",
            expected_fruits=["apple", "banana", "pear"],
            sleep=lambda _: None,
            log=lambda *_: None,
        )
    assert client.activated == []


def test_ambiguous_activation_retries_once_with_the_same_key(tmp_path: Path):
    class AmbiguousClient(FakeClient):
        calls = 0

        def activate(
            self,
            fruit,
            orientation_degrees=0.0,
            idempotency_key=None,
            search_policy=None,
        ):
            self.activated.append(fruit)
            self.orientation_degrees.append(orientation_degrees)
            self.idempotency_keys.append(idempotency_key)
            self.search_policies.append(search_policy)
            self.calls += 1
            if self.calls == 1:
                raise TimeoutError("request timed out after server acceptance")
            return {"run": {"run_id": "run-1"}, "activation_reused": True}

    client = AmbiguousClient(
        [READY],
        results_by_id={"run-1": [terminal("run-1")]},
        sidecar=SIDECAR,
    )
    output = tmp_path / "soak.json"
    session = run_session(
        client,
        runs=1,
        seed=7,
        output_path=output,
        expected_build_label="base-soak-v2-orientation (demo/base)",
        expected_fruits=["apple", "banana", "pear"],
        sleep=lambda _: None,
        log=lambda *_: None,
    )
    saved = json.loads(output.read_text())
    assert len(client.activated) == 2
    assert len(client.orientation_degrees) == 2
    assert client.idempotency_keys[0] == client.idempotency_keys[1]
    assert len(saved["runs"]) == 1
    assert session["aborted"] is None


def test_complete_ten_run_soak_is_balanced_and_scores_cleanly(tmp_path: Path):
    results = {
        f"run-{number}": [terminal(f"run-{number}")]
        for number in range(1, 11)
    }
    client = FakeClient([READY], results_by_id=results, sidecar=SIDECAR)
    output = tmp_path / "ten-run-soak.json"
    session = run_session(
        client,
        runs=10,
        seed=20260810,
        output_path=output,
        expected_build_label="base-soak-v2-orientation (demo/base)",
        sleep=lambda _: None,
        log=lambda *_: None,
    )

    counts = Counter(client.activated)
    assert len(session["runs"]) == 10
    assert max(counts.values()) - min(counts.values()) <= 1
    assert set(counts) == {"apple", "banana", "pear"}
    assert session["scorecard"]["criteria"]["completion"]["passed"] is True
    assert session["scorecard"]["criteria"]["fruit_coverage"]["passed"] is True
    assert session["scorecard"]["criteria"]["home_gate"]["passed"] is True
    assert output.exists()
    assert not output.with_suffix(".json.tmp").exists()


def test_summarize_run_flattens_terminal_measurements():
    record = summarize_run(terminal("run-9")["run"], "pear", 9)
    assert record["number"] == 9
    assert record["home_distance_m"] == 0.05
    assert record["run_id"] == "run-9"
    assert record["stage_results"]["approach_fruit"]["forward_pulse_count"] == 12
    assert record["failure_details"] is None  # present even when the run succeeded
