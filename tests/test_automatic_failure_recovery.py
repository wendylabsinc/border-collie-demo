import asyncio
from copy import deepcopy

import pytest

from border_collie_demo.recovery import FailedRunHomeRecovery
from border_collie_demo.run_results import RunResultStore


class RecoveryProbe:
    def __init__(
        self,
        *,
        pose_healthy: bool = True,
        pose_age_s: float = 0.01,
        trusted_home: bool = True,
        fault: str | None = None,
        active_operation: str | None = None,
        autonomy_enabled: bool = True,
        armed: bool = False,
        fusion_samples: int = 4,
        fusion_trusted: bool = True,
        stop_errors: list[str] | None = None,
        stop_exception: Exception | None = None,
        fail_return: bool = False,
        down_posture: str = "stand_down",
    ) -> None:
        self.pose_healthy = pose_healthy
        self.pose_age_s = pose_age_s
        self.trusted_home = trusted_home
        self.fault = fault
        self.active_operation = active_operation
        self.autonomy_enabled = autonomy_enabled
        self.armed = armed
        self.fusion_samples = fusion_samples
        self.fusion_trusted = fusion_trusted
        self.stop_errors = list(stop_errors or [])
        self.stop_exception = stop_exception
        self.fail_return = fail_return
        self.down_posture = down_posture
        self.x_m = 0.8
        self.calls: list[str] = []
        self.trace_phase: str | None = None

    def status(self) -> dict[str, object]:
        return {
            "configured": True,
            "autonomy_enabled": self.autonomy_enabled,
            "connected": True,
            "fault": self.fault,
            "active_operation": self.active_operation,
            "pose": {
                "healthy": self.pose_healthy,
                "age_s": self.pose_age_s,
                "pose": {"x_m": self.x_m, "y_m": 0.0, "yaw_rad": 0.0},
            },
            "motion": {
                "initialized": True,
                "armed": self.armed,
                "fault": None,
            },
            "continuous_home_fusion": {
                "running": True,
                "consecutive_trusted_samples": self.fusion_samples,
                "latest_age_s": 0.01,
                "latest": {"trusted": self.fusion_trusted},
            },
        }

    def estimate_home(self, home: dict[str, object]) -> dict[str, object]:
        del home
        return {
            "state": "trusted" if self.trusted_home else "unavailable",
            "trusted": self.trusted_home,
            "source": "planar_sensor_fusion" if self.trusted_home else None,
            "unavailable_reason": None if self.trusted_home else "fusion untrusted",
            "home_distance_m": self.x_m if self.trusted_home else None,
            "heading_error_rad": 0.0 if self.trusted_home else None,
            "pose_from_home": (
                {"x_m": self.x_m, "y_m": 0.0, "yaw_rad": 0.0}
                if self.trusted_home
                else None
            ),
            "evidence": {},
        }

    async def emergency_stop(self) -> list[str]:
        self.calls.append("emergency_stop")
        if self.stop_exception is not None:
            raise self.stop_exception
        return list(self.stop_errors)

    async def stand_down(self) -> dict[str, object]:
        self.calls.append("stand_down")
        return {"posture": self.down_posture, "motion_commands_sent": True}

    async def stand_up(self, *, settle_s: float = 1.0) -> dict[str, object]:
        self.calls.append("stand_up")
        return {
            "posture": "balance_stand",
            "motion_commands_sent": True,
            "settle_s": settle_s,
        }

    def set_motion_authority(self, run_id: str, epoch: str, phase: str) -> None:
        del run_id, epoch
        self.calls.append(f"authority:{phase}")

    def start_motion_trace(self, phase: str) -> None:
        self.trace_phase = phase

    def motion_trace(self) -> list[dict[str, object]]:
        return [{"phase": self.trace_phase}]

    async def turn_toward_home(
        self, home: dict[str, object], **options: float
    ) -> dict[str, object]:
        del home, options
        self.calls.append("turn_toward_home")
        return {"home_distance_m": self.x_m, "motion_commands_sent": True}

    async def return_home(
        self, home: dict[str, object], **options: object
    ) -> dict[str, object]:
        del home
        self.calls.append("return_home")
        if self.fail_return:
            raise RuntimeError("return controller failed")
        self.x_m = 0.08
        return {
            "home_distance_m": self.x_m,
            "requested_forward_pulses": options["forward_pulse_count"],
            "motion_commands_sent": True,
        }


def failed_run(
    results: RunResultStore,
    *,
    reason: str = "ARRIVAL_FAILURE",
    pulses: int = 3,
) -> str:
    run = results.start_run(target_fruit="pear", activation_source="test")
    run_id = run["run_id"]
    results.record_home(
        run_id,
        {
            "x_m": 0.0,
            "y_m": 0.0,
            "yaw_rad": 0.0,
            "captured_monotonic_s": 10.0,
            "age_s": 0.01,
            "source": "rt/sportmodestate",
        },
    )
    commands = [
        {
            "sequence": sequence,
            "phase": "approach_fruit",
            "forward_mps": 0.55,
        }
        for sequence in range(1, pulses + 1)
    ]
    results.seal(
        run_id,
        phase="failed",
        outcome="FAILED",
        reason=reason,
        message="approach failed",
        final_safety_state="DISARMED_CONFIRMED",
        failed_phase="approach_fruit",
        failure_details={"motion_commands": commands},
    )
    return run_id


@pytest.mark.parametrize(
    ("probe_kwargs", "pulses", "guard", "message"),
    [
        ({"trusted_home": False}, 3, None, "trusted pose"),
        ({"pose_healthy": False}, 3, None, "pose is unavailable"),
        ({"pose_age_s": 0.8}, 3, None, "pose is stale"),
        ({}, 0, None, "no forward pulse"),
        ({}, 201, None, "exceeds its recovery bound"),
        ({"fault": "motor overtemperature"}, 3, None, "hardware fault"),
        ({"active_operation": "approach_target"}, 3, None, "operation active"),
        ({"autonomy_enabled": False}, 3, None, "autonomous motion is disabled"),
        ({"armed": True}, 3, None, "not disarmed"),
        ({}, 3, "physical remote takeover is latched", "takeover"),
    ],
)
def test_unsafe_arrival_failure_records_skipped_recovery_without_translation(
    tmp_path,
    monkeypatch,
    probe_kwargs,
    pulses,
    guard,
    message,
) -> None:
    monkeypatch.setattr("border_collie_demo.recovery.FAILURE_DOWN_HOLD_S", 0.0)
    results = RunResultStore(tmp_path)
    run_id = failed_run(results, pulses=pulses)
    probe = RecoveryProbe(**probe_kwargs)
    recovery = FailedRunHomeRecovery(
        probe,
        results,
        automatic_recovery_guard=(lambda: guard),
    )

    attempt = asyncio.run(recovery.recover_automatically(run_id))

    assert attempt is not None
    assert attempt["outcome"] == "FAILED"
    assert attempt["reason"] == "AUTOMATIC_RECOVERY_SKIPPED"
    assert message in attempt["message"]
    assert "stand_down" not in probe.calls
    assert "turn_toward_home" not in probe.calls
    assert "return_home" not in probe.calls
    run = results.get(run_id)
    assert run["outcome"] == "FAILED"
    assert run["reason"] == "ARRIVAL_FAILURE"


def test_camera_failure_never_enters_automatic_recovery(tmp_path) -> None:
    results = RunResultStore(tmp_path)
    run_id = failed_run(results, reason="CAMERA_FAILURE")
    probe = RecoveryProbe()
    recovery = FailedRunHomeRecovery(probe, results)

    assert asyncio.run(recovery.recover_automatically(run_id)) is None
    assert probe.calls == []
    run = results.get(run_id)
    assert run["reason"] == "CAMERA_FAILURE"
    assert run.get("recovery_attempts") is None


def test_initial_stop_error_prevents_posture_and_translation(tmp_path) -> None:
    results = RunResultStore(tmp_path)
    run_id = failed_run(results)
    probe = RecoveryProbe(stop_errors=["sport stop timed out"])
    recovery = FailedRunHomeRecovery(probe, results)

    attempt = asyncio.run(recovery.recover_automatically(run_id))

    assert attempt is not None
    assert attempt["outcome"] == "FAILED"
    assert attempt["reason"] == "RECOVERY_FAILURE"
    assert attempt["final_safety_state"] == "STOP_REQUESTED_UNCONFIRMED"
    assert "stand_down" not in probe.calls
    assert "turn_toward_home" not in probe.calls
    assert "return_home" not in probe.calls


def test_raised_stop_error_is_durable_and_prevents_translation(tmp_path) -> None:
    results = RunResultStore(tmp_path)
    run_id = failed_run(results)
    probe = RecoveryProbe(stop_exception=RuntimeError("DDS stop unavailable"))
    recovery = FailedRunHomeRecovery(probe, results)

    attempt = asyncio.run(recovery.recover_automatically(run_id))

    assert attempt is not None
    assert attempt["reason"] == "RECOVERY_FAILURE"
    assert attempt["final_safety_state"] == "STOP_REQUESTED_UNCONFIRMED"
    assert "emergency stop raised RuntimeError" in attempt["message"]
    assert "stand_down" not in probe.calls
    assert "turn_toward_home" not in probe.calls
    assert "return_home" not in probe.calls


@pytest.mark.parametrize(
    ("probe_kwargs", "message"),
    [
        ({"fusion_samples": 2}, "fewer than three trusted samples"),
        ({"fusion_trusted": False}, "fusion sample is untrusted"),
    ],
)
def test_untrusted_posture_exit_fusion_prevents_home_turn(
    tmp_path,
    monkeypatch,
    probe_kwargs,
    message,
) -> None:
    monkeypatch.setattr("border_collie_demo.recovery.FAILURE_DOWN_HOLD_S", 0.0)
    results = RunResultStore(tmp_path)
    run_id = failed_run(results)
    probe = RecoveryProbe(**probe_kwargs)
    recovery = FailedRunHomeRecovery(probe, results)

    attempt = asyncio.run(recovery.recover_automatically(run_id))

    assert attempt is not None
    assert attempt["reason"] == "AUTOMATIC_RECOVERY_SKIPPED"
    assert message in attempt["message"]
    assert "stand_down" in probe.calls
    assert "stand_up" in probe.calls
    assert "turn_toward_home" not in probe.calls
    assert "return_home" not in probe.calls


def test_unconfirmed_failure_posture_prevents_home_motion(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("border_collie_demo.recovery.FAILURE_DOWN_HOLD_S", 0.0)
    results = RunResultStore(tmp_path)
    run_id = failed_run(results)
    probe = RecoveryProbe(down_posture="unknown")
    recovery = FailedRunHomeRecovery(probe, results)

    attempt = asyncio.run(recovery.recover_automatically(run_id))

    assert attempt is not None
    assert attempt["reason"] == "RECOVERY_FAILURE"
    assert "did not confirm stand_down" in attempt["message"]
    assert "stand_up" not in probe.calls
    assert "turn_toward_home" not in probe.calls
    assert "return_home" not in probe.calls


def test_recovery_failure_preserves_failed_run_disarms_and_is_not_retried(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr("border_collie_demo.recovery.FAILURE_DOWN_HOLD_S", 0.0)
    results = RunResultStore(tmp_path)
    run_id = failed_run(results)
    original = deepcopy(results.get(run_id))
    probe = RecoveryProbe(fail_return=True)
    recovery = FailedRunHomeRecovery(probe, results)

    first = asyncio.run(recovery.recover_automatically(run_id))
    second = asyncio.run(recovery.recover_automatically(run_id))

    assert first is not None and second is not None
    assert first["recovery_id"] == second["recovery_id"]
    assert first["outcome"] == "FAILED"
    assert first["reason"] == "RECOVERY_FAILURE"
    assert first["final_safety_state"] == "DISARMED_CONFIRMED"
    assert probe.calls.count("return_home") == 1
    assert probe.calls[-1] == "emergency_stop"
    run = results.get(run_id)
    assert len(run["recovery_attempts"]) == 1
    for field in ("outcome", "reason", "message", "final_safety_state", "failed_phase"):
        assert run[field] == original[field]
