from __future__ import annotations

import asyncio
import math

import pytest

from border_collie_demo.fruits import fruit_policy
from border_collie_demo.guidance import GuidancePhase
from border_collie_demo.hardware import CameraFailure, HardwareUnavailable, TargetLost
from border_collie_demo.models import MissionPhase
from border_collie_demo.orchestrator import StageContext, StageFailure
from border_collie_demo.production import ProductionStageExecutor
from border_collie_demo.run_tuning import RunTuning


class FakeProductionHardware:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    async def turn_relative(
        self, angle_rad: float, **options: float
    ) -> dict[str, object]:
        self.calls.append(("turn_relative", angle_rad, options))
        return {"measured_yaw_change_rad": angle_rad, "motion_commands_sent": True}

    async def emergency_stop(self) -> list[str]:
        self.calls.append(("emergency_stop",))
        return []

    async def find_target(
        self, status_reader, target_fruit: str, **options: float
    ) -> dict[str, object]:
        self.calls.append(("find_target", status_reader, target_fruit, options))
        return {
            "label": target_fruit,
            "stable_detections": 5,
            "motion_commands_sent": True,
        }

    async def approach_target(
        self, status_reader, target_fruit: str, **options: float
    ) -> dict[str, object]:
        self.calls.append(("approach_target", status_reader, target_fruit, options))
        return {"arrival_confirmed": True, "motion_commands_sent": True}

    async def stand_down(self) -> dict[str, object]:
        self.calls.append(("stand_down",))
        return {"posture": "stand_down", "motion_commands_sent": True}

    async def stand_up(self, **options: float) -> dict[str, object]:
        self.calls.append(("stand_up", options))
        return {
            "posture": "balance_stand",
            "motion_commands_sent": True,
            **options,
        }

    async def turn_toward_home(
        self, home: dict[str, object], **options: float
    ) -> dict[str, object]:
        self.calls.append(("turn_toward_home", home, options))
        return {
            "home_bearing_error_rad": 0.02,
            "motion_path": "sport_yaw",
            "pose_age_s": 0.02,
            "motion_commands_sent": True,
        }

    async def return_home_position(
        self, home: dict[str, object], **options: float
    ) -> dict[str, object]:
        self.calls.append(("return_home_position", home, options))
        return {"home_distance_m": 0.08, "motion_commands_sent": True}

    def measure_home_position(self, home: dict[str, object]) -> dict[str, object]:
        self.calls.append(("measure_home_position", home, {}))
        return {
            "home_distance_m": 0.08,
            "pose_age_s": 0.02,
            "pose_captured_monotonic_s": 42.0,
            "pose_source": "rt/sportmodestate",
        }


class FakeBark:
    async def bark(self) -> dict[str, object]:
        return {"bark_played": True}


class FailedBark:
    async def bark(self) -> dict[str, object]:
        raise RuntimeError("bark sidecar timed out")


def context(
    *,
    outbound_forward_pulses: int = 0,
    search_experiment: dict[str, object] | None = None,
    final_push: dict[str, float] | None = None,
) -> StageContext:
    tuning_payload = None
    if final_push is not None:
        tuning_payload = {
            "arrival": {
                "final_push_mps": final_push["speed_mps"],
                "final_push_duration_s": final_push["duration_s"],
            }
        }
    return StageContext(
        run_id="run-1",
        target_fruit="pear",
        home={"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0},
        outbound_forward_pulses=outbound_forward_pulses,
        run_tuning=RunTuning.from_payload("pear", tuning_payload).to_dict(),
        search_experiment=search_experiment,
    )


def test_turn_to_fruit_rotates_until_the_pear_is_recognized() -> None:
    async def scenario() -> None:
        hardware = FakeProductionHardware()
        status_reader = lambda: {"camera_healthy": True, "target_ready": False}
        stages = ProductionStageExecutor(hardware, status_reader, FakeBark())

        evidence = await stages.execute(MissionPhase.TURN_TO_FRUIT, context())

        name, reader, fruit, options = hardware.calls[0]
        assert (name, reader, fruit) == (
            "find_target",
            status_reader,
            "pear",
        )
        assert options == {
            "yaw_rps": 0.50,
            "sweep_rad": pytest.approx(2.0 * 3.141592653589793),
            "timeout_s": 30.0,
        }
        assert evidence["stable_detections"] == 5

    asyncio.run(scenario())


def test_turn_to_fruit_uses_the_selected_red_apple_target() -> None:
    async def scenario() -> None:
        hardware = FakeProductionHardware()
        status_reader = lambda: {"camera_healthy": True, "target_ready": False}
        stages = ProductionStageExecutor(hardware, status_reader, FakeBark())
        apple = StageContext(
            run_id="run-apple",
            target_fruit="apple",
            home={"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0},
        )

        evidence = await stages.execute(MissionPhase.TURN_TO_FRUIT, apple)

        assert hardware.calls[0][2] == "apple"
        assert evidence["label"] == "apple"

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "phase",
    [MissionPhase.TURN_TO_FRUIT, MissionPhase.FIND_FRUIT],
)
def test_search_stage_skips_motion_when_target_is_already_visible(
    phase: MissionPhase,
) -> None:
    async def scenario() -> None:
        hardware = FakeProductionHardware()
        status_reader = lambda: {
            "camera_healthy": True,
            "target_ready": True,
            "target_fruit": "pear",
            "detection": {
                "label": "pear",
                "confidence": 0.82,
                "consecutive_detections": 7,
                "age_s": 0.04,
                "center_x_ratio": 0.54,
                "center_y_ratio": 0.61,
                "bottom_ratio": 0.70,
            },
        }
        stages = ProductionStageExecutor(hardware, status_reader, FakeBark())

        evidence = await stages.execute(phase, context())

        assert hardware.calls == []
        assert evidence == {
            "label": "pear",
            "confidence": 0.82,
            "stable_detections": 7,
            "detection_age_s": 0.04,
            "center_x_ratio": 0.54,
            "center_y_ratio": 0.61,
            "bottom_ratio": 0.70,
            "search_progress_rad": 0.0,
            "search_skipped": True,
            "skip_reason": "target_already_visible",
            "motion_commands_sent": False,
        }

    asyncio.run(scenario())


def test_find_fruit_runs_bounded_camera_guided_search() -> None:
    async def scenario() -> None:
        hardware = FakeProductionHardware()
        status_reader = lambda: {"ready": True}
        stages = ProductionStageExecutor(hardware, status_reader, FakeBark())

        evidence = await stages.execute(MissionPhase.FIND_FRUIT, context())

        name, reader, fruit, options = hardware.calls[0]
        assert (name, reader, fruit) == ("find_target", status_reader, "pear")
        assert options == {
            "yaw_rps": 0.20,
            "sweep_rad": pytest.approx(1.308997),
            "timeout_s": 9.0,
        }
        assert evidence["stable_detections"] == 5

    asyncio.run(scenario())


def test_stage_result_records_the_exact_velocity_commands() -> None:
    class TracedHardware(FakeProductionHardware):
        def __init__(self) -> None:
            super().__init__()
            self.phase: str | None = None

        def start_motion_trace(self, phase: str, *, run_id: str | None = None) -> None:
            self.phase = phase
            self.run_id = run_id

        def motion_trace(self) -> list[dict[str, object]]:
            return [
                {
                    "sequence": 1,
                    "phase": self.phase,
                    "forward_mps": 0.0,
                    "yaw_rps": 0.5,
                    "reason": "find_target",
                }
            ]

    async def scenario() -> None:
        hardware = TracedHardware()
        stages = ProductionStageExecutor(hardware, dict, FakeBark())

        evidence = await stages.execute(MissionPhase.TURN_TO_FRUIT, context())

        assert evidence["motion_commands"] == [
            {
                "sequence": 1,
                "phase": "turn_to_fruit",
                "forward_mps": 0.0,
                "yaw_rps": 0.5,
                "reason": "find_target",
            }
        ]

    asyncio.run(scenario())


def test_approach_uses_measured_factory_motion_and_one_final_push() -> None:
    async def scenario() -> None:
        hardware = FakeProductionHardware()
        status_reader = lambda: {"ready": True}
        stages = ProductionStageExecutor(hardware, status_reader, FakeBark())

        evidence = await stages.execute(
            MissionPhase.APPROACH_FRUIT,
            context(final_push={"speed_mps": 0.60, "duration_s": 0.40}),
        )

        name, reader, fruit, options = hardware.calls[0]
        assert (name, reader, fruit) == ("approach_target", status_reader, "pear")
        assert options == {
            # The factory-avoidance calibration established 0.50 m/s as the
            # deadband edge, not a production value with usable margin. The
            # camera-guided approach uses the separately verified 1.0 m/s
            # signal, while the final off-screen movement uses this Demo Run's
            # bounded activation tuning before stop-and-lie-down.
            "forward_mps": 1.0,
            "maximum_yaw_rps": 0.30,
            "near_bottom_ratio": 0.86,
            "near_center_ratio": 0.72,
            "near_confirmations": 3,
            "near_loss_grace_s": 0.75,
            "final_push_mps": 0.6,
            "final_push_duration_s": 0.4,
            "timeout_s": 20.0,
        }
        assert evidence["arrival_confirmed"] is True

    asyncio.run(scenario())


def test_mission_lifetime_guidance_uses_the_exact_one_run_final_push_tuning() -> None:
    class GuidedHardware(FakeProductionHardware):
        async def guide_target(
            self, _status_reader, guidance, *, allow_forward: bool, timeout_s: float
        ) -> dict[str, object]:
            assert allow_forward is True
            assert timeout_s == 20.0
            return {
                "arrival_confirmed": True,
                "forward_pulse_count": 1,
                "final_push_mps": guidance.config.final_push_mps,
                "final_push_duration_s": guidance.config.final_push_duration_s,
            }

    async def scenario() -> None:
        stages = ProductionStageExecutor(GuidedHardware(), dict, FakeBark())

        evidence = await stages.execute(
            MissionPhase.APPROACH_FRUIT,
            context(final_push={"speed_mps": 0.60, "duration_s": 0.40}),
        )

        assert evidence["final_push_mps"] == 0.60
        assert evidence["final_push_duration_s"] == 0.40

    asyncio.run(scenario())


def test_production_reuses_one_guidance_identity_across_all_fruit_stages() -> None:
    class GuidedHardware(FakeProductionHardware):
        def __init__(self) -> None:
            super().__init__()
            self.guidance_ids: list[int] = []

        async def guide_target(
            self,
            status_reader,
            guidance,
            *,
            allow_forward: bool,
            timeout_s: float,
        ) -> dict[str, object]:
            self.guidance_ids.append(id(guidance))
            self.calls.append(("guide_target", allow_forward, timeout_s))
            if not allow_forward:
                guidance.acquisition_epoch = 1
                guidance.phase = GuidancePhase.LOCKED
                return {
                    "label": guidance.target_fruit,
                    "acquisition_epoch": 1,
                    "motion_commands_sent": True,
                }
            guidance.phase = GuidancePhase.ARRIVED
            return {
                "arrival_confirmed": True,
                "forward_pulse_count": 8,
                "acquisition_epoch": 1,
                "motion_commands_sent": True,
            }

    async def scenario() -> None:
        hardware = GuidedHardware()
        stages = ProductionStageExecutor(hardware, dict, FakeBark())

        locked = await stages.execute(MissionPhase.TURN_TO_FRUIT, context())
        find = await stages.execute(MissionPhase.FIND_FRUIT, context())
        arrived = await stages.execute(MissionPhase.APPROACH_FRUIT, context())

        assert locked["acquisition_epoch"] == 1
        assert find["skip_reason"] == "mission_lifetime_identity_already_locked"
        assert arrived["arrival_confirmed"] is True
        assert len(set(hardware.guidance_ids)) == 1
        assert hardware.calls == [
            ("guide_target", False, 30.0),
            ("guide_target", True, 20.0),
        ]

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("fruit", "focus", "lock"),
    [
        ("apple", 0.52, 0.42),
        ("pear", 0.70, 0.66),
        ("banana", 0.30, 0.25),
    ],
)
def test_production_applies_selected_fruit_search_confidence_only_to_the_run(
    fruit: str,
    focus: float,
    lock: float,
) -> None:
    baseline = fruit_policy(fruit)

    class GuidedHardware(FakeProductionHardware):
        async def guide_target(
            self,
            _status_reader,
            guidance,
            *,
            allow_forward: bool,
            timeout_s: float,
        ) -> dict[str, object]:
            assert allow_forward is False
            assert timeout_s == 30.0
            assert guidance.config.search_yaw_rps == 0.45
            assert guidance.config.center_confirmations == 4
            assert guidance.config.center_tolerance_ratio == 0.10
            assert guidance.target_fruit == fruit
            assert guidance.policy.focus_confidence == focus
            assert guidance.policy.acquisition_confidence == lock
            guidance.acquisition_epoch = 1
            guidance.phase = GuidancePhase.LOCKED
            return {"label": fruit, "acquisition_epoch": 1}

    async def scenario() -> None:
        stages = ProductionStageExecutor(GuidedHardware(), dict, FakeBark())
        evidence = await stages.execute(
            MissionPhase.TURN_TO_FRUIT,
            StageContext(
                run_id=f"{fruit}-experiment",
                target_fruit=fruit,
                home={"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0},
                search_experiment={
                    "target_fruit": fruit,
                    "search_yaw_rps": 0.45,
                    "focus_confidence": focus,
                    "lock_confidence": lock,
                    "center_confirmations": 4,
                    "center_tolerance_ratio": 0.10,
                },
            ),
        )

        assert evidence["acquisition_epoch"] == 1

    asyncio.run(scenario())
    assert fruit_policy(fruit) == baseline


@pytest.mark.parametrize(
    ("fruit", "focus", "lock"),
    [
        ("apple", 0.50, 0.40),
        ("pear", None, 0.65),
        ("banana", None, 0.20),
    ],
)
def test_production_preserves_baseline_fruit_policy_without_an_override(
    fruit: str,
    focus: float | None,
    lock: float,
) -> None:
    class GuidedHardware(FakeProductionHardware):
        async def guide_target(
            self,
            _status_reader,
            guidance,
            *,
            allow_forward: bool,
            timeout_s: float,
        ) -> dict[str, object]:
            assert allow_forward is False
            assert timeout_s == 30.0
            assert guidance.policy.focus_confidence == focus
            assert guidance.policy.acquisition_confidence == lock
            guidance.acquisition_epoch = 1
            guidance.phase = GuidancePhase.LOCKED
            return {"label": fruit, "acquisition_epoch": 1}

    async def scenario() -> None:
        stages = ProductionStageExecutor(GuidedHardware(), dict, FakeBark())
        await stages.execute(
            MissionPhase.TURN_TO_FRUIT,
            StageContext(
                run_id=f"{fruit}-baseline",
                target_fruit=fruit,
                home={"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0},
            ),
        )

    asyncio.run(scenario())


def test_audience_action_sits_barks_then_stands_in_separate_stages() -> None:
    async def scenario() -> None:
        hardware = FakeProductionHardware()
        bark = FakeBark()
        holds: list[float] = []

        async def hold(duration_s: float) -> None:
            holds.append(duration_s)

        stages = ProductionStageExecutor(hardware, dict, bark, sleep=hold)

        action = await stages.execute(MissionPhase.SIT_AND_BARK, context())
        standing = await stages.execute(MissionPhase.STAND, context())

        assert hardware.calls == [
            ("emergency_stop",),
            ("stand_down",),
            ("stand_up", {"settle_s": 1.0}),
        ]
        assert holds == [1.0, 5.0]
        assert action == {
            "posture": "stand_down",
            "motion_commands_sent": True,
            "bark_played": True,
            "arrival_stop_confirmed": True,
            "arrival_stop_settle_s": 1.0,
            "down_hold_s": 5.0,
        }
        assert standing == {
            "posture": "balance_stand",
            "motion_commands_sent": True,
            "settle_s": 1.0,
        }

    asyncio.run(scenario())


def test_bark_failure_is_recorded_but_cannot_skip_hold_stand_or_home() -> None:
    async def scenario() -> None:
        hardware = FakeProductionHardware()
        holds: list[float] = []

        async def hold(duration_s: float) -> None:
            holds.append(duration_s)

        stages = ProductionStageExecutor(hardware, dict, FailedBark(), sleep=hold)

        action = await stages.execute(MissionPhase.SIT_AND_BARK, context())
        standing = await stages.execute(MissionPhase.STAND, context())
        turned = await stages.execute(MissionPhase.TURN_TOWARD_HOME, context())
        returned = await stages.execute(MissionPhase.RETURN_HOME, context())

        assert action["posture"] == "stand_down"
        assert action["bark_played"] is False
        assert action["bark_error"] == "bark sidecar timed out"
        assert action["down_hold_s"] == 5.0
        assert holds == [1.0, 5.0]
        assert standing["posture"] == "balance_stand"
        assert turned["home_bearing_error_rad"] == 0.02
        assert returned["home_distance_m"] == 0.08
        assert [call[0] for call in hardware.calls] == [
            "emergency_stop",
            "stand_down",
            "stand_up",
            "turn_toward_home",
            "return_home_position",
        ]

    asyncio.run(scenario())


def test_return_stages_use_captured_home_and_locked_arrival_rules() -> None:
    async def scenario() -> None:
        hardware = FakeProductionHardware()
        stages = ProductionStageExecutor(hardware, dict, FakeBark())

        turn = await stages.execute(MissionPhase.TURN_TOWARD_HOME, context())
        returned = await stages.execute(
            MissionPhase.RETURN_HOME,
            context(outbound_forward_pulses=12),
        )
        restored = await stages.execute(MissionPhase.RESTORE_HEADING, context())

        assert [call[0] for call in hardware.calls] == [
            "turn_toward_home",
            "return_home_position",
            "measure_home_position",
        ]
        assert all(call[1] == context().home for call in hardware.calls)
        assert hardware.calls[0][2] == {
            "yaw_rps": 0.50,
            "tolerance_rad": pytest.approx(0.0872665),
            "response_timeout_s": 0.75,
            "response_min_progress_rad": pytest.approx(0.0349066),
            "recovery_settle_s": 1.0,
            "timeout_s": 30.0,
        }
        assert hardware.calls[1][2] == {
            "forward_mps": 1.0,
            "arrival_tolerance_m": 0.10,
            "heading_gate_rad": pytest.approx(0.349066),
            "maximum_yaw_rps": 0.50,
            "minimum_progress_m": 0.03,
            "stall_timeout_s": 2.0,
            "timeout_s": 30.0,
        }
        assert turn["home_bearing_error_rad"] == 0.02
        assert returned["home_distance_m"] == 0.08
        assert restored["heading_restoration_skipped"] is True
        assert restored["motion_commands_sent"] is False

    asyncio.run(scenario())


def test_return_cannot_start_when_home_turn_did_not_finish_inside_bearing_gate() -> None:
    class BadHomeTurnHardware(FakeProductionHardware):
        async def turn_toward_home(
            self, home: dict[str, object], **options: float
        ) -> dict[str, object]:
            self.calls.append(("turn_toward_home", home, options))
            return {
                "home_bearing_error_rad": math.radians(9.0),
                "motion_path": "sport_yaw",
                "pose_age_s": 0.02,
                "motion_commands_sent": True,
            }

    async def scenario() -> None:
        hardware = BadHomeTurnHardware()
        stages = ProductionStageExecutor(hardware, dict, FakeBark())

        with pytest.raises(StageFailure, match="fresh bearing gate"):
            await stages.execute(MissionPhase.TURN_TOWARD_HOME, context())

        assert [call[0] for call in hardware.calls] == ["turn_toward_home"]

    asyncio.run(scenario())


def test_camera_failure_is_preserved_as_the_terminal_stage_reason() -> None:
    class FailedCameraHardware(FakeProductionHardware):
        async def find_target(self, *_args, **_options):
            raise CameraFailure("source progress is stale")

    async def scenario() -> None:
        stages = ProductionStageExecutor(FailedCameraHardware(), dict, FakeBark())

        with pytest.raises(StageFailure) as failure:
            await stages.execute(MissionPhase.FIND_FRUIT, context())

        assert failure.value.reason == "CAMERA_FAILURE"
        assert failure.value.message == "source progress is stale"

    asyncio.run(scenario())


def test_hardware_motion_failure_is_structured_as_a_non_tolerable_blocker() -> None:
    class FailedMotionHardware(FakeProductionHardware):
        async def approach_target(self, *_args, **_options):
            raise HardwareUnavailable("motion watchdog did not confirm exact stop")

    async def scenario() -> None:
        stages = ProductionStageExecutor(FailedMotionHardware(), dict, FakeBark())

        with pytest.raises(StageFailure) as failure:
            await stages.execute(MissionPhase.APPROACH_FRUIT, context())

        assert failure.value.reason == "ARRIVAL_FAILURE"
        assert failure.value.details == {"safety_class": "motion"}

    asyncio.run(scenario())


def test_search_target_loss_is_classified_with_recognition_evidence() -> None:
    class DistantPearHardware(FakeProductionHardware):
        async def find_target(self, *_args, **_options):
            raise TargetLost(
                "pear was not found in the bounded search sweep",
                evidence={
                    "samples": 42,
                    "pear_candidate_samples": 27,
                    "maximum_confidence": 0.019,
                    "maximum_bbox_area_ratio": 0.001,
                },
            )

    async def scenario() -> None:
        stages = ProductionStageExecutor(DistantPearHardware(), dict, FakeBark())

        with pytest.raises(StageFailure) as failure:
            await stages.execute(MissionPhase.TURN_TO_FRUIT, context())

        assert failure.value.reason == "TARGET_RECOGNITION_FAILURE"
        assert failure.value.message == (
            "pear was not found in the bounded search sweep"
        )
        assert failure.value.details == {
            "recognition": {
                "samples": 42,
                "pear_candidate_samples": 27,
                "maximum_confidence": 0.019,
                "maximum_bbox_area_ratio": 0.001,
            }
        }

    asyncio.run(scenario())
