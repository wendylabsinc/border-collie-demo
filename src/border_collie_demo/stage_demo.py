"""One deep interface for starting, observing, and stopping a Demo Run.

HTTP, voice, MCP, and soak adapters should translate their input into one
``FruitMission`` and use this module instead of reproducing lifecycle logic.
The implementation owns durable activation, preflight, Home capture, stage
execution, and terminal exact-zero disarm finalization.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Protocol

from .evidence import EvidenceArtifact
from .fruits import QUALIFIED_FRUITS
from .mission import MissionMachine, RestartRequired
from .orchestrator import DemoOrchestrator, FailureEpilogue, StageExecutor
from .preflight import evaluate_preflight, preflight_check_ready
from .run_results import ActiveRunError, RunResultStore
from .run_tuning import RunTuning


class HardwareBoundary(Protocol):
    async def start(self) -> None: ...

    async def close(self) -> list[str]: ...

    async def emergency_stop(self) -> list[str]: ...

    def capture_home(self) -> dict[str, object]: ...

    def status(self) -> dict[str, object]: ...


class ActivationConflict(ValueError):
    """An activation id was reused for a different Fruit Mission."""


@dataclass(frozen=True)
class FruitMission:
    """Caller-owned identity and intent for exactly one Demo Run attempt."""

    target_fruit: str
    activation_source: str
    activation_id: str
    tuning: RunTuning | None = None

    def __post_init__(self) -> None:
        target = self.target_fruit.casefold().strip()
        source = self.activation_source.casefold().strip()
        activation_id = self.activation_id.strip()
        if target not in QUALIFIED_FRUITS:
            raise ValueError(f"unqualified Target Fruit: {self.target_fruit}")
        if not source:
            raise ValueError("activation_source must not be empty")
        if not activation_id:
            raise ValueError("activation_id must not be empty")
        object.__setattr__(self, "target_fruit", target)
        object.__setattr__(self, "activation_source", source)
        object.__setattr__(self, "activation_id", activation_id)
        if self.tuning is None:
            object.__setattr__(
                self,
                "tuning",
                RunTuning.defaults(target),
            )
        elif self.tuning.target_fruit != target:
            raise ValueError("run tuning Target Fruit does not match the Demo Run")


@dataclass(frozen=True)
class Activation:
    run: dict[str, Any]
    idempotent_replay: bool


class _FinalizingStageExecutor:
    """Internal adapter that makes exact stop part of orchestration success."""

    def __init__(
        self,
        delegate: StageExecutor,
        finalize: Callable[[], Awaitable[list[str]]],
    ) -> None:
        self._delegate = delegate
        self._finalize = finalize

    async def execute(self, phase: Any, context: Any) -> dict[str, Any]:
        return await self._delegate.execute(phase, context)

    async def stop(self) -> list[str]:
        errors: list[str] = []
        try:
            errors.extend(await self._delegate.stop())
        except Exception as exc:  # noqa: BLE001 - exact stop must still run
            errors.append(f"stage stop failed: {exc}")
        try:
            errors.extend(await self._finalize())
        except Exception as exc:  # noqa: BLE001 - report unconfirmed safety
            errors.append(f"exact stop failed: {exc}")
        return errors


class StageDemo:
    """Own one Demo Run lifecycle behind a caller-neutral interface.

    ``activation_id`` is the idempotency key. Reusing it with the same mission
    returns the original durable Run Result, including after process restart.
    Reusing it with different intent fails without changing robot state.
    """

    def __init__(
        self,
        mission: MissionMachine,
        results: RunResultStore,
        hardware: HardwareBoundary,
        camera_perception_status: Callable[[], dict[str, object]],
        *,
        media_status: Callable[[], dict[str, object]] | None = None,
        select_perception_target: Callable[[str], object] | None = None,
        stage_executor: StageExecutor | None = None,
        terminal_evidence: Callable[[], list[EvidenceArtifact]] | None = None,
        failure_epilogue: FailureEpilogue | None = None,
        stage_home_margin_m: float = 0.50,
    ) -> None:
        if (
            not math.isfinite(stage_home_margin_m)
            or not 0.10 <= stage_home_margin_m <= 1.0
        ):
            raise ValueError("stage_home_margin_m must be within 0.10..1.0 m")
        self._mission = mission
        self._results = results
        self._hardware = hardware
        self._camera_perception_status = camera_perception_status
        self._media_status = media_status
        self._select_perception_target = select_perception_target
        self._lifecycle_lock = asyncio.Lock()
        self._run_task: asyncio.Task[dict[str, Any]] | None = None
        self._started = False
        self._stage_home: dict[str, Any] | None = None
        self._stage_executor = stage_executor
        self._stage_home_margin_m = stage_home_margin_m
        self._orchestrator = None
        if stage_executor is not None:
            self._orchestrator = DemoOrchestrator(
                mission,
                results,
                _FinalizingStageExecutor(stage_executor, self._exact_stop),
                terminal_evidence=terminal_evidence,
                failure_epilogue=failure_epilogue,
            )

    async def start(self) -> None:
        async with self._lifecycle_lock:
            if self._started:
                return
            self._results.seal_interrupted_runs()
            await self._hardware.start()
            self._stage_home = self._latest_recorded_home()
            self._started = True

    async def close(self) -> list[str]:
        async with self._lifecycle_lock:
            task = self._run_task
            self._run_task = None
            if task is not None and not task.done():
                task.cancel()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        errors = await self._stop_execution()
        if self._results.active_run_id is not None:
            try:
                self._mission.fail("application stopped during Demo Run")
            except RestartRequired:
                pass
            self._results.seal(
                self._results.active_run_id,
                phase=self._mission.phase.value,
                outcome="FAILED",
                reason="PROCESS_INTERRUPTED",
                message="application stopped before the Demo Run completed",
                final_safety_state=(
                    "DISARMED_CONFIRMED" if not errors else "STOP_REQUESTED_UNCONFIRMED"
                ),
            )
        errors.extend(await self._hardware.close())
        self._started = False
        return errors

    async def activate(self, mission: FruitMission) -> Activation:
        async with self._lifecycle_lock:
            self._require_started()
            prior = self._find_activation(mission.activation_id)
            if prior is not None:
                self._require_same_mission(prior, mission)
                return Activation(prior, idempotent_replay=True)
            if self._mission.takeover_latched:
                raise RestartRequired(
                    "physical remote takeover is latched; restart required"
                )
            clearance = self._inter_run_clearance(self._hardware.status())
            if clearance is not None and not clearance["returned_home"]:
                raise ActiveRunError(
                    "the prior Demo Run has not returned Home; another run cannot start"
                )

            run = self._results.start_run(
                target_fruit=mission.target_fruit,
                activation_source=mission.activation_source,
                activation_id=mission.activation_id,
                run_tuning=mission.tuning.to_dict(),
                search_experiment=mission.tuning.search_experiment_dict(),
            )
            self._mission.begin_run("Demo Run activation persisted")
            run = self._results.enter_phase(
                run["run_id"],
                phase=self._mission.phase.value,
                reason="PREFLIGHT_STARTED",
                message=(
                    "preflight entered; verifying production motion and media gates"
                ),
            )
            camera = self._select_target_and_read_camera(mission.target_fruit)
            media = self._read_media()
            report = evaluate_preflight(self._hardware.status(), camera, media)
            run = self._results.record_preflight(run["run_id"], report)
            if not report["ready"]:
                errors = await self._exact_stop()
                self._mission.fail("preflight readiness failed")
                run = self._results.seal(
                    run["run_id"],
                    phase=self._mission.phase.value,
                    outcome="FAILED",
                    reason="PREFLIGHT_FAILURE",
                    message="required preflight readiness checks did not pass",
                    final_safety_state=(
                        "DISARMED_CONFIRMED"
                        if not errors
                        and preflight_check_ready(
                            evaluate_preflight(self._hardware.status(), camera, media),
                            "motion_disarmed",
                        )
                        else "STOP_REQUESTED_UNCONFIRMED"
                    ),
                    failed_phase="preflight",
                )
                return Activation(run, idempotent_replay=False)

            self._mission.advance("preflight readiness passed")
            run = self._results.enter_phase(
                run["run_id"],
                phase=self._mission.phase.value,
                reason="CAPTURE_HOME_STARTED",
                message=(
                    "preflight passed; ready to capture Home"
                    if self._stage_home is None
                    else "preflight passed; reusing the persisted Stage Home"
                ),
            )
            try:
                measured = self._hardware.capture_home()
            except Exception as exc:  # noqa: BLE001 - hardware evidence seam
                errors = await self._exact_stop()
                self._mission.fail("fresh Home pose capture failed")
                run = self._results.seal(
                    run["run_id"],
                    phase=self._mission.phase.value,
                    outcome="FAILED",
                    reason="PREFLIGHT_FAILURE",
                    message=f"fresh Home pose capture failed: {exc}",
                    final_safety_state=(
                        "DISARMED_CONFIRMED"
                        if not errors
                        else "STOP_REQUESTED_UNCONFIRMED"
                    ),
                    failed_phase="capture_home",
                )
                return Activation(run, idempotent_replay=False)

            source = "REUSED" if self._stage_home is not None else "CAPTURED"
            if self._stage_home is None:
                self._stage_home = deepcopy(measured)
            home = deepcopy(self._stage_home)
            provenance = self._home_provenance(home, measured, source=source)
            run = self._results.record_home(
                run["run_id"], home, provenance=provenance
            )
            self._mission.advance(
                "fresh Home pose captured"
                if provenance["source"] == "CAPTURED"
                else "persisted Stage Home reused"
            )
            run = self._results.enter_phase(
                run["run_id"],
                phase=self._mission.phase.value,
                reason="WAITING_FOR_COMMAND",
                message=(
                    "Home captured; waiting for the qualified "
                    f"{mission.target_fruit} command"
                ),
            )
            if self._orchestrator is not None:
                self._run_task = asyncio.create_task(
                    self._orchestrator.run(run["run_id"]),
                    name=f"fruit-mission-{run['run_id']}",
                )
            return Activation(run, idempotent_replay=False)

    async def recapture_home(self) -> dict[str, Any]:
        """Replace the persisted Stage Home on explicit operator intent only.

        Home never resets implicitly. A new Demo Run, a new cohort, or a failed
        run all reuse the persisted Stage Home; only this control moves it.
        """
        async with self._lifecycle_lock:
            self._require_started()
            if self._results.active_run_id is not None:
                raise ActiveRunError(
                    "a Demo Run is active; Home cannot be recaptured"
                )
            if self._mission.takeover_latched:
                raise RestartRequired(
                    "physical remote takeover is latched; restart required"
                )
            previous = self._stage_home
            measured = self._hardware.capture_home()
            self._stage_home = deepcopy(measured)
            return {
                "home": deepcopy(self._stage_home),
                "previous_home": deepcopy(previous),
                "offset_from_previous_m": self._home_offset_m(previous, measured),
            }

    def result(self, run_id: str) -> dict[str, Any]:
        return self._results.get(run_id)

    async def wait(
        self,
        run_id: str,
        *,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        run = self._results.get(run_id)
        if run["outcome"] is not None:
            return run
        task = self._run_task
        if task is None or self._results.active_run_id != run_id:
            raise ActiveRunError("Demo Run is active but has no execution adapter")
        await asyncio.wait_for(asyncio.shield(task), timeout=timeout_s)
        if task is self._run_task:
            self._run_task = None
        return self._results.get(run_id)

    async def stop(self) -> dict[str, Any]:
        async with self._lifecycle_lock:
            self._require_started()
            task = self._run_task
            self._run_task = None
            if task is not None and not task.done():
                task.cancel()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        errors = await self._stop_execution()
        try:
            self._mission.stop()
        except RestartRequired:
            pass
        run = None
        if self._results.active_run_id is not None:
            run = self._results.seal(
                self._results.active_run_id,
                phase=self._mission.phase.value,
                outcome="STOPPED",
                reason="OPERATOR_STOP",
                message="operator stopped the Demo Run",
                final_safety_state=(
                    "DISARMED_CONFIRMED" if not errors else "STOP_REQUESTED_UNCONFIRMED"
                ),
            )
        return {
            "mission": self._mission.status(),
            "hardware": self._hardware.status(),
            "stop_errors": errors,
            "run": run,
        }

    def status(self) -> dict[str, Any]:
        camera = self._read_camera()
        media = self._read_media()
        hardware = self._hardware.status()
        report = evaluate_preflight(hardware, camera, media)
        clearance = self._inter_run_clearance(hardware)
        blockers = [
            {"name": item["name"], "detail": item["detail"]}
            for item in report["checks"]
            if not item["ready"]
        ]
        if clearance is not None and not clearance["returned_home"]:
            distance = clearance["home_distance_m"]
            detail = (
                "fresh current Home distance is unavailable"
                if distance is None
                else (
                    f"prior run is {distance:.3f} m from Home; "
                    f"required <= {self._stage_home_margin_m:.3f} m"
                )
            )
            blockers.append({"name": "inter_run_home_clearance", "detail": detail})
        activation: dict[str, Any] = {
            "ready": report["ready"]
            and (clearance is None or clearance["returned_home"]),
            "blockers": blockers,
        }
        if clearance is not None:
            activation["inter_run"] = clearance
        return {
            "mission": self._mission.status(),
            "hardware": hardware,
            "active_run_id": self._results.active_run_id,
            "activation": activation,
            "stage_home": deepcopy(self._stage_home),
        }

    def list_results(self) -> list[dict[str, Any]]:
        return self._results.list_results()

    def _latest_recorded_home(self) -> dict[str, Any] | None:
        """Seed the Stage Home from durable evidence so it survives restart."""
        recorded = next(
            (
                run
                for run in self._results.list_results()
                if isinstance(run.get("home"), dict)
            ),
            None,
        )
        return None if recorded is None else deepcopy(recorded["home"])

    @staticmethod
    def _home_offset_m(
        home: dict[str, Any] | None, measured: dict[str, Any]
    ) -> float | None:
        if not isinstance(home, dict):
            return None
        values = (
            home.get("x_m"),
            home.get("y_m"),
            measured.get("x_m"),
            measured.get("y_m"),
        )
        if not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in values
        ):
            return None
        return math.hypot(
            float(values[2]) - float(values[0]), float(values[3]) - float(values[1])
        )

    def _home_provenance(
        self, home: dict[str, Any], measured: dict[str, Any], *, source: str
    ) -> dict[str, Any]:
        return {
            "source": source,
            "measured_pose": deepcopy(measured),
            "activation_offset_m": self._home_offset_m(home, measured),
        }

    def _inter_run_clearance(
        self, hardware: dict[str, object]
    ) -> dict[str, object] | None:
        prior = next(
            (
                run
                for run in self._results.list_results()
                if run.get("outcome") is not None and isinstance(run.get("home"), dict)
            ),
            None,
        )
        if prior is None:
            return None
        # The persisted Stage Home is authoritative; a prior run's recorded Home
        # is only a fallback for a store that predates Home persistence.
        home = self._stage_home if self._stage_home is not None else prior["home"]
        pose_status = hardware.get("pose")
        current = (
            pose_status.get("pose")
            if isinstance(pose_status, dict) and pose_status.get("healthy") is True
            else None
        )
        home_x = home.get("x_m")
        home_y = home.get("y_m")
        current_x = current.get("x_m") if isinstance(current, dict) else None
        current_y = current.get("y_m") if isinstance(current, dict) else None
        coordinates = (home_x, home_y, current_x, current_y)
        distance = (
            math.hypot(
                float(current_x) - float(home_x), float(current_y) - float(home_y)
            )
            if all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                for value in coordinates
            )
            else None
        )
        returned_home = (
            prior.get("final_safety_state") == "DISARMED_CONFIRMED"
            and distance is not None
            and distance <= self._stage_home_margin_m
        )
        return {
            "required": True,
            "prior_run_id": prior["run_id"],
            "prior_outcome": prior["outcome"],
            "home_distance_m": distance,
            "stage_home_margin_m": self._stage_home_margin_m,
            "returned_home": returned_home,
        }

    async def _exact_stop(self) -> list[str]:
        try:
            errors = list(await self._hardware.emergency_stop())
        except Exception as exc:  # noqa: BLE001 - status may still prove disarm
            errors = [f"emergency stop failed: {exc}"]
        try:
            status = self._hardware.status()
        except Exception as exc:  # noqa: BLE001 - missing proof fails closed
            errors.append(f"post-stop hardware status failed: {exc}")
            return errors
        if status.get("active_operation") is not None:
            errors.append("hardware operation remained active after stop")
        motion = status.get("motion")
        if isinstance(motion, dict):
            if motion.get("armed", False):
                errors.append("motion remained armed after stop")
            command = motion.get("last_command")
            if isinstance(command, dict):
                for key in ("forward_mps", "yaw_rps"):
                    value = command.get(key)
                    if not isinstance(value, (int, float)) or float(value) != 0.0:
                        errors.append(f"motion {key} was not exact zero after stop")
        return errors

    async def _stop_execution(self) -> list[str]:
        errors: list[str] = []
        if self._stage_executor is not None:
            try:
                errors.extend(await self._stage_executor.stop())
            except Exception as exc:  # noqa: BLE001 - exact stop must still run
                errors.append(f"stage stop failed: {exc}")
        try:
            errors.extend(await self._exact_stop())
        except Exception as exc:  # noqa: BLE001 - report unconfirmed safety
            errors.append(f"exact stop failed: {exc}")
        return errors

    def _find_activation(self, activation_id: str) -> dict[str, Any] | None:
        return next(
            (
                run
                for run in self._results.list_results()
                if run.get("activation_id") == activation_id
            ),
            None,
        )

    @staticmethod
    def _require_same_mission(run: dict[str, Any], mission: FruitMission) -> None:
        if (
            run.get("target_fruit") != mission.target_fruit
            or run.get("activation_source") != mission.activation_source
            or run.get("run_tuning") != mission.tuning.to_dict()
        ):
            raise ActivationConflict(
                "activation_id already belongs to a different Fruit Mission"
            )

    def _select_target_and_read_camera(self, target_fruit: str) -> dict[str, object]:
        try:
            if self._select_perception_target is not None:
                self._select_perception_target(target_fruit)
            return self._camera_perception_status()
        except Exception as exc:  # noqa: BLE001 - external adapter seam
            return {
                "ready": False,
                "detail": f"camera/perception readiness error: {exc}",
            }

    def _read_camera(self) -> dict[str, object]:
        try:
            return self._camera_perception_status()
        except Exception as exc:  # noqa: BLE001 - external adapter seam
            return {
                "ready": False,
                "detail": f"camera/perception readiness error: {exc}",
            }

    def _read_media(self) -> dict[str, object] | None:
        if self._media_status is None:
            return None
        try:
            return self._media_status()
        except Exception as exc:  # noqa: BLE001 - external adapter seam
            return {"ready": False, "detail": f"bark media readiness error: {exc}"}

    def _require_started(self) -> None:
        if not self._started:
            raise RuntimeError("StageDemo.start() must complete before use")
