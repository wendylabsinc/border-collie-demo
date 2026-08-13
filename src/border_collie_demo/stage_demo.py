"""One deep interface for starting, observing, and stopping a Demo Run.

HTTP, voice, MCP, and soak adapters should translate their input into one
``FruitMission`` and use this module instead of reproducing lifecycle logic.
The implementation owns durable activation, preflight, Home capture, stage
execution, and terminal exact-zero disarm finalization.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol

from .evidence import EvidenceArtifact
from .fruits import QUALIFIED_FRUITS
from .mission import MissionMachine, RestartRequired
from .orchestrator import DemoOrchestrator, FailureEpilogue, StageExecutor
from .preflight import evaluate_preflight, preflight_check_ready
from .run_results import ActiveRunError, RunResultStore
from .search_experiment import SearchExperimentTuning


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
    search_experiment: SearchExperimentTuning | None = None

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
        if self.search_experiment is None:
            object.__setattr__(
                self,
                "search_experiment",
                SearchExperimentTuning.defaults(target),
            )


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
    ) -> None:
        self._mission = mission
        self._results = results
        self._hardware = hardware
        self._camera_perception_status = camera_perception_status
        self._media_status = media_status
        self._select_perception_target = select_perception_target
        self._lifecycle_lock = asyncio.Lock()
        self._run_task: asyncio.Task[dict[str, Any]] | None = None
        self._started = False
        self._stage_executor = stage_executor
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
                    "DISARMED_CONFIRMED"
                    if not errors
                    else "STOP_REQUESTED_UNCONFIRMED"
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

            run = self._results.start_run(
                target_fruit=mission.target_fruit,
                activation_source=mission.activation_source,
                activation_id=mission.activation_id,
                search_experiment=mission.search_experiment.to_dict(),
            )
            self._mission.begin_run("Demo Run activation persisted")
            run = self._results.enter_phase(
                run["run_id"],
                phase=self._mission.phase.value,
                reason="PREFLIGHT_STARTED",
                message=(
                    "preflight entered; verifying production motion "
                    "and media gates"
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
                            evaluate_preflight(
                                self._hardware.status(), camera, media
                            ),
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
                message="preflight passed; ready to capture Home",
            )
            try:
                home = self._hardware.capture_home()
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

            run = self._results.record_home(run["run_id"], home)
            self._mission.advance("fresh Home pose captured")
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
                    "DISARMED_CONFIRMED"
                    if not errors
                    else "STOP_REQUESTED_UNCONFIRMED"
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
        report = evaluate_preflight(self._hardware.status(), camera, media)
        return {
            "mission": self._mission.status(),
            "hardware": self._hardware.status(),
            "active_run_id": self._results.active_run_id,
            "activation": {
                "ready": report["ready"],
                "blockers": [
                    {"name": item["name"], "detail": item["detail"]}
                    for item in report["checks"]
                    if not item["ready"]
                ],
            },
        }

    def list_results(self) -> list[dict[str, Any]]:
        return self._results.list_results()

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
            or run.get("search_experiment") != mission.search_experiment.to_dict()
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
