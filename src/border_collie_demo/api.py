from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .evidence import EvidenceArtifact
from .flight_recorder import FlightRecorder, terminal_evidence_bundle
from .fruits import QUALIFIED_FRUITS, SUPPORTED_FRUITS
from .hardware import HardwareManager, HardwareUnavailable
from .mission import MissionMachine, RestartRequired
from .models import MissionPhase
from .orchestrator import EXECUTED_STAGES, DemoOrchestrator, StageExecutor
from .preflight import evaluate_preflight, preflight_check_ready
from .recovery import (
    RECOVERY_CONFIRMATION,
    FailedRunHomeRecovery,
    recovery_forward_pulse_budget,
)
from .release import ReleaseCohort
from .run_coordinator import RunActivation, RunCoordinator
from .run_results import ActiveRunError, RunResultNotFound, RunResultStore
from .search_policy import SearchPolicy


def build_label() -> str:
    """Identify which demo branch is deployed.

    Display only. Three branches are deployed to the same robot one at a time,
    so an operator needs to confirm from the UI which build produced a run
    before recording its result against a branch.
    """
    return os.environ.get("BORDER_COLLIE_BUILD_LABEL", "unlabelled").strip() or "unlabelled"


class ForwardPulseRequest(BaseModel):
    confirmation: str


class RunRequest(BaseModel):
    target_fruit: Literal["apple", "banana", "pear"] = "pear"
    activation_source: Literal["audience_ui", "voice"] = "audience_ui"
    orientation_degrees: float = Field(default=0.0, ge=0.0, lt=360.0)
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=128)


class FruitPreviewRequest(BaseModel):
    target_fruit: Literal["apple", "banana", "pear"]


class FailedRunRecoveryRequest(BaseModel):
    confirmation: str


def create_app(
    mission: MissionMachine | None = None,
    hardware: HardwareManager | None = None,
    web_root: Path | None = None,
    runs_root: Path | None = None,
    camera_perception_status: Callable[[], dict[str, object]] | None = None,
    camera_frame: Callable[[], bytes] | None = None,
    select_perception_target: Callable[[str], dict[str, object]] | None = None,
    media_status: Callable[[], dict[str, object]] | None = None,
    stage_executor: StageExecutor | None = None,
    terminal_evidence: Callable[[], list[EvidenceArtifact]] | None = None,
    flight_recorder: FlightRecorder | None = None,
    release_cohort: ReleaseCohort | None = None,
    search_policy: SearchPolicy | None = None,
    runtime_mode: Literal["production", "simulation"] = "production",
) -> FastAPI:
    selected_search_policy = search_policy or SearchPolicy.configured()
    machine = mission or MissionMachine()
    robot = hardware or HardwareManager()
    root = web_root or Path(os.environ.get("BORDER_COLLIE_WEB_ROOT", "web")).resolve()
    runs_directory = runs_root or Path(
        os.environ.get("BORDER_COLLIE_RUNS_DIR", "artifacts/runs")
    )
    results = RunResultStore(runs_directory)
    recorder = flight_recorder or FlightRecorder(
        Path(runs_directory) / "_flight_recorder"
    )
    coordinator = RunCoordinator(machine, results, recorder)
    set_recorder = getattr(robot, "set_flight_recorder", None)
    if callable(set_recorder):
        set_recorder(recorder)
    read_camera_perception = camera_perception_status or (
        lambda: {
            "ready": False,
            "detail": "production camera/perception adapter is not connected",
        }
    )
    read_media = media_status

    def current_media_status() -> dict[str, object] | None:
        if read_media is None:
            return None
        try:
            return read_media()
        except Exception as exc:  # noqa: BLE001 - external adapter boundary
            return {
                "ready": False,
                "detail": f"bark media readiness error: {exc}",
            }

    active_tasks: set[asyncio.Task[dict[str, object]]] = set()
    recovery = FailedRunHomeRecovery(
        robot,
        results,
        automatic_recovery_guard=lambda: (
            "physical remote takeover is latched; automatic recovery is forbidden"
            if machine.takeover_latched
            else None
        ),
    )

    def capture_terminal_bundle() -> list[EvidenceArtifact]:
        return terminal_evidence_bundle(recorder, terminal_evidence)

    orchestrator = (
        None
        if stage_executor is None
        else DemoOrchestrator(
            coordinator,
            results,
            stage_executor,
            terminal_evidence=capture_terminal_bundle,
            automatic_failure_recovery=recovery,
        )
    )

    async def capture_failed_evidence(run_id: str) -> None:
        try:
            recorder.record(
                "terminal_evidence_requested",
                {"run": results.get(run_id)},
                run_id=run_id,
            )
            artifacts = await asyncio.to_thread(capture_terminal_bundle)
            results.record_artifacts(run_id, artifacts)
        except Exception as exc:  # noqa: BLE001 - evidence must not mask safety
            results.record_evidence_unavailable(
                run_id,
                f"terminal evidence capture failed: {exc}",
            )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        await robot.start()
        recorder.record(
            "application_started",
            {
                "runtime_mode": runtime_mode,
                "search_policy": selected_search_policy.name,
            },
        )
        if results.has_interrupted_work():
            startup_stop_errors = await robot.emergency_stop()
            motion = robot.status().get("motion")
            startup_disarmed = (
                not startup_stop_errors
                and isinstance(motion, dict)
                and motion.get("armed") is False
            )
            coordinator.recover_interrupted(
                final_safety_state=(
                    "DISARMED_CONFIRMED"
                    if startup_disarmed
                    else "STOP_REQUESTED_UNCONFIRMED"
                )
            )
        try:
            yield
        finally:
            recorder.record("application_stopping", {"active_tasks": len(active_tasks)})
            tasks = list(active_tasks)
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            await robot.close()

    app = FastAPI(
        title="Border Collie Demo",
        version=build_label(),
        lifespan=lifespan,
    )

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(root / "index.html")

    @app.get("/debug")
    async def debug() -> FileResponse:
        return FileResponse(root / "debug.html")

    @app.get("/fruit-test")
    async def fruit_test() -> FileResponse:
        return FileResponse(root / "fruit-test.html")

    @app.get("/api/camera/frame.jpg")
    async def camera_frame_proxy() -> Response:
        if camera_frame is None:
            raise HTTPException(
                status_code=503, detail="camera preview is not connected"
            )
        try:
            jpeg = await asyncio.to_thread(camera_frame)
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"camera preview unavailable: {exc}",
            ) from exc
        return Response(
            content=jpeg,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/api/fruits")
    async def fruits() -> dict[str, object]:
        return {
            "supported_fruits": list(SUPPORTED_FRUITS),
            "qualified_fruits": list(QUALIFIED_FRUITS),
        }

    @app.post("/api/fruits/preview")
    async def preview_fruit(request: FruitPreviewRequest) -> dict[str, object]:
        if results.active_run_id is not None:
            raise HTTPException(
                status_code=409,
                detail="fruit preview cannot change during an active Demo Run",
            )
        if select_perception_target is None:
            raise HTTPException(
                status_code=503,
                detail="fruit perception selector is not connected",
            )
        try:
            selected = select_perception_target(request.target_fruit)
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"fruit perception selection failed: {exc}",
            ) from exc
        return {
            "target_fruit": request.target_fruit,
            "qualified_for_demo": request.target_fruit in QUALIFIED_FRUITS,
            "supported_fruits": selected.get(
                "supported_fruits", list(SUPPORTED_FRUITS)
            ),
        }

    @app.get("/api/status")
    async def status() -> dict[str, object]:
        try:
            camera_perception = read_camera_perception()
        except Exception as exc:  # noqa: BLE001 - external adapter boundary
            camera_perception = {
                "ready": False,
                "detail": f"camera/perception readiness error: {exc}",
            }
        media = current_media_status()
        preflight = evaluate_preflight(robot.status(), camera_perception, media)
        return {
            "build_label": build_label(),
            "search_policy": selected_search_policy.name,
            "runtime_mode": runtime_mode,
            "release": None if release_cohort is None else release_cohort.to_dict(),
            "mission": machine.status(),
            "hardware": robot.status(),
            "active_run_id": results.active_run_id,
            "active_recovery": results.active_recovery,
            "flight_recorder": recorder.status(),
            "activation": {
                "ready": preflight["ready"] and results.active_recovery is None,
                "blockers": [
                    {
                        "name": check["name"],
                        "detail": check["detail"],
                    }
                    for check in preflight["checks"]
                    if not check["ready"]
                ]
                + (
                    [
                        {
                            "name": "failed_run_recovery_active",
                            "detail": "failed-run Home recovery is active",
                        }
                    ]
                    if results.active_recovery is not None
                    else []
                ),
            },
        }

    @app.post("/api/run", status_code=201)
    async def activate_run(request: RunRequest) -> dict[str, object]:
        if machine.takeover_latched:
            raise HTTPException(
                status_code=423,
                detail="physical remote takeover is latched; restart required",
            )
        if results.active_recovery is not None:
            raise HTTPException(
                status_code=409,
                detail="failed-run Home recovery is active",
            )
        try:
            decision = coordinator.activate(
                RunActivation(
                    target_fruit=request.target_fruit,
                    activation_source=request.activation_source,
                    orientation_degrees=request.orientation_degrees,
                    idempotency_key=request.idempotency_key,
                )
            )
        except ActiveRunError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        run = decision.run
        if not decision.created:
            return {"run": run, "activation_reused": True}
        try:
            if select_perception_target is not None:
                select_perception_target(request.target_fruit)
            camera_perception = read_camera_perception()
        except Exception as exc:  # noqa: BLE001 - external adapter boundary
            camera_perception = {
                "ready": False,
                "detail": f"camera/perception readiness error: {exc}",
            }
        media = current_media_status()
        report = evaluate_preflight(robot.status(), camera_perception, media)
        run = results.record_preflight(run["run_id"], report)
        if not report["ready"]:
            stop_errors = await robot.emergency_stop()
            await capture_failed_evidence(run["run_id"])
            run = coordinator.finish(
                run["run_id"],
                terminal_phase=MissionPhase.FAILED,
                outcome="FAILED",
                reason="PREFLIGHT_FAILURE",
                message="required preflight readiness checks did not pass",
                final_safety_state=(
                    "DISARMED_CONFIRMED"
                    if not stop_errors
                    and preflight_check_ready(
                        evaluate_preflight(robot.status(), camera_perception, media),
                        "motion_disarmed",
                    )
                    else "STOP_REQUESTED_UNCONFIRMED"
                ),
                failed_phase="preflight",
            )
        else:
            run = coordinator.advance(
                run["run_id"],
                reason="preflight readiness passed",
                event_reason="CAPTURE_HOME_STARTED",
                message="preflight passed; ready to capture Home",
            )
            try:
                home = robot.capture_home()
                run = results.record_home(run["run_id"], home)
            except Exception as exc:  # noqa: BLE001 - hardware evidence boundary
                stop_errors = await robot.emergency_stop()
                await capture_failed_evidence(run["run_id"])
                run = coordinator.finish(
                    run["run_id"],
                    terminal_phase=MissionPhase.FAILED,
                    outcome="FAILED",
                    reason="PREFLIGHT_FAILURE",
                    message=f"fresh Home pose capture failed: {exc}",
                    final_safety_state=(
                        "DISARMED_CONFIRMED"
                        if not stop_errors
                        else "STOP_REQUESTED_UNCONFIRMED"
                    ),
                    failed_phase="capture_home",
                )
            else:
                run = coordinator.advance(
                    run["run_id"],
                    reason="fresh Home pose captured",
                    event_reason="WAITING_FOR_COMMAND",
                    message=(
                        "Home captured; waiting for the qualified "
                        f"{request.target_fruit} command"
                    ),
                )
                if orchestrator is not None:
                    task = asyncio.create_task(orchestrator.run(run["run_id"]))
                    active_tasks.add(task)
                    task.add_done_callback(active_tasks.discard)
        return {"run": run}

    @app.post("/api/results/{run_id}/recover-home", status_code=202)
    async def recover_failed_run_home(
        run_id: str,
        request: FailedRunRecoveryRequest,
    ) -> dict[str, object]:
        if request.confirmation.strip().upper() != RECOVERY_CONFIRMATION:
            raise HTTPException(
                status_code=409,
                detail=f'type exactly "{RECOVERY_CONFIRMATION}"',
            )
        if machine.takeover_latched:
            raise HTTPException(
                status_code=423,
                detail="physical remote takeover is latched; restart required",
            )
        try:
            run = results.get(run_id)
            pulse_count, pulse_source = recovery.validate(run)
            attempt = results.start_recovery(
                run_id,
                confirmation=RECOVERY_CONFIRMATION,
            )
        except RunResultNotFound as exc:
            raise HTTPException(status_code=404, detail="Run Result not found") from exc
        except (ActiveRunError, HardwareUnavailable) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        task = asyncio.create_task(
            recovery.run(run_id, str(attempt["recovery_id"]))
        )
        active_tasks.add(task)
        task.add_done_callback(active_tasks.discard)
        return {
            "recovery": attempt,
            "outbound_forward_pulses": pulse_count,
            "maximum_forward_pulses": recovery_forward_pulse_budget(pulse_count),
            "forward_pulse_source": pulse_source,
        }

    @app.get("/api/results/{run_id}")
    async def get_result(run_id: str) -> dict[str, object]:
        try:
            return {"run": results.get(run_id)}
        except RunResultNotFound as exc:
            raise HTTPException(status_code=404, detail="Run Result not found") from exc

    @app.get("/api/results/{run_id}/artifacts/{filename}")
    async def get_result_artifact(run_id: str, filename: str) -> FileResponse:
        try:
            path, content_type = results.artifact_path(run_id, filename)
        except RunResultNotFound as exc:
            raise HTTPException(
                status_code=404, detail="Run artifact not found"
            ) from exc
        return FileResponse(
            path,
            media_type=content_type,
            filename=filename,
        )

    @app.get("/api/results")
    async def list_results() -> dict[str, object]:
        return {"runs": results.list_results()}

    @app.get("/api/diagnostics/stages")
    async def diagnostic_stages() -> dict[str, object]:
        runs = results.list_results()
        latest = runs[0] if runs else None
        completed = (latest or {}).get("stage_results", {})
        compact_completed = set(
            ((latest or {}).get("key_values") or {}).get("completed_stages", [])
        )
        failed_phase = (latest or {}).get("failed_phase")
        stages = []
        for phase in EXECUTED_STAGES:
            if phase.value in completed or phase.value in compact_completed:
                stage_status = "COMPLETED"
            elif failed_phase == phase.value:
                stage_status = "FAILED"
            else:
                stage_status = "NOT_RUN"
            stages.append(
                {
                    "phase": phase.value,
                    "status": stage_status,
                    "evidence": completed.get(phase.value),
                }
            )
        return {
            "latest_run": (
                None
                if latest is None
                else {
                    "run_id": latest["run_id"],
                    "outcome": latest["outcome"],
                    "reason": latest["reason"],
                    "failed_phase": latest["failed_phase"],
                    "failure_details": latest.get("failure_details"),
                    "artifacts": latest.get("artifacts", []),
                    "evidence_capture": latest.get("evidence_capture"),
                }
            ),
            "stages": stages,
        }

    @app.post("/api/hardware/forward-pulse")
    async def forward_pulse(
        request: ForwardPulseRequest,
    ) -> dict[str, object]:
        if machine.takeover_latched:
            raise HTTPException(
                status_code=423,
                detail="physical remote takeover is latched; restart required",
            )
        try:
            result = await robot.run_forward_pulse(request.confirmation)
        except HardwareUnavailable as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"result": result, "hardware": robot.status()}

    @app.post("/api/stop")
    async def stop() -> dict[str, object]:
        tasks = list(active_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        active_tasks.difference_update(tasks)
        stop_errors = await robot.emergency_stop()
        if stage_executor is not None:
            stop_errors.extend(await stage_executor.stop())
        run = None
        if results.active_run_id is not None:
            await capture_failed_evidence(results.active_run_id)
            run = coordinator.finish(
                results.active_run_id,
                terminal_phase=MissionPhase.STOPPED,
                outcome="STOPPED",
                reason="OPERATOR_STOP",
                message="operator stopped the Demo Run",
                final_safety_state=(
                    "DISARMED_CONFIRMED"
                    if not stop_errors
                    else "STOP_REQUESTED_UNCONFIRMED"
                ),
            )
        else:
            try:
                machine.stop()
            except RestartRequired:
                pass
        return {
            "mission": machine.status(),
            "hardware": robot.status(),
            "stop_errors": stop_errors,
            "run": run,
        }

    return app
