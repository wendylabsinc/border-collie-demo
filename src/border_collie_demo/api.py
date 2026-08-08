from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from time import monotonic
from typing import Literal

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .evidence import EvidenceArtifact
from .fruits import QUALIFIED_FRUITS, SUPPORTED_FRUITS
from .hardware import HardwareManager, HardwareUnavailable
from .mission import MissionMachine, RestartRequired
from .models import RemoteInput
from .orchestrator import EXECUTED_STAGES, DemoOrchestrator, StageExecutor
from .ports import RemoteInputPort
from .preflight import evaluate_preflight, preflight_check_ready
from .run_results import ActiveRunError, RunResultNotFound, RunResultStore


class ForwardPulseRequest(BaseModel):
    confirmation: str


class RunRequest(BaseModel):
    target_fruit: Literal["apple", "banana", "pear"] = "pear"
    activation_source: Literal["audience_ui", "voice"] = "audience_ui"


class FruitPreviewRequest(BaseModel):
    target_fruit: Literal["apple", "banana", "pear"]


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
    remote_input: RemoteInputPort | None = None,
    runtime_mode: Literal["production", "simulation"] = "production",
) -> FastAPI:
    machine = mission or MissionMachine()
    robot = hardware or HardwareManager()
    root = web_root or Path(os.environ.get("BORDER_COLLIE_WEB_ROOT", "web")).resolve()
    results = RunResultStore(
        runs_root or Path(os.environ.get("BORDER_COLLIE_RUNS_DIR", "artifacts/runs"))
    )
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
    takeover_lock = asyncio.Lock()
    orchestrator = (
        None
        if stage_executor is None
        else DemoOrchestrator(
            machine,
            results,
            stage_executor,
            terminal_evidence=terminal_evidence,
        )
    )

    def current_remote_input_status() -> dict[str, object] | None:
        if remote_input is None:
            return None
        try:
            return remote_input.status()
        except Exception as exc:  # noqa: BLE001 - external adapter boundary
            return {
                "ready": False,
                "error": f"physical controller monitor status failed: {exc}",
            }

    async def handle_remote_takeover(
        received: RemoteInput,
        *,
        monitor_error: str | None = None,
    ) -> None:
        async with takeover_lock:
            if machine.takeover_latched:
                return
            interrupted_phase = machine.phase.value
            active_run_id = results.active_run_id
            machine.remote_takeover(received)

            tasks = list(active_tasks)
            for task in tasks:
                task.cancel()

            stop_errors = await robot.emergency_stop()
            if stage_executor is not None:
                stop_errors.extend(await stage_executor.stop())
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            active_tasks.difference_update(tasks)

            if active_run_id is not None:
                hardware_after_release = robot.status()
                motion_after_release = hardware_after_release.get("motion")
                results.seal(
                    active_run_id,
                    phase=machine.phase.value,
                    outcome="REMOTE_TAKEOVER",
                    reason="REMOTE_TAKEOVER",
                    message=(
                        "physical controller took authority; application restart required"
                    ),
                    final_safety_state="REMOTE_OWNED",
                    failed_phase=interrupted_phase,
                    failure_details={
                        "interrupted_phase": interrupted_phase,
                        "remote_input": {
                            "source": received.source,
                            "control": received.control,
                            "received_monotonic_s": received.received_monotonic_s,
                        },
                        "monitor_error": monitor_error,
                        "application_stop_errors": stop_errors,
                        "application_motion_after_release": (
                            {
                                "armed": motion_after_release.get("armed"),
                                "mode": motion_after_release.get("mode"),
                                "last_command": motion_after_release.get("last_command"),
                            }
                            if isinstance(motion_after_release, dict)
                            else None
                        ),
                    },
                )

    async def watch_remote_input() -> None:
        assert remote_input is not None
        try:
            await remote_input.watch(handle_remote_takeover)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - monitor loss fails closed
            await handle_remote_takeover(
                RemoteInput(
                    source="unitree_remote_monitor",
                    control="monitor_failure",
                    received_monotonic_s=monotonic(),
                ),
                monitor_error=str(exc),
            )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        results.seal_interrupted_runs()
        await robot.start()
        remote_watch_task = (
            asyncio.create_task(watch_remote_input())
            if remote_input is not None
            else None
        )
        if remote_watch_task is not None:
            await asyncio.sleep(0)
        try:
            yield
        finally:
            if remote_watch_task is not None:
                remote_watch_task.cancel()
                await asyncio.gather(remote_watch_task, return_exceptions=True)
            tasks = list(active_tasks)
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            await robot.close()

    app = FastAPI(
        title="Border Collie Demo",
        version="0.1.0",
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
        remote_status = current_remote_input_status()
        preflight = evaluate_preflight(
            robot.status(), camera_perception, media, remote_status
        )
        takeover_blocker = (
            [
                {
                    "name": "remote_takeover_latched",
                    "detail": "physical controller owns authority; restart required",
                }
            ]
            if machine.takeover_latched
            else []
        )
        return {
            "runtime_mode": runtime_mode,
            "mission": machine.status(),
            "hardware": robot.status(),
            "active_run_id": results.active_run_id,
            "remote_input": remote_status,
            "activation": {
                "ready": preflight["ready"] and not machine.takeover_latched,
                "blockers": takeover_blocker + [
                    {
                        "name": check["name"],
                        "detail": check["detail"],
                    }
                    for check in preflight["checks"]
                    if not check["ready"]
                ],
            },
        }

    @app.post("/api/run", status_code=201)
    async def activate_run(request: RunRequest) -> dict[str, object]:
        if machine.takeover_latched:
            raise HTTPException(
                status_code=423,
                detail="physical remote takeover is latched; restart required",
            )
        try:
            run = results.start_run(
                target_fruit=request.target_fruit,
                activation_source=request.activation_source,
            )
        except ActiveRunError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        machine.begin_run("Demo Run activation persisted")
        run = results.enter_phase(
            run["run_id"],
            phase=machine.phase.value,
            reason="PREFLIGHT_STARTED",
            message="preflight entered; verifying production motion and media gates",
        )
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
        report = evaluate_preflight(
            robot.status(),
            camera_perception,
            media,
            current_remote_input_status(),
        )
        run = results.record_preflight(run["run_id"], report)
        if not report["ready"]:
            stop_errors = await robot.emergency_stop()
            machine.fail("preflight readiness failed")
            run = results.seal(
                run["run_id"],
                phase=machine.phase.value,
                outcome="FAILED",
                reason="PREFLIGHT_FAILURE",
                message="required preflight readiness checks did not pass",
                final_safety_state=(
                    "DISARMED_CONFIRMED"
                    if not stop_errors
                    and preflight_check_ready(
                        evaluate_preflight(
                            robot.status(),
                            camera_perception,
                            media,
                            current_remote_input_status(),
                        ),
                        "motion_disarmed",
                    )
                    else "STOP_REQUESTED_UNCONFIRMED"
                ),
                failed_phase="preflight",
            )
        else:
            machine.advance("preflight readiness passed")
            run = results.enter_phase(
                run["run_id"],
                phase=machine.phase.value,
                reason="CAPTURE_HOME_STARTED",
                message="preflight passed; ready to capture Home",
            )
            try:
                home = robot.capture_home()
                run = results.record_home(run["run_id"], home)
            except Exception as exc:  # noqa: BLE001 - hardware evidence boundary
                stop_errors = await robot.emergency_stop()
                machine.fail("fresh Home pose capture failed")
                run = results.seal(
                    run["run_id"],
                    phase=machine.phase.value,
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
                machine.advance("fresh Home pose captured")
                run = results.enter_phase(
                    run["run_id"],
                    phase=machine.phase.value,
                    reason="WAITING_FOR_COMMAND",
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
        failed_phase = (latest or {}).get("failed_phase")
        stages = []
        for phase in EXECUTED_STAGES:
            if phase.value in completed:
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
        try:
            machine.stop()
        except RestartRequired:
            pass
        run = None
        if results.active_run_id is not None:
            run = results.seal(
                results.active_run_id,
                phase=machine.phase.value,
                outcome="STOPPED",
                reason="OPERATOR_STOP",
                message="operator stopped the Demo Run",
                final_safety_state=(
                    "DISARMED_CONFIRMED"
                    if not stop_errors
                    else "STOP_REQUESTED_UNCONFIRMED"
                ),
            )
        return {
            "mission": machine.status(),
            "hardware": robot.status(),
            "stop_errors": stop_errors,
            "run": run,
        }

    return app
