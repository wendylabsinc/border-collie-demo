from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .hardware import HardwareManager, HardwareUnavailable
from .mission import MissionMachine, RestartRequired
from .orchestrator import EXECUTED_STAGES, DemoOrchestrator, StageExecutor
from .preflight import evaluate_preflight, preflight_check_ready
from .run_results import ActiveRunError, RunResultNotFound, RunResultStore


class ForwardPulseRequest(BaseModel):
    confirmation: str


class RunRequest(BaseModel):
    target_fruit: Literal["pear"] = "pear"


def create_app(
    mission: MissionMachine | None = None,
    hardware: HardwareManager | None = None,
    web_root: Path | None = None,
    runs_root: Path | None = None,
    camera_perception_status: Callable[[], dict[str, object]] | None = None,
    media_status: Callable[[], dict[str, object]] | None = None,
    stage_executor: StageExecutor | None = None,
    runtime_mode: Literal["production", "simulation"] = "production",
) -> FastAPI:
    machine = mission or MissionMachine()
    robot = hardware or HardwareManager()
    root = web_root or Path(os.environ.get("BORDER_COLLIE_WEB_ROOT", "web")).resolve()
    results = RunResultStore(
        runs_root
        or Path(os.environ.get("BORDER_COLLIE_RUNS_DIR", "artifacts/runs"))
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
    orchestrator = (
        None
        if stage_executor is None
        else DemoOrchestrator(machine, results, stage_executor)
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        results.seal_interrupted_runs()
        await robot.start()
        try:
            yield
        finally:
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
            "runtime_mode": runtime_mode,
            "mission": machine.status(),
            "hardware": robot.status(),
            "active_run_id": results.active_run_id,
            "activation": {
                "ready": preflight["ready"],
                "blockers": [
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
                activation_source="audience_ui",
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
                        evaluate_preflight(robot.status(), camera_perception, media),
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
                    message="Home captured; waiting for the qualified pear command",
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
