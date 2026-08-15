from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from .black_box import RunBlackBox
from .cohort_policy import CohortPolicy, FailureSelector
from .cohorts import CohortConflict, CohortController
from .evidence import EvidenceArtifact
from .fruits import QUALIFIED_FRUITS, SUPPORTED_FRUITS
from .hardware import HardwareManager, HardwareUnavailable
from .mission import MissionMachine, RestartRequired
from .orchestrator import EXECUTED_STAGES, FailureEpilogue, StageExecutor
from .run_results import ActiveRunError, RunResultNotFound, RunResultStore
from .run_tuning import RunTuning
from .search_experiment import search_experiment_contract, search_experiment_scorecard
from .stage_demo import ActivationConflict, FruitMission, StageDemo
from .system_audio import SystemAudioPolicy


def build_label() -> str:
    """Identify which demo branch is deployed.

    Display only. Three branches are deployed to the same robot one at a time,
    so an operator needs to confirm from the UI which build produced a run
    before recording its result against a branch.
    """
    return (
        os.environ.get("BORDER_COLLIE_BUILD_LABEL", "unlabelled").strip()
        or "unlabelled"
    )


class ForwardPulseRequest(BaseModel):
    confirmation: str


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_fruit: Literal["apple", "banana", "pear"] = "pear"
    activation_source: Literal["audience_ui", "voice", "soak"] = "audience_ui"
    activation_id: str | None = None
    tuning: dict[str, object] | None = None


class FruitPreviewRequest(BaseModel):
    target_fruit: Literal["apple", "banana", "pear"]


class CocoTestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool
    minimum_confidence: float | None = Field(default=None, ge=0.01, le=0.95)
    reset: bool = False


class FailureSelectorRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    failed_phase: str | None = None
    reason: str | None = None


class CohortRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    runs: int = Field(default=5, ge=1, le=100)
    randomized: bool = True
    target_fruit: Literal["apple", "banana", "pear"] | None = None
    seed: int | None = Field(default=None, ge=0)
    tolerated_failures: list[FailureSelectorRequest] = Field(default_factory=list)


def create_app(
    mission: MissionMachine | None = None,
    hardware: HardwareManager | None = None,
    web_root: Path | None = None,
    runs_root: Path | None = None,
    camera_perception_status: Callable[[], dict[str, object]] | None = None,
    camera_frame: Callable[[], bytes] | None = None,
    raw_camera_frame: Callable[[], bytes] | None = None,
    select_perception_target: Callable[[str], dict[str, object]] | None = None,
    media_status: Callable[[], dict[str, object]] | None = None,
    coco_test_status: Callable[[], dict[str, object]] | None = None,
    configure_coco_test: (
        Callable[[dict[str, object]], dict[str, object]] | None
    ) = None,
    stage_executor: StageExecutor | None = None,
    terminal_evidence: Callable[[], list[EvidenceArtifact]] | None = None,
    failure_epilogue: FailureEpilogue | None = None,
    black_box: RunBlackBox | None = None,
    system_audio: SystemAudioPolicy | None = None,
    runtime_mode: Literal["production", "simulation"] = "production",
    stage_home_margin_m: float | None = None,
    cohorts_root: Path | None = None,
) -> FastAPI:
    machine = mission or MissionMachine()
    robot = hardware or HardwareManager()
    root = web_root or Path(os.environ.get("BORDER_COLLIE_WEB_ROOT", "web")).resolve()
    run_storage_root = runs_root or Path(
        os.environ.get("BORDER_COLLIE_RUNS_DIR", "artifacts/runs")
    )
    results = RunResultStore(run_storage_root, black_box=black_box)
    read_camera_perception = camera_perception_status or (
        lambda: {
            "ready": False,
            "detail": "production camera/perception adapter is not connected",
        }
    )
    read_media = media_status

    demo = StageDemo(
        machine,
        results,
        robot,
        read_camera_perception,
        media_status=read_media,
        select_perception_target=select_perception_target,
        stage_executor=stage_executor,
        terminal_evidence=terminal_evidence,
        failure_epilogue=failure_epilogue,
        stage_home_margin_m=(
            stage_home_margin_m
            if stage_home_margin_m is not None
            else float(os.environ.get("BORDER_COLLIE_STAGE_HOME_MARGIN_M", "0.50"))
        ),
    )
    cohorts = CohortController(
        demo,
        cohorts_root
        or Path(
            os.environ.get(
                "BORDER_COLLIE_COHORTS_DIR",
                str(run_storage_root.parent / "cohorts"),
            )
        ),
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        await demo.start()
        if system_audio is not None:
            await system_audio.start_muted()
        try:
            yield
        finally:
            await cohorts.close()
            await demo.close()
            if system_audio is not None:
                await system_audio.close()
            results.black_box.close()

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

    @app.get("/api/camera/raw.jpg")
    async def raw_camera_frame_proxy() -> Response:
        if raw_camera_frame is None:
            raise HTTPException(
                status_code=503, detail="raw camera preview is not connected"
            )
        try:
            jpeg = await asyncio.to_thread(raw_camera_frame)
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"raw camera preview unavailable: {exc}",
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

    @app.get("/api/fruits/preview")
    async def preview_fruit_status() -> dict[str, object]:
        """Return the selected fruit's raw live observation without motion."""
        try:
            return await asyncio.to_thread(read_camera_perception)
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"fruit perception status unavailable: {exc}",
            ) from exc

    @app.get("/api/coco-test")
    async def read_coco_test() -> dict[str, object]:
        if coco_test_status is None:
            raise HTTPException(status_code=503, detail="COCO tester is not connected")
        try:
            return await asyncio.to_thread(coco_test_status)
        except Exception as exc:
            raise HTTPException(
                status_code=503, detail=f"COCO tester unavailable: {exc}"
            ) from exc

    @app.post("/api/coco-test")
    async def change_coco_test(request: CocoTestRequest) -> dict[str, object]:
        if results.active_run_id is not None:
            raise HTTPException(
                status_code=409,
                detail="COCO tester cannot change during an active Demo Run",
            )
        if configure_coco_test is None:
            raise HTTPException(status_code=503, detail="COCO tester is not connected")
        payload = request.model_dump()
        try:
            return await asyncio.to_thread(configure_coco_test, payload)
        except Exception as exc:
            raise HTTPException(
                status_code=503, detail=f"COCO tester configuration failed: {exc}"
            ) from exc

    @app.get("/api/status")
    async def status() -> dict[str, object]:
        current = demo.status()
        return {
            "build_label": build_label(),
            "runtime_mode": runtime_mode,
            **current,
            "run_tuning": RunTuning.contract(),
            "cohort": cohorts.current(),
            "search_experiment": search_experiment_contract(),
        }

    @app.post("/api/cohorts", status_code=201)
    async def start_cohort(request: CohortRequest) -> dict[str, object]:
        try:
            policy = CohortPolicy(
                runs=request.runs,
                randomized=request.randomized,
                target_fruit=request.target_fruit,
                seed=(
                    request.seed
                    if request.seed is not None
                    else int.from_bytes(os.urandom(8), "big")
                ),
                tolerated_failures=tuple(
                    FailureSelector(
                        failed_phase=item.failed_phase,
                        reason=item.reason,
                    )
                    for item in request.tolerated_failures
                ),
            )
            cohort = await cohorts.start(policy, list(QUALIFIED_FRUITS))
        except (CohortConflict, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"cohort": cohort}

    @app.get("/api/cohorts/active")
    async def active_cohort() -> dict[str, object]:
        return {"cohort": cohorts.current()}

    @app.post("/api/cohorts/active/stop")
    async def stop_cohort() -> dict[str, object]:
        return {"cohort": await cohorts.stop()}

    @app.get("/api/cohorts/{cohort_id}")
    async def get_cohort(cohort_id: str) -> dict[str, object]:
        cohort = cohorts.get(cohort_id)
        if cohort is None:
            raise HTTPException(status_code=404, detail="cohort not found")
        return {"cohort": cohort}

    @app.post("/api/run", status_code=201)
    async def activate_run(request: RunRequest) -> dict[str, object]:
        if cohorts.running():
            raise HTTPException(
                status_code=409,
                detail="the active cohort owns Demo Run activation",
            )
        try:
            activation = await demo.activate(
                FruitMission(
                    target_fruit=request.target_fruit,
                    activation_source=request.activation_source,
                    activation_id=request.activation_id or str(uuid4()),
                    tuning=RunTuning.from_payload(request.target_fruit, request.tuning),
                )
            )
        except ActiveRunError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (ActivationConflict, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except RestartRequired as exc:
            raise HTTPException(status_code=423, detail=str(exc)) from exc
        return {
            "run": activation.run,
            "idempotent_replay": activation.idempotent_replay,
        }

    @app.get("/api/results/{run_id}")
    async def get_result(run_id: str) -> dict[str, object]:
        try:
            return {"run": results.get(run_id)}
        except RunResultNotFound as exc:
            raise HTTPException(status_code=404, detail="Run Result not found") from exc

    @app.get("/api/results/{run_id}/black-box.ndjson")
    async def get_black_box(run_id: str) -> FileResponse:
        try:
            path = results.black_box.path(run_id)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(
                status_code=404, detail="Run black-box trace not found"
            ) from exc
        return FileResponse(
            path,
            media_type="application/x-ndjson",
            filename=f"{run_id}-black-box.ndjson",
        )

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

    @app.get("/api/experiments/search")
    async def search_experiments() -> dict[str, object]:
        return search_experiment_scorecard(results.list_results())

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
        return await demo.stop()

    return app
