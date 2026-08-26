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
from .cohorts import INTER_RUN_PAUSE_S, CohortConflict, CohortController
from .controller_start import (
    FRUIT_RUN_BUTTONS,
    ControllerStartAdapter,
    ControllerStartRefused,
    ControllerStartSource,
)
from .evidence import EvidenceArtifact
from .fruits import QUALIFIED_FRUITS, SUPPORTED_FRUITS
from .hardware import HardwareManager, HardwareUnavailable
from .mission import MissionMachine, RestartRequired
from .models import RemoteInput
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

    target_fruit: Literal["apple", "banana", "mango", "pear"] = "pear"
    activation_source: Literal["audience_ui", "voice", "soak"] = "audience_ui"
    activation_id: str | None = None
    tuning: dict[str, object] | None = None


class FruitPreviewRequest(BaseModel):
    target_fruit: Literal["apple", "banana", "mango", "pear"]


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
    target_fruit: Literal["apple", "banana", "mango", "pear"] | None = None
    fruit_subset: list[Literal["apple", "banana", "mango", "pear"]] | None = None
    seed: int | None = Field(default=None, ge=0)
    tolerated_failures: list[FailureSelectorRequest] = Field(default_factory=list)
    tuning: dict[str, object] | None = None


# The physical Go2 Start button carries no fruit, seed or tuning fields, so it
# reuses the exact payload the "Run Apple + Mango + Pear once" UI button posts
# to /api/cohorts. Banana is not motion-qualified and is deliberately absent.
CONTROLLER_START_COHORT_RUNS = 3
CONTROLLER_START_COHORT_FRUITS: tuple[str, ...] = ("apple", "mango", "pear")
# Deployed stage defaults; do not drift these without re-qualifying on Woof.
#
# The search sweep follows the same environment variable the per-fruit guidance
# defaults use. It was hardcoded, which quietly gave two different sweep speeds
# for the same stage: lowering the deployment to 0.60 rad/s changed web-UI runs
# while every controller-started run kept sweeping at 0.80, so a Pear started
# with X did not behave like the Pear the operator had just tuned.
#
# Home alignment stays a literal on purpose: 0.80 is its own qualified value and
# is unrelated to how fast the dog scans for a fruit.
CONTROLLER_START_SEARCH_YAW_RPS = float(
    os.environ.get("BORDER_COLLIE_GUIDANCE_SEARCH_YAW_RPS", "0.80")
)
CONTROLLER_START_HOME_ALIGN_YAW_RPS = 0.80
CONTROLLER_START_TOLERATED_FAILURES: tuple[dict[str, str], ...] = (
    {"reason": "TARGET_RECOGNITION_FAILURE"},
    {"reason": "TARGET_LOST"},
    {"reason": "ARRIVAL_FAILURE"},
    {"reason": "ACTION_FAILURE"},
    {"failed_phase": "turn_to_fruit"},
    {"failed_phase": "find_fruit"},
    {"failed_phase": "approach_fruit"},
    {"failed_phase": "arrived"},
    {"failed_phase": "sit_and_bark"},
    {"failed_phase": "stand"},
)

# Provenance recorded on every Demo Run the physical controller activates.
#
# Deliberately NOT added to RunRequest.activation_source: that Literal is the
# public HTTP contract, and an HTTP client must not be able to claim it drove
# the physical controller. The cohort runner already sets its own out-of-band
# "cohort" source the same way, so FruitMission.activation_source is a free
# string by design and this follows that precedent.
CONTROLLER_ACTIVATION_SOURCE = "controller"


def controller_run_tuning() -> dict[str, object]:
    """Return the stage tuning every controller-activated Demo Run uses.

    Shared verbatim by the Start cohort and the per-fruit face buttons, so a
    Pear started with X behaves exactly like the Pear inside a Start cohort.
    """
    return {
        "search": {"yaw_rps": CONTROLLER_START_SEARCH_YAW_RPS},
        "home": {"align_yaw_rps": CONTROLLER_START_HOME_ALIGN_YAW_RPS},
    }


def three_fruit_cohort_request(seed: int | None = None) -> CohortRequest:
    """Return the canonical one-Apple, one-Mango, one-Pear cohort request.

    ``seed`` only permutes the order; three qualified fruits over three runs
    always yields exactly one Demo Run each.
    """
    return CohortRequest(
        runs=CONTROLLER_START_COHORT_RUNS,
        randomized=True,
        target_fruit=None,
        fruit_subset=list(CONTROLLER_START_COHORT_FRUITS),
        seed=seed,
        tolerated_failures=[
            FailureSelectorRequest(**item)
            for item in CONTROLLER_START_TOLERATED_FAILURES
        ],
        tuning=controller_run_tuning(),
    )


def controller_start_seed() -> int | None:
    """Read the optional fixed controller seed; ``None`` means fresh per press."""
    raw = os.environ.get("BORDER_COLLIE_CONTROLLER_START_SEED", "").strip()
    if not raw:
        return None
    try:
        seed = int(raw)
    except ValueError:
        return None
    return seed if seed >= 0 else None


def create_app(
    mission: MissionMachine | None = None,
    hardware: HardwareManager | None = None,
    web_root: Path | None = None,
    runs_root: Path | None = None,
    camera_perception_status: Callable[[], dict[str, object]] | None = None,
    camera_frame: Callable[[], bytes] | None = None,
    raw_camera_frame: Callable[[], bytes] | None = None,
    select_perception_target: Callable[[str], dict[str, object]] | None = None,
    # Activation selects the Target Fruit through its own seam so a Demo Run
    # takes the long run-scoped inference lease, while the fruit-test page only
    # ever gets the short preview one.
    select_run_perception_target: Callable[[str], dict[str, object]] | None = None,
    preview_camera_perception: Callable[[], dict[str, object]] | None = None,
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
    controller_start_source: ControllerStartSource | None = None,
    inter_run_pause_s: float | None = None,
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
    read_preview_perception = preview_camera_perception or read_camera_perception

    demo = StageDemo(
        machine,
        results,
        robot,
        read_camera_perception,
        media_status=read_media,
        select_perception_target=(
            select_run_perception_target or select_perception_target
        ),
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
        inter_run_pause_s=(
            inter_run_pause_s
            if inter_run_pause_s is not None
            else float(
                os.environ.get(
                    "BORDER_COLLIE_INTER_RUN_PAUSE_S", str(INTER_RUN_PAUSE_S)
                )
            )
        ),
    )

    async def start_cohort_from_request(request: CohortRequest) -> dict[str, object]:
        """Own the single cohort-start path shared by HTTP and the Start button.

        Raises ``CohortConflict`` or ``ValueError``; callers decide how to
        surface them.
        """
        policy = CohortPolicy(
            runs=request.runs,
            randomized=request.randomized,
            target_fruit=request.target_fruit,
            fruit_subset=(
                None if request.fruit_subset is None else tuple(request.fruit_subset)
            ),
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
        return await cohorts.start(
            policy,
            list(QUALIFIED_FRUITS),
            tuning_template=request.tuning,
        )

    def refuse_unless_idle() -> None:
        """Reject a controller activation unless nothing owns the robot.

        The guards mirror the HTTP contract exactly: a latched Remote Takeover
        is the 423 case and must never be overridden from the controller, and
        an already-active cohort or Demo Run is the 409 case.  Refusing here
        rather than queueing keeps one press from stacking activations.

        The adapter already refuses to count presses while a run is active --
        those presses stop it instead -- so this is the second, authoritative
        gate against a race between a stop completing and a launch landing.
        """
        if machine.takeover_latched:
            raise ControllerStartRefused(
                "takeover_latched",
                "physical remote takeover is latched; restart required",
            )
        if cohorts.running():
            raise ControllerStartRefused(
                "active_cohort",
                "the active cohort owns Demo Run activation",
            )
        if results.active_run_id is not None:
            raise ControllerStartRefused(
                "active_run",
                "a Demo Run is already active",
            )

    async def start_controller_cohort() -> dict[str, object]:
        """Start one three-fruit cohort for one physical Start rising edge."""
        refuse_unless_idle()
        return await start_cohort_from_request(
            three_fruit_cohort_request(controller_start_seed())
        )

    async def start_controller_fruit_run(
        *, target_fruit: str, activation_id: str
    ) -> dict[str, object]:
        """Start one single-fruit Demo Run for one physical face-button triple.

        Same seam as ``POST /api/run``: one ``FruitMission`` through
        ``StageDemo.activate``, so preflight, Home capture, the per-run Home
        clearance gate and terminal exact-zero disarm all apply unchanged. The
        only differences from the audience UI are the recorded provenance and
        the fact that the tuning is the controller's, matching the Start
        cohort's runs exactly.
        """
        refuse_unless_idle()
        if target_fruit not in QUALIFIED_FRUITS:
            # Belt and braces: FRUIT_RUN_BUTTONS only maps qualified fruits, so
            # reaching here means the mapping drifted rather than the operator
            # doing something unusual. Refuse instead of activating.
            raise ControllerStartRefused(
                "rejected",
                f"unqualified Target Fruit for a controller run: {target_fruit}",
            )
        activation = await demo.activate(
            FruitMission(
                target_fruit=target_fruit,
                activation_source=CONTROLLER_ACTIVATION_SOURCE,
                activation_id=activation_id,
                tuning=RunTuning.from_payload(target_fruit, controller_run_tuning()),
            )
        )
        return {
            "run": activation.run,
            "idempotent_replay": activation.idempotent_replay,
        }

    def controller_owns_activation() -> bool:
        """True while a cohort or a Demo Run is the thing an input would stop."""
        return cohorts.running() or results.active_run_id is not None

    async def stop_for_controller_input(
        remote_input: RemoteInput,
    ) -> dict[str, object]:
        """Stop whatever the operator interrupted, down the existing safe path.

        The latch goes on first so nothing can activate in the gap, then the
        stop runs through exactly the calls ``POST /api/cohorts/active/stop``
        and ``POST /api/stop`` already make: exact-zero disarm and a sealed Run
        Result. Nothing here talks to a motion client.
        """
        machine.remote_takeover(remote_input)
        if cohorts.running():
            return {"cohort": await cohorts.stop()}
        return {"stop": await demo.stop()}

    def release_controller_takeover() -> None:
        machine.release_remote_takeover(
            "physical remote released; the application may be started again"
        )

    controller_start_adapter = (
        None
        if controller_start_source is None
        else ControllerStartAdapter(
            start_cohort=start_controller_cohort,
            start_fruit_run=start_controller_fruit_run,
            stop_active=stop_for_controller_input,
            black_box=results.black_box,
            active_run_id=lambda: results.active_run_id,
            is_busy=controller_owns_activation,
            takeover_latched=lambda: machine.takeover_latched,
            release_takeover=release_controller_takeover,
        )
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        await demo.start()
        if system_audio is not None:
            await system_audio.start_muted()
        if controller_start_source is not None and controller_start_adapter is not None:
            await controller_start_source.start(controller_start_adapter.observe)
        try:
            yield
        finally:
            if controller_start_source is not None:
                await controller_start_source.close()
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
        """Return the selected fruit's raw live observation without motion.

        The fruit-test page polls this while it is open, and each poll renews
        the short preview lease. Closing the page stops the polling, so the
        detector falls idle on its own with nothing to switch it off.
        """
        try:
            return await asyncio.to_thread(read_preview_perception)
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
            "controller_start": (
                {
                    "enabled": False,
                    "detail": "physical Start activation is disabled",
                }
                if controller_start_source is None or controller_start_adapter is None
                else {
                    "enabled": True,
                    "cohort_request": three_fruit_cohort_request(
                        controller_start_seed()
                    ).model_dump(),
                    # Which face button runs which fruit, so the operator sheet
                    # can be read off the running build instead of memory.
                    "fruit_run_buttons": dict(FRUIT_RUN_BUTTONS),
                    "run_activation_source": CONTROLLER_ACTIVATION_SOURCE,
                    "source": controller_start_source.status(),
                    "adapter": controller_start_adapter.status(),
                }
            ),
        }

    @app.post("/api/cohorts", status_code=201)
    async def start_cohort(request: CohortRequest) -> dict[str, object]:
        if request.target_fruit == "banana":
            raise HTTPException(
                status_code=409,
                detail=(
                    "Banana Demo Runs are temporarily disabled; choose Apple, "
                    "Mango, or Pear"
                ),
            )
        try:
            cohort = await start_cohort_from_request(request)
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
        if request.target_fruit == "banana":
            raise HTTPException(
                status_code=409,
                detail=(
                    "Banana Demo Runs are temporarily disabled; choose Apple, "
                    "Mango, or Pear"
                ),
            )
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

    @app.post("/api/home/recapture")
    async def recapture_home() -> dict[str, object]:
        if cohorts.running():
            raise HTTPException(
                status_code=409,
                detail="the active cohort owns Demo Run activation",
            )
        try:
            return await demo.recapture_home()
        except ActiveRunError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except HardwareUnavailable as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except RestartRequired as exc:
            raise HTTPException(status_code=423, detail=str(exc)) from exc

    @app.post("/api/stop")
    async def stop() -> dict[str, object]:
        return await demo.stop()

    @app.post("/api/bark/test")
    async def bark_test() -> dict[str, object]:
        """Sound one bark down the same path a Demo Run uses.

        Goes through the speaker policy exactly as sit_and_bark does, so this
        exercises unmute -> bark -> remute rather than only the sidecar.
        """
        return await thermal_beep()

    @app.post("/api/thermal/beep")
    async def thermal_beep() -> dict[str, object]:
        """Sound the speaker for a thermal alert.

        The thermal monitor is a separate app that posts here on a critical.
        It previously 404'd, so the alarm has never been audible. Routing it
        through the audio policy is what unmutes the speaker for the sound.
        """
        if system_audio is None:
            raise HTTPException(
                status_code=503, detail="speaker policy is not connected"
            )
        try:
            result = await system_audio.bark()
        except Exception as exc:  # noqa: BLE001 - SDK and sidecar errors are untyped
            # Report rather than raise: a failed alarm must still be visible,
            # and this endpoint must never become a way to wedge the monitor.
            return {"sounded": False, "error": str(exc)[:240]}
        return {"sounded": True, **result}

    return app
