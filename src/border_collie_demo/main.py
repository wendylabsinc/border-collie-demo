from __future__ import annotations

import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI

from .api import create_app
from .black_box import RunBlackBox
from .config import HardwareConfig, PerceptionConfig, env_bool
from .controller_start import Go2ControllerStartSource
from .evidence import TerminalEvidenceClient
from .failure_epilogue import PositionOnlyFailureEpilogue
from .hardware import HardwareManager
from .home_recording import (
    HomeRecordingHub,
    HomeRecordingStatus,
    SharedHomeEventJournal,
)
from .media import BarkClient, BarkConfig
from .operator_logging import OperatorEventLogger
from .orchestrator import SimulatedStageExecutor
from .perception import PerceptionStatusClient
from .production import ProductionStageExecutor
from .run_tuning import HomeTuning
from .simulation import SimulatedHardware, simulated_camera_perception
from .system_audio import SystemAudioConfig, create_system_audio_policy


def build_app_from_env() -> FastAPI:
    runtime_mode = (
        os.environ.get("BORDER_COLLIE_RUNTIME_MODE", "production").strip().lower()
    )
    if runtime_mode == "simulation":
        return create_app(
            hardware=SimulatedHardware(),
            camera_perception_status=simulated_camera_perception,
            media_status=lambda: {"ready": True, "detail": "simulated bark is ready"},
            stage_executor=SimulatedStageExecutor(),
            runtime_mode="simulation",
        )
    if runtime_mode != "production":
        raise ValueError("BORDER_COLLIE_RUNTIME_MODE must be production or simulation")
    runs_root = Path(
        os.environ.get("BORDER_COLLIE_RUNS_DIR", "artifacts/runs")
    ).resolve()
    black_box = HomeRecordingHub(
        RunBlackBox(runs_root, operator_sink=OperatorEventLogger.from_env()),
        SharedHomeEventJournal(
            Path(
                os.environ.get(
                    "BORDER_COLLIE_HOME_RECORDING_EVENTS_DIR",
                    "/state/home-recorder/events",
                )
            )
        ),
    )
    recording_status = HomeRecordingStatus(
        black_box,
        passive_status_url=os.environ.get(
            "BORDER_COLLIE_HOME_RECORDER_STATUS_URL",
            "http://127.0.0.1:8112/status",
        ),
    )
    hardware = HardwareManager(HardwareConfig.from_env(), black_box=black_box)
    perception = PerceptionStatusClient(PerceptionConfig.from_env())
    bark = BarkClient(BarkConfig.from_env())
    system_audio = create_system_audio_policy(
        bark,
        SystemAudioConfig.from_env(),
    )

    def best_effort_bark_status() -> dict[str, object]:
        try:
            status = system_audio.status()
        except Exception as exc:  # noqa: BLE001 - bark cannot block motion readiness
            status = {"ready": False, "detail": str(exc)}
        return {
            "ready": True,
            "detail": "bark is best effort and does not block Demo Run readiness",
            "bark_ready": status.get("ready") is True,
            "bark_detail": status.get("detail"),
        }

    terminal_evidence = TerminalEvidenceClient.from_env()
    controller_start_source = (
        Go2ControllerStartSource()
        if env_bool("BORDER_COLLIE_CONTROLLER_START_ENABLED")
        else None
    )
    return create_app(
        hardware=hardware,
        camera_perception_status=perception.status,
        camera_frame=perception.camera_frame,
        select_perception_target=perception.select_target,
        runs_root=runs_root,
        media_status=best_effort_bark_status,
        stage_executor=ProductionStageExecutor(
            hardware, perception.status, system_audio
        ),
        terminal_evidence=terminal_evidence.capture,
        failure_epilogue=PositionOnlyFailureEpilogue(
            hardware,
            arrival_tolerance_m=HomeTuning.from_env().arrival_tolerance_m,
        ),
        black_box=black_box,
        system_audio=system_audio,
        recording_status=recording_status,
        home_recordings_root=Path(
            os.environ.get(
                "BORDER_COLLIE_HOME_RECORDINGS_DIR",
                "/state/home-recorder/runs",
            )
        ),
        runtime_mode="production",
        controller_start_source=controller_start_source,
    )


def main() -> None:
    port = int(os.environ.get("BORDER_COLLIE_PORT", "8110"))
    uvicorn.run(
        build_app_from_env(),
        host="0.0.0.0",
        port=port,
        access_log=False,
    )


if __name__ == "__main__":
    main()
