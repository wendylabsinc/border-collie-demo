from __future__ import annotations

import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI

from .api import create_app
from .black_box import RunBlackBox
from .config import HardwareConfig, PerceptionConfig
from .evidence import TerminalEvidenceClient
from .failure_epilogue import PositionOnlyFailureEpilogue
from .hardware import HardwareManager
from .media import BarkClient, BarkConfig
from .orchestrator import SimulatedStageExecutor
from .perception import PerceptionStatusClient
from .production import ProductionStageExecutor
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
    black_box = RunBlackBox(runs_root)
    hardware = HardwareManager(HardwareConfig.from_env(), black_box=black_box)
    perception = PerceptionStatusClient(PerceptionConfig.from_env())
    bark = BarkClient(BarkConfig.from_env())
    system_audio = create_system_audio_policy(
        bark,
        SystemAudioConfig.from_env(),
    )
    terminal_evidence = TerminalEvidenceClient.from_env()
    return create_app(
        hardware=hardware,
        camera_perception_status=perception.status,
        camera_frame=perception.camera_frame,
        select_perception_target=perception.select_target,
        runs_root=runs_root,
        media_status=system_audio.status,
        stage_executor=ProductionStageExecutor(
            hardware, perception.status, system_audio
        ),
        terminal_evidence=terminal_evidence.capture,
        failure_epilogue=PositionOnlyFailureEpilogue(hardware),
        black_box=black_box,
        system_audio=system_audio,
        runtime_mode="production",
    )


def main() -> None:
    port = int(os.environ.get("BORDER_COLLIE_PORT", "8110"))
    uvicorn.run(
        build_app_from_env(),
        host="0.0.0.0",
        port=port,
    )


if __name__ == "__main__":
    main()
