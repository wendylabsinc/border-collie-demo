from __future__ import annotations

import os

import uvicorn
from fastapi import FastAPI

from .api import create_app
from .config import HardwareConfig, PerceptionConfig
from .evidence import TerminalEvidenceClient
from .hardware import HardwareManager
from .media import BarkClient, BarkConfig
from .orchestrator import SimulatedStageExecutor
from .perception import PerceptionStatusClient
from .production import ProductionStageExecutor
from .release import ReleaseCohort
from .simulation import SimulatedHardware, simulated_camera_perception


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
    hardware = HardwareManager(HardwareConfig.from_env())
    release_cohort = ReleaseCohort.from_env("app")
    perception = PerceptionStatusClient(
        PerceptionConfig.from_env(), release_cohort=release_cohort
    )
    bark = BarkClient(BarkConfig.from_env(), release_cohort=release_cohort)
    terminal_evidence = TerminalEvidenceClient.from_env()
    return create_app(
        hardware=hardware,
        camera_perception_status=perception.status,
        camera_frame=perception.camera_frame,
        select_perception_target=perception.select_target,
        media_status=bark.status,
        stage_executor=ProductionStageExecutor(hardware, perception.status, bark),
        terminal_evidence=terminal_evidence.capture,
        release_cohort=release_cohort,
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
