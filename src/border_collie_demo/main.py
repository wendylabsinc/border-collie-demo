from __future__ import annotations

import argparse
import os
from collections.abc import Sequence

import uvicorn
from fastapi import FastAPI

from .api import create_app
from .config import HardwareConfig, PerceptionConfig
from .evidence import TerminalEvidenceClient
from .go2_lidar import PearLidarHandoffConfig, PearLidarHandoffProvider
from .hardware import HardwareManager
from .media import BarkClient, BarkConfig
from .orchestrator import SimulatedStageExecutor
from .perception import PerceptionStatusClient
from .production import ProductionStageExecutor
from .release import ReleaseCohort
from .search_policy import SEARCH_POLICY_NAMES, SearchPolicy
from .simulation import SimulatedHardware, simulated_camera_perception


def build_app_from_env(*, search_policy: SearchPolicy | None = None) -> FastAPI:
    selected_search_policy = search_policy or SearchPolicy.configured()
    runtime_mode = (
        os.environ.get("BORDER_COLLIE_RUNTIME_MODE", "production").strip().lower()
    )
    if runtime_mode == "simulation":
        return create_app(
            hardware=SimulatedHardware(),
            camera_perception_status=simulated_camera_perception,
            media_status=lambda: {"ready": True, "detail": "simulated bark is ready"},
            stage_executor=SimulatedStageExecutor(),
            search_policy=selected_search_policy,
            runtime_mode="simulation",
        )
    if runtime_mode != "production":
        raise ValueError("BORDER_COLLIE_RUNTIME_MODE must be production or simulation")
    release_cohort = ReleaseCohort.from_env("app")
    perception = PerceptionStatusClient(
        PerceptionConfig.from_env(), release_cohort=release_cohort
    )
    hardware_config = HardwareConfig.from_env()
    lidar_config = PearLidarHandoffConfig.from_env()
    metric_range_provider = (
        PearLidarHandoffProvider(lidar_config)
        if hardware_config.metric_arrival_required and lidar_config.enabled
        else None
    )
    hardware = HardwareManager(
        hardware_config,
        visual_odometry=perception,
        metric_range_provider=metric_range_provider,
        search_policy=selected_search_policy,
    )
    bark = BarkClient(BarkConfig.from_env(), release_cohort=release_cohort)
    terminal_evidence = TerminalEvidenceClient.from_env()
    return create_app(
        hardware=hardware,
        camera_perception_status=perception.status,
        camera_frame=perception.camera_frame,
        select_perception_target=perception.select_target,
        media_status=bark.status,
        stage_executor=ProductionStageExecutor(
            hardware,
            perception.status,
            bark,
            metric_arrival_required=hardware_config.metric_arrival_required,
            search_policy=selected_search_policy,
        ),
        terminal_evidence=terminal_evidence.capture,
        release_cohort=release_cohort,
        search_policy=selected_search_policy,
        runtime_mode="production",
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the Border Collie demo app")
    parser.add_argument("--search-policy", choices=SEARCH_POLICY_NAMES)
    args = parser.parse_args(argv)
    search_policy = SearchPolicy.configured(cli_name=args.search_policy)
    port = int(os.environ.get("BORDER_COLLIE_PORT", "8110"))
    uvicorn.run(
        build_app_from_env(search_policy=search_policy),
        host="0.0.0.0",
        port=port,
    )


if __name__ == "__main__":
    main()
