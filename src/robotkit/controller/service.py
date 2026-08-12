"""Disposable wrapper around the pure controller."""

from __future__ import annotations

import os
from datetime import timedelta
import math
from datetime import datetime

from robotkit.client import WorldStateClient
from robotkit.contracts import Effect, Preconditions
from robotkit.controller.logic import control
from robotkit.executor.mission import MissionConfig
from robotkit.runtime import (
    configure_logging,
    deployment_generation,
    instance_id,
    run_loop,
    world_state_url,
)


def control_tick(at: datetime, renewal_seconds: float) -> int:
    """Replica-stable renewal identity derived only from A's capture time."""
    if renewal_seconds <= 0:
        raise ValueError("renewal_seconds must be positive")
    return math.floor(at.timestamp() / renewal_seconds)


def main() -> None:
    configure_logging()
    client = WorldStateClient(world_state_url())
    controller_id = os.getenv("ROBOTKIT_CONTROLLER_ID", "velocity-controller-v1")
    deployment = instance_id()
    generation = deployment_generation()
    mission_config = MissionConfig(
        search_angular_rps=float(os.getenv("SEARCH_ANGULAR_RPS", "0.45")),
        max_linear_mps=float(os.getenv("MISSION_MAX_LINEAR_MPS", "0.30")),
        max_angular_rps=float(os.getenv("MISSION_MAX_ANGULAR_RPS", "0.80")),
        steering_gain=float(os.getenv("MISSION_STEERING_GAIN", "1.5")),
        camera_horizontal_fov_rad=float(
            os.getenv("CAMERA_HORIZONTAL_FOV_RAD", "1.5707963267948966")
        ),
        apple_stop_distance_m=float(os.getenv("APPLE_STOP_DISTANCE_M", "0.30")),
        home_x_m=float(os.getenv("HOME_X_M", "0")),
        home_y_m=float(os.getenv("HOME_Y_M", "0")),
        home_yaw_rad=float(os.getenv("HOME_YAW_RAD", "0")),
        home_position_tolerance_m=float(os.getenv("HOME_POSITION_TOLERANCE_M", "0.15")),
        home_yaw_tolerance_rad=float(os.getenv("HOME_YAW_TOLERANCE_RAD", "0.15")),
    )
    effect_ttl_seconds = float(os.getenv("EFFECT_TTL_SECONDS", "1"))
    renewal_seconds = float(os.getenv("CONTROL_RENEWAL_SECONDS", "0.25"))
    if renewal_seconds <= 0 or renewal_seconds >= effect_ttl_seconds:
        raise ValueError(
            "CONTROL_RENEWAL_SECONDS must be positive and less than EFFECT_TTL_SECONDS"
        )

    def step() -> None:
        snapshot = client.snapshot()
        goal = client.current_goal()
        if goal is None or goal.status != "active" or goal.valid_until <= snapshot.captured_at:
            return
        decision = control(snapshot, goal, mission_config=mission_config)
        # captured_at is supplied by durable A, not local process state. The
        # bucket makes velocity renewal deterministic and replica-idempotent
        # while allowing an unchanged world projection to keep the Go2's
        # dead-man command alive.
        tick = control_tick(snapshot.captured_at, renewal_seconds)
        client.publish_effect(
            Effect(
                idempotency_key=(
                    f"{controller_id}:generation:{generation}:goal:{goal.goal_id}:"
                    f"state:{snapshot.state_revision}:tick:{tick}"
                ),
                controller_id=controller_id,
                instance_id=deployment,
                deployment_generation=generation,
                goal_id=goal.goal_id,
                based_on_revision=snapshot.state_revision,
                effect_type=decision.effect_type,
                created_at=snapshot.captured_at,
                valid_until=snapshot.captured_at
                + timedelta(seconds=effect_ttl_seconds),
                parameters=decision.parameters,
                preconditions=Preconditions(
                    max_world_revision_drift=int(os.getenv("MAX_STATE_DRIFT", "4")),
                    require_fresh_streams=decision.require_fresh_streams,
                ),
            )
        )

    try:
        run_loop(step, float(os.getenv("INTERVAL_SECONDS", "0.25")))
    finally:
        client.close()


if __name__ == "__main__":
    main()
