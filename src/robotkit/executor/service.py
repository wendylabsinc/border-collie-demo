"""Claim, validate, apply, and durably acknowledge effects."""

from __future__ import annotations

import os

from robotkit.client import WorldStateClient
from robotkit.contracts import EffectCompletion
from robotkit.executor.adapters import (
    EffectAdapter,
    LogAdapter,
    Ros2StringCommandAdapter,
    Ros2TwistAdapter,
    RoutingAdapter,
    UnitreeSportAdapter,
    WavPlaybackAdapter,
)
from robotkit.executor.safety import validate_effect
from robotkit.runtime import configure_logging, instance_id, run_loop, world_state_url


def _adapter() -> EffectAdapter:
    mode = os.getenv("ROBOTKIT_EXECUTOR_MODE", "log")
    if mode == "log":
        return LogAdapter()
    if mode == "ros2":
        twist = Ros2TwistAdapter(os.getenv("ROS2_CMD_VEL_TOPIC", "/cmd_vel"))
        bark = WavPlaybackAdapter(
            os.getenv("BARK_WAV_PATH", "/opt/robotkit/assets/bark.wav"),
            player=os.getenv("AUDIO_PLAYER") or None,
            max_duration_seconds=float(os.getenv("MAX_BARK_SECONDS", "10")),
        )
        return RoutingAdapter(
            {
                "unitree_lie_down": Ros2StringCommandAdapter(
                    os.getenv("ROS2_POSTURE_COMMAND_TOPIC", "/robotkit/posture_command")
                ),
                "cmd_vel": twist,
                "unitree_bark": bark,
                "audio": bark,
            }
        )
    if mode == "unitree_sport":
        sport = UnitreeSportAdapter(
            os.getenv("GO2_NETWORK_INTERFACE", "enP8p1s0"),
            watchdog_seconds=float(os.getenv("UNITREE_MOTION_WATCHDOG_SECONDS", "0.8")),
        )
        bark = WavPlaybackAdapter(
            os.getenv("BARK_WAV_PATH", "/opt/robotkit/assets/bark.wav"),
            player=os.getenv("AUDIO_PLAYER") or None,
            max_duration_seconds=float(os.getenv("MAX_BARK_SECONDS", "10")),
        )
        return RoutingAdapter(
            {
                "cmd_vel": sport,
                "unitree_lie_down": sport,
                "unitree_bark": bark,
                "audio": bark,
            }
        )
    raise ValueError(f"unknown ROBOTKIT_EXECUTOR_MODE: {mode}")


def main() -> None:
    configure_logging()
    client = WorldStateClient(world_state_url())
    adapter = _adapter()
    executor_id = instance_id()

    def step() -> None:
        effect = client.claim_effect(executor_id)
        if effect is None:
            return
        snapshot = client.snapshot()
        goal = client.current_goal()
        if goal is None or goal.status != "active":
            client.complete_effect(
                effect.effect_id,
                EffectCompletion(
                    executor_id=executor_id,
                    status="rejected",
                    result={"reason": "no active goal"},
                ),
            )
            return
        verdict = validate_effect(
            effect,
            snapshot,
            snapshot.captured_at,
            active_goal_id=goal.goal_id,
            max_linear_mps=float(os.getenv("MAX_LINEAR_MPS", "0.5")),
            max_angular_rps=float(os.getenv("MAX_ANGULAR_RPS", "1.0")),
        )
        if not verdict.allowed:
            client.complete_effect(
                effect.effect_id,
                EffectCompletion(
                    executor_id=executor_id,
                    status="rejected",
                    result={"reason": verdict.reason},
                ),
            )
            return
        try:
            result = adapter.apply(effect)
        except Exception as exc:
            client.complete_effect(
                effect.effect_id,
                EffectCompletion(
                    executor_id=executor_id,
                    status="failed",
                    result={"error": type(exc).__name__, "message": str(exc)},
                ),
            )
            return
        client.complete_effect(
            effect.effect_id,
            EffectCompletion(executor_id=executor_id, status="applied", result=result),
        )

    try:
        run_loop(step, float(os.getenv("INTERVAL_SECONDS", "0.05")))
    finally:
        adapter.close()
        client.close()


if __name__ == "__main__":
    main()
