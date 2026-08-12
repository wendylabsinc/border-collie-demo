"""Disposable wrapper around the pure planner."""

from __future__ import annotations

import os
from datetime import timedelta

from robotkit.client import WorldStateClient
from robotkit.contracts import Goal
from robotkit.planner.logic import plan
from robotkit.runtime import (
    configure_logging,
    deployment_generation,
    instance_id,
    run_loop,
    world_state_url,
)


def main() -> None:
    configure_logging()
    client = WorldStateClient(world_state_url())
    planner_id = os.getenv("ROBOTKIT_PLANNER_ID", "deterministic-planner-v1")
    deployment = instance_id()
    generation = deployment_generation()

    def step() -> None:
        snapshot = client.snapshot()
        current_goal = client.current_goal()
        latest_effect = (
            client.latest_effect(current_goal.goal_id) if current_goal is not None else None
        )
        decision = plan(
            snapshot,
            snapshot.captured_at,
            current_goal=current_goal,
            latest_effect=latest_effect,
        )
        renewal_margin = float(os.getenv("GOAL_RENEWAL_MARGIN_SECONDS", "1"))
        if (
            current_goal is not None
            and current_goal.status == "active"
            and current_goal.goal_type == decision.goal_type
            and current_goal.priority == decision.priority
            and current_goal.parameters == decision.parameters
            and (current_goal.valid_until - snapshot.captured_at).total_seconds()
            > renewal_margin
        ):
            return
        predecessor = str(current_goal.goal_id) if current_goal is not None else "none"
        effect_revision = latest_effect.revision if latest_effect is not None else 0
        client.publish_goal(
            Goal(
                idempotency_key=(
                    f"{planner_id}:generation:{generation}:predecessor:{predecessor}:"
                    f"effect:{effect_revision}:state:{snapshot.state_revision}:"
                    f"stage:{decision.goal_type}"
                ),
                planner_id=planner_id,
                instance_id=deployment,
                deployment_generation=generation,
                based_on_revision=snapshot.state_revision,
                goal_type=decision.goal_type,
                priority=decision.priority,
                created_at=snapshot.captured_at,
                valid_until=snapshot.captured_at
                + timedelta(seconds=float(os.getenv("GOAL_TTL_SECONDS", "5"))),
                parameters=decision.parameters,
                rationale=decision.rationale,
            )
        )

    try:
        run_loop(step, float(os.getenv("INTERVAL_SECONDS", "0.5")))
    finally:
        client.close()


if __name__ == "__main__":
    main()
