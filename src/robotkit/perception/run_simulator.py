"""Container entrypoint for the simulator B component."""

from __future__ import annotations

import os
from datetime import datetime, timezone

from robotkit.client import WorldStateClient
from robotkit.contracts import Observation
from robotkit.perception.simulator import interpret_world
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
    producer_id = os.getenv("ROBOTKIT_PRODUCER_ID", "go2-world-simulator")
    deployment = instance_id()
    generation = deployment_generation()

    def step() -> None:
        now = datetime.now(timezone.utc)
        tick = int(now.timestamp() * 2)
        for item in interpret_world(now):
            client.publish_observation(
                Observation(
                    idempotency_key=f"{deployment}:{item.stream}:{tick}",
                    producer_id=producer_id,
                    instance_id=deployment,
                    deployment_generation=generation,
                    stream=item.stream,
                    observation_type=item.observation_type,
                    observed_at=now,
                    frame_id=item.frame_id,
                    confidence=item.confidence,
                    ttl_seconds=3,
                    payload=item.payload,
                )
            )

    try:
        run_loop(step, float(os.getenv("INTERVAL_SECONDS", "0.5")))
    finally:
        client.close()


if __name__ == "__main__":
    main()
