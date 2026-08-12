from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import UUID

import pytest

from robotkit.contracts import (
    Effect,
    EffectRecord,
    Goal,
    GoalRecord,
    Observation,
    ObservationRecord,
    Preconditions,
    WorldSnapshot,
)


NOW = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def observation_factory():
    def make(
        *,
        stream: str = "localization.pose",
        payload: dict | None = None,
        observed_at: datetime = NOW,
        ttl_seconds: float = 5,
        revision: int = 1,
        producer_id: str = "test-perception",
        key: str = "sample-1",
    ) -> ObservationRecord:
        observation = Observation(
            idempotency_key=key,
            producer_id=producer_id,
            instance_id="blue-abc",
            stream=stream,
            observation_type=stream,
            observed_at=observed_at,
            confidence=0.9,
            ttl_seconds=ttl_seconds,
            payload=payload or {},
        )
        return ObservationRecord(
            **observation.model_dump(), revision=revision, received_at=NOW
        )

    return make


@pytest.fixture
def snapshot_factory(observation_factory):
    def make(observations=None, *, state_revision: int | None = None):
        observations = observations or []
        derived_revision = max((item.revision for item in observations), default=0)
        state_revision = derived_revision if state_revision is None else state_revision
        return WorldSnapshot(
            revision=state_revision,
            state_revision=state_revision,
            captured_at=NOW,
            observations=observations,
        )

    return make


@pytest.fixture
def goal_factory():
    def make(goal_type: str = "hold_position", parameters: dict | None = None):
        goal = Goal(
            idempotency_key="goal-1",
            planner_id="planner-v1",
            instance_id="green-def",
            based_on_revision=2,
            goal_type=goal_type,
            created_at=NOW,
            valid_until=NOW + timedelta(seconds=5),
            parameters=parameters or {},
        )
        return GoalRecord(**goal.model_dump(), revision=3, status="active")

    return make


@pytest.fixture
def effect_factory(goal_factory):
    def make(
        *,
        linear: float = 0.2,
        angular: float = 0.0,
        based_on_revision: int = 0,
        required: list[str] | None = None,
    ):
        goal = goal_factory()
        effect = Effect(
            idempotency_key="effect-1",
            controller_id="controller-v1",
            instance_id="controller-green",
            goal_id=goal.goal_id,
            based_on_revision=based_on_revision,
            effect_type="cmd_vel",
            created_at=NOW,
            valid_until=NOW + timedelta(seconds=1),
            parameters={"linear_x_mps": linear, "angular_z_rps": angular},
            preconditions=Preconditions(
                max_world_revision_drift=3,
                require_fresh_streams=required or [],
            ),
        )
        return EffectRecord(**effect.model_dump(), revision=4, status="claimed")

    return make
