from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from robotkit.contracts import Effect, EffectCompletion, Goal, Observation, Preconditions
from robotkit.world_state.store import ConflictError, WorldStateStore
from tests.conftest import NOW


FAR_FUTURE = datetime(2099, 1, 1, tzinfo=timezone.utc)


def observation(
    *, key: str, at=NOW, payload=None, instance="blue", generation: int = 0
) -> Observation:
    return Observation(
        idempotency_key=key,
        producer_id="yolo",
        instance_id=instance,
        deployment_generation=generation,
        stream="vision.obstacles",
        observation_type="objects.v1",
        observed_at=at,
        ttl_seconds=10,
        payload=payload or {"count": 1},
    )


def test_observation_is_idempotent_and_black_box_is_append_only(tmp_path):
    store = WorldStateStore(tmp_path / "world.sqlite3")
    first = store.publish_observation(observation(key="frame-1"))
    duplicate = store.publish_observation(observation(key="frame-1", instance="green"))

    assert first.duplicate is False
    assert duplicate == first.model_copy(update={"duplicate": True})
    assert store.snapshot().observations[0].payload == {"count": 1}
    assert [event.category for event in store.events()] == ["observation.published"]


def test_late_observation_is_audited_but_does_not_regress_latest_state(tmp_path):
    store = WorldStateStore(tmp_path / "world.sqlite3")
    store.publish_observation(observation(key="new", at=NOW, payload={"count": 2}))
    store.publish_observation(
        observation(key="old", at=NOW - timedelta(seconds=1), payload={"count": 1})
    )

    snapshot = store.snapshot()
    assert snapshot.observations[0].payload == {"count": 2}
    assert snapshot.state_revision == 1
    assert len(store.events()) == 2


def test_higher_generation_fences_old_b_deployment(tmp_path):
    store = WorldStateStore(tmp_path / "world.sqlite3")
    store.publish_observation(
        observation(key="green", generation=2, payload={"deployment": "green"})
    )
    store.publish_observation(
        observation(
            key="blue-later",
            at=NOW + timedelta(seconds=10),
            generation=1,
            payload={"deployment": "blue"},
        )
    )
    assert store.snapshot().observations[0].payload == {"deployment": "green"}
    assert len(store.events()) == 2


def test_database_survives_process_reopen(tmp_path):
    path = tmp_path / "world.sqlite3"
    first = WorldStateStore(path)
    first.publish_observation(observation(key="frame-1"))
    first.close()

    reopened = WorldStateStore(path)
    assert reopened.snapshot().observations[0].producer_id == "yolo"
    assert len(reopened.events()) == 1


def test_goal_effect_claim_and_ack_are_durable(tmp_path):
    store = WorldStateStore(tmp_path / "world.sqlite3")
    goal = Goal(
        idempotency_key="state-1",
        planner_id="planner",
        instance_id="planner-blue",
        based_on_revision=0,
        goal_type="hold_position",
        created_at=NOW,
        valid_until=FAR_FUTURE,
    )
    store.publish_goal(goal)
    assert store.current_goal().goal_id == goal.goal_id

    effect = Effect(
        idempotency_key="goal-1-state-1",
        controller_id="controller",
        instance_id="controller-blue",
        goal_id=goal.goal_id,
        based_on_revision=0,
        effect_type="cmd_vel",
        created_at=NOW,
        valid_until=FAR_FUTURE,
        parameters={"linear_x_mps": 0, "angular_z_rps": 0},
        preconditions=Preconditions(),
    )
    store.publish_effect(effect)
    claimed = store.claim_effect("executor-1")
    assert claimed.effect_id == effect.effect_id
    assert store.latest_effect(goal.goal_id).status == "claimed"
    assert store.claim_effect("executor-2") is None

    applied = store.complete_effect(
        effect.effect_id,
        EffectCompletion(executor_id="executor-1", status="applied", result={"ok": True}),
    )
    assert applied.status == "applied"
    assert applied.result == {"ok": True}
    assert store.latest_effect(goal.goal_id).status == "applied"
    assert [item.category for item in store.events()] == [
        "goal.published",
        "effect.published",
        "effect.claimed",
        "effect.applied",
    ]


def test_only_claiming_executor_can_complete_effect(tmp_path):
    store = WorldStateStore(tmp_path / "world.sqlite3")
    goal = Goal(
        idempotency_key="g",
        planner_id="p",
        instance_id="p1",
        based_on_revision=0,
        goal_type="hold_position",
        created_at=NOW,
        valid_until=FAR_FUTURE,
    )
    store.publish_goal(goal)
    effect = Effect(
        idempotency_key="e",
        controller_id="c",
        instance_id="c1",
        goal_id=goal.goal_id,
        based_on_revision=0,
        effect_type="cmd_vel",
        created_at=NOW,
        valid_until=FAR_FUTURE,
        parameters={},
    )
    store.publish_effect(effect)
    store.claim_effect("executor-1")

    with pytest.raises(ConflictError):
        store.complete_effect(
            effect.effect_id,
            EffectCompletion(executor_id="executor-2", status="applied"),
        )


def test_higher_planner_generation_fences_old_planner(tmp_path):
    store = WorldStateStore(tmp_path / "world.sqlite3")
    common = dict(
        planner_id="planner",
        based_on_revision=0,
        created_at=NOW,
        valid_until=FAR_FUTURE,
    )
    green = Goal(
        **common,
        idempotency_key="green",
        instance_id="green",
        deployment_generation=2,
        goal_type="patrol_forward",
    )
    old_blue = Goal(
        **common,
        idempotency_key="blue",
        instance_id="blue",
        deployment_generation=1,
        goal_type="hold_position",
    )
    store.publish_goal(green)
    store.publish_goal(old_blue)
    assert store.current_goal().goal_id == green.goal_id
    assert len(store.events()) == 2


def test_higher_controller_generation_rejects_queued_old_effects(tmp_path):
    store = WorldStateStore(tmp_path / "world.sqlite3")
    goal = Goal(
        idempotency_key="goal",
        planner_id="planner",
        instance_id="planner",
        based_on_revision=0,
        goal_type="hold_position",
        created_at=NOW,
        valid_until=FAR_FUTURE,
    )
    store.publish_goal(goal)

    def effect(key: str, generation: int) -> Effect:
        return Effect(
            idempotency_key=key,
            controller_id="controller",
            instance_id=key,
            deployment_generation=generation,
            goal_id=goal.goal_id,
            based_on_revision=0,
            effect_type="cmd_vel",
            created_at=NOW,
            valid_until=FAR_FUTURE,
            parameters={"linear_x_mps": 0, "angular_z_rps": 0},
        )

    blue = effect("blue", 1)
    green = effect("green", 2)
    store.publish_effect(blue)
    store.publish_effect(green)

    assert store.claim_effect("executor").effect_id == green.effect_id
    categories = [item.category for item in store.events()]
    assert categories == [
        "goal.published",
        "effect.published",
        "effect.rejected",
        "effect.published",
        "effect.claimed",
    ]
