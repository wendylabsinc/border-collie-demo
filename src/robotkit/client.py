"""Typed client used by disposable RobotKit workers."""

from __future__ import annotations

from uuid import UUID

import httpx

from robotkit.contracts import (
    Effect,
    EffectCompletion,
    EffectRecord,
    EventRecord,
    Goal,
    GoalRecord,
    Observation,
    PublishResult,
    WorldSnapshot,
)


class WorldStateClient:
    def __init__(self, base_url: str, *, timeout: float = 5.0) -> None:
        self._client = httpx.Client(base_url=base_url, timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def health(self) -> bool:
        return self._client.get("/healthz").is_success

    def publish_observation(self, observation: Observation) -> PublishResult:
        response = self._client.post(
            "/v1/observations", json=observation.model_dump(mode="json")
        )
        response.raise_for_status()
        return PublishResult.model_validate(response.json())

    def snapshot(self) -> WorldSnapshot:
        response = self._client.get("/v1/state")
        response.raise_for_status()
        return WorldSnapshot.model_validate(response.json())

    def events(self, *, after: int = 0, limit: int = 100) -> list[EventRecord]:
        response = self._client.get(
            "/v1/events", params={"after": after, "limit": limit}
        )
        response.raise_for_status()
        return [EventRecord.model_validate(item) for item in response.json()]

    def publish_goal(self, goal: Goal) -> PublishResult:
        response = self._client.post("/v1/goals", json=goal.model_dump(mode="json"))
        response.raise_for_status()
        return PublishResult.model_validate(response.json())

    def current_goal(self) -> GoalRecord | None:
        response = self._client.get("/v1/goals/current")
        response.raise_for_status()
        return GoalRecord.model_validate(response.json()) if response.json() else None

    def publish_effect(self, effect: Effect) -> PublishResult:
        response = self._client.post("/v1/effects", json=effect.model_dump(mode="json"))
        response.raise_for_status()
        return PublishResult.model_validate(response.json())

    def claim_effect(self, executor_id: str) -> EffectRecord | None:
        response = self._client.post(
            "/v1/effects/claim", params={"executor_id": executor_id}
        )
        response.raise_for_status()
        return EffectRecord.model_validate(response.json()) if response.json() else None

    def latest_effect(self, goal_id: UUID | None = None) -> EffectRecord | None:
        parameters = {"goal_id": str(goal_id)} if goal_id is not None else None
        response = self._client.get("/v1/effects/latest", params=parameters)
        response.raise_for_status()
        return EffectRecord.model_validate(response.json()) if response.json() else None

    def complete_effect(
        self, effect_id: UUID, completion: EffectCompletion
    ) -> EffectRecord:
        response = self._client.post(
            f"/v1/effects/{effect_id}/complete", json=completion.model_dump(mode="json")
        )
        response.raise_for_status()
        return EffectRecord.model_validate(response.json())
