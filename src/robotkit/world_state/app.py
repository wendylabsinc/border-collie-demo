"""HTTP API for container A."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator
from uuid import UUID

from fastapi import FastAPI, HTTPException, Query, Response, status

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
from robotkit.world_state.store import ConflictError, NotFoundError, WorldStateStore


def create_app(database_path: str | Path | None = None) -> FastAPI:
    path = database_path or os.getenv("ROBOTKIT_DB_PATH", "/tmp/robotkit-world-state.sqlite3")
    store = WorldStateStore(path)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        store.close()

    app = FastAPI(title="RobotKit World State", version="1.0.0", lifespan=lifespan)
    app.state.store = store

    @app.get("/healthz")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/observations", response_model=PublishResult, status_code=status.HTTP_201_CREATED)
    def publish_observation(observation: Observation, response: Response) -> PublishResult:
        result = store.publish_observation(observation)
        if result.duplicate:
            response.status_code = status.HTTP_200_OK
        return result

    @app.get("/v1/state", response_model=WorldSnapshot)
    def state() -> WorldSnapshot:
        return store.snapshot()

    @app.get("/v1/events", response_model=list[EventRecord])
    def events(
        after: int = Query(default=0, ge=0), limit: int = Query(default=100, ge=1, le=1000)
    ) -> list[EventRecord]:
        return store.events(after=after, limit=limit)

    @app.post("/v1/goals", response_model=PublishResult, status_code=status.HTTP_201_CREATED)
    def publish_goal(goal: Goal, response: Response) -> PublishResult:
        result = store.publish_goal(goal)
        if result.duplicate:
            response.status_code = status.HTTP_200_OK
        return result

    @app.get("/v1/goals/current", response_model=GoalRecord | None)
    def current_goal() -> GoalRecord | None:
        return store.current_goal()

    @app.post("/v1/effects", response_model=PublishResult, status_code=status.HTTP_201_CREATED)
    def publish_effect(effect: Effect, response: Response) -> PublishResult:
        result = store.publish_effect(effect)
        if result.duplicate:
            response.status_code = status.HTTP_200_OK
        return result

    @app.post("/v1/effects/claim", response_model=EffectRecord | None)
    def claim_effect(executor_id: str = Query(min_length=1, max_length=200)) -> EffectRecord | None:
        return store.claim_effect(executor_id)

    @app.get("/v1/effects/latest", response_model=EffectRecord | None)
    def latest_effect(goal_id: UUID | None = None) -> EffectRecord | None:
        return store.latest_effect(goal_id)

    @app.post("/v1/effects/{effect_id}/complete", response_model=EffectRecord)
    def complete_effect(effect_id: UUID, completion: EffectCompletion) -> EffectRecord:
        try:
            return store.complete_effect(effect_id, completion)
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail="effect not found") from exc
        except ConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    return app


app = create_app()
