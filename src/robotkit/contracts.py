"""Versioned wire contracts shared by every RobotKit container."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Observation(Contract):
    """One immutable interpretation emitted by a B component."""

    schema_version: Literal["1"] = "1"
    event_id: UUID = Field(default_factory=uuid4)
    idempotency_key: str = Field(min_length=1, max_length=200)
    producer_id: str = Field(min_length=1, max_length=100)
    instance_id: str = Field(min_length=1, max_length=200)
    deployment_generation: int = Field(default=0, ge=0)
    stream: str = Field(min_length=1, max_length=100)
    observation_type: str = Field(min_length=1, max_length=100)
    observed_at: datetime = Field(default_factory=utc_now)
    frame_id: str | None = Field(default=None, max_length=100)
    confidence: float | None = Field(default=None, ge=0, le=1)
    ttl_seconds: float = Field(default=5.0, gt=0, le=86_400)
    payload: dict[str, Any]

    @field_validator("observed_at")
    @classmethod
    def timestamp_must_have_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("observed_at must include a timezone")
        return value


class ObservationRecord(Observation):
    revision: int = Field(ge=1)
    received_at: datetime

    def is_stale(self, at: datetime | None = None) -> bool:
        at = at or utc_now()
        return (at - self.observed_at).total_seconds() > self.ttl_seconds


class WorldSnapshot(Contract):
    schema_version: Literal["1"] = "1"
    revision: int = Field(ge=0)
    state_revision: int = Field(ge=0)
    captured_at: datetime
    observations: list[ObservationRecord]


class Goal(Contract):
    schema_version: Literal["1"] = "1"
    goal_id: UUID = Field(default_factory=uuid4)
    idempotency_key: str = Field(min_length=1, max_length=200)
    planner_id: str = Field(min_length=1, max_length=100)
    instance_id: str = Field(min_length=1, max_length=200)
    deployment_generation: int = Field(default=0, ge=0)
    based_on_revision: int = Field(ge=0)
    goal_type: str = Field(min_length=1, max_length=100)
    priority: int = Field(default=0, ge=0, le=100)
    created_at: datetime = Field(default_factory=utc_now)
    valid_until: datetime
    parameters: dict[str, Any] = Field(default_factory=dict)
    rationale: str = Field(default="", max_length=2000)

    @field_validator("created_at", "valid_until")
    @classmethod
    def goal_timestamp_must_have_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamps must include a timezone")
        return value


class GoalRecord(Goal):
    revision: int = Field(ge=1)
    status: Literal["active", "superseded", "expired"] = "active"


class Preconditions(Contract):
    """Conditions checked immediately before an effect reaches ROS2."""

    max_world_revision_drift: int = Field(default=10, ge=0)
    require_fresh_streams: list[str] = Field(default_factory=list)


class Effect(Contract):
    schema_version: Literal["1"] = "1"
    effect_id: UUID = Field(default_factory=uuid4)
    idempotency_key: str = Field(min_length=1, max_length=200)
    controller_id: str = Field(min_length=1, max_length=100)
    instance_id: str = Field(min_length=1, max_length=200)
    deployment_generation: int = Field(default=0, ge=0)
    goal_id: UUID
    based_on_revision: int = Field(ge=0)
    effect_type: str = Field(min_length=1, max_length=100)
    created_at: datetime = Field(default_factory=utc_now)
    valid_until: datetime
    parameters: dict[str, Any] = Field(default_factory=dict)
    preconditions: Preconditions = Field(default_factory=Preconditions)

    @field_validator("created_at", "valid_until")
    @classmethod
    def effect_timestamp_must_have_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamps must include a timezone")
        return value


class EffectStatus(str, Enum):
    PENDING = "pending"
    CLAIMED = "claimed"
    APPLIED = "applied"
    REJECTED = "rejected"
    FAILED = "failed"
    EXPIRED = "expired"


class EffectRecord(Effect):
    revision: int = Field(ge=1)
    status: EffectStatus = EffectStatus.PENDING
    claimed_by: str | None = None
    result: dict[str, Any] | None = None


class EffectCompletion(Contract):
    executor_id: str = Field(min_length=1, max_length=200)
    status: Literal["applied", "rejected", "failed"]
    result: dict[str, Any] = Field(default_factory=dict)


class EventRecord(Contract):
    revision: int = Field(ge=1)
    category: str
    entity_id: str
    recorded_at: datetime
    data: dict[str, Any]


class PublishResult(Contract):
    revision: int
    duplicate: bool
