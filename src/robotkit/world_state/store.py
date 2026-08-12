"""SQLite event store and current-state projections."""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator
from uuid import UUID

from robotkit.contracts import (
    Effect,
    EffectCompletion,
    EffectRecord,
    EffectStatus,
    EventRecord,
    Goal,
    GoalRecord,
    Observation,
    ObservationRecord,
    PublishResult,
    WorldSnapshot,
    utc_now,
)


def _dump(model: object) -> str:
    return model.model_dump_json()  # type: ignore[attr-defined]


class ConflictError(RuntimeError):
    pass


class NotFoundError(RuntimeError):
    pass


class WorldStateStore:
    """A small durable log optimized for a single always-on edge node."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._migrate()

    def close(self) -> None:
        self._connection.close()

    def _migrate(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value INTEGER NOT NULL
            );
            INSERT OR IGNORE INTO metadata(key, value) VALUES ('revision', 0);

            CREATE TABLE IF NOT EXISTS events (
                revision INTEGER PRIMARY KEY,
                category TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                recorded_at TEXT NOT NULL,
                data TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS observations (
                event_id TEXT PRIMARY KEY,
                producer_id TEXT NOT NULL,
                stream TEXT NOT NULL,
                idempotency_key TEXT NOT NULL,
                revision INTEGER NOT NULL UNIQUE,
                record TEXT NOT NULL,
                UNIQUE(producer_id, stream, idempotency_key)
            );
            CREATE TABLE IF NOT EXISTS current_observations (
                producer_id TEXT NOT NULL,
                stream TEXT NOT NULL,
                event_id TEXT NOT NULL REFERENCES observations(event_id),
                PRIMARY KEY(producer_id, stream)
            );

            CREATE TABLE IF NOT EXISTS goals (
                goal_id TEXT PRIMARY KEY,
                planner_id TEXT NOT NULL,
                idempotency_key TEXT NOT NULL,
                revision INTEGER NOT NULL UNIQUE,
                status TEXT NOT NULL,
                record TEXT NOT NULL,
                UNIQUE(planner_id, idempotency_key)
            );
            CREATE TABLE IF NOT EXISTS current_goal (
                singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                goal_id TEXT NOT NULL REFERENCES goals(goal_id)
            );

            CREATE TABLE IF NOT EXISTS effects (
                effect_id TEXT PRIMARY KEY,
                controller_id TEXT NOT NULL,
                deployment_generation INTEGER NOT NULL,
                idempotency_key TEXT NOT NULL,
                revision INTEGER NOT NULL UNIQUE,
                status TEXT NOT NULL,
                claimed_by TEXT,
                result TEXT,
                record TEXT NOT NULL,
                UNIQUE(controller_id, idempotency_key)
            );
            CREATE INDEX IF NOT EXISTS effects_pending
            ON effects(status, revision);
            """
        )
        effect_columns = {
            row["name"]
            for row in self._connection.execute("PRAGMA table_info(effects)").fetchall()
        }
        if "deployment_generation" not in effect_columns:
            self._connection.execute(
                "ALTER TABLE effects ADD COLUMN deployment_generation INTEGER NOT NULL DEFAULT 0"
            )
        self._connection.commit()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                yield self._connection
            except Exception:
                self._connection.rollback()
                raise
            else:
                self._connection.commit()

    @staticmethod
    def _next_revision(connection: sqlite3.Connection) -> int:
        connection.execute("UPDATE metadata SET value = value + 1 WHERE key = 'revision'")
        return int(
            connection.execute("SELECT value FROM metadata WHERE key = 'revision'").fetchone()[0]
        )

    @staticmethod
    def _event(
        connection: sqlite3.Connection,
        revision: int,
        category: str,
        entity_id: str,
        data: str,
    ) -> None:
        connection.execute(
            "INSERT INTO events VALUES (?, ?, ?, ?, ?)",
            (revision, category, entity_id, utc_now().isoformat(), data),
        )

    def publish_observation(self, observation: Observation) -> PublishResult:
        with self._transaction() as connection:
            duplicate = connection.execute(
                """SELECT revision FROM observations
                   WHERE event_id = ? OR
                   (producer_id = ? AND stream = ? AND idempotency_key = ?)""",
                (
                    str(observation.event_id),
                    observation.producer_id,
                    observation.stream,
                    observation.idempotency_key,
                ),
            ).fetchone()
            if duplicate:
                return PublishResult(revision=duplicate["revision"], duplicate=True)

            revision = self._next_revision(connection)
            record = ObservationRecord(
                **observation.model_dump(), revision=revision, received_at=utc_now()
            )
            encoded = _dump(record)
            connection.execute(
                "INSERT INTO observations VALUES (?, ?, ?, ?, ?, ?)",
                (
                    str(record.event_id),
                    record.producer_id,
                    record.stream,
                    record.idempotency_key,
                    revision,
                    encoded,
                ),
            )
            current = connection.execute(
                """SELECT o.record FROM current_observations c
                   JOIN observations o ON o.event_id = c.event_id
                   WHERE c.producer_id = ? AND c.stream = ?""",
                (record.producer_id, record.stream),
            ).fetchone()
            current_record = (
                ObservationRecord.model_validate_json(current["record"])
                if current is not None
                else None
            )
            if current_record is None or (
                record.deployment_generation > current_record.deployment_generation
                or (
                    record.deployment_generation == current_record.deployment_generation
                    and current_record.observed_at <= record.observed_at
                )
            ):
                connection.execute(
                    """INSERT INTO current_observations(producer_id, stream, event_id)
                       VALUES (?, ?, ?)
                       ON CONFLICT(producer_id, stream)
                       DO UPDATE SET event_id = excluded.event_id""",
                    (record.producer_id, record.stream, str(record.event_id)),
                )
            self._event(connection, revision, "observation.published", str(record.event_id), encoded)
            return PublishResult(revision=revision, duplicate=False)

    def snapshot(self) -> WorldSnapshot:
        with self._lock:
            revision = int(
                self._connection.execute(
                    "SELECT value FROM metadata WHERE key = 'revision'"
                ).fetchone()[0]
            )
            rows = self._connection.execute(
                """SELECT o.record FROM current_observations c
                   JOIN observations o ON o.event_id = c.event_id
                   ORDER BY c.producer_id, c.stream"""
            ).fetchall()
            state_revision = max(
                (ObservationRecord.model_validate_json(row["record"]).revision for row in rows),
                default=0,
            )
        return WorldSnapshot(
            revision=revision,
            state_revision=state_revision,
            captured_at=utc_now(),
            observations=[ObservationRecord.model_validate_json(row["record"]) for row in rows],
        )

    def events(self, after: int = 0, limit: int = 100) -> list[EventRecord]:
        with self._lock:
            rows = self._connection.execute(
                """SELECT revision, category, entity_id, recorded_at, data
                   FROM events WHERE revision > ? ORDER BY revision LIMIT ?""",
                (after, limit),
            ).fetchall()
        return [
            EventRecord(
                revision=row["revision"],
                category=row["category"],
                entity_id=row["entity_id"],
                recorded_at=datetime.fromisoformat(row["recorded_at"]),
                data=json.loads(row["data"]),
            )
            for row in rows
        ]

    def publish_goal(self, goal: Goal) -> PublishResult:
        with self._transaction() as connection:
            duplicate = connection.execute(
                "SELECT revision FROM goals WHERE goal_id = ? OR (planner_id = ? AND idempotency_key = ?)",
                (str(goal.goal_id), goal.planner_id, goal.idempotency_key),
            ).fetchone()
            if duplicate:
                return PublishResult(revision=duplicate["revision"], duplicate=True)

            current = connection.execute(
                """SELECT g.goal_id, g.record FROM current_goal c
                   JOIN goals g ON g.goal_id = c.goal_id WHERE c.singleton = 1"""
            ).fetchone()
            should_activate = True
            if current:
                current_record = GoalRecord.model_validate_json(current["record"])
                should_activate = goal.deployment_generation >= current_record.deployment_generation
            if current and should_activate:
                connection.execute(
                    "UPDATE goals SET status = 'superseded' WHERE goal_id = ? AND status = 'active'",
                    (current["goal_id"],),
                )
            revision = self._next_revision(connection)
            goal_status = "active" if should_activate else "superseded"
            record = GoalRecord(**goal.model_dump(), revision=revision, status=goal_status)
            encoded = _dump(record)
            connection.execute(
                "INSERT INTO goals VALUES (?, ?, ?, ?, ?, ?)",
                (
                    str(goal.goal_id),
                    goal.planner_id,
                    goal.idempotency_key,
                    revision,
                    goal_status,
                    encoded,
                ),
            )
            if should_activate:
                connection.execute(
                    """INSERT INTO current_goal(singleton, goal_id) VALUES (1, ?)
                       ON CONFLICT(singleton) DO UPDATE SET goal_id = excluded.goal_id""",
                    (str(goal.goal_id),),
                )
            self._event(connection, revision, "goal.published", str(goal.goal_id), encoded)
            return PublishResult(revision=revision, duplicate=False)

    def current_goal(self) -> GoalRecord | None:
        with self._lock:
            row = self._connection.execute(
                """SELECT g.record, g.status FROM current_goal c
                   JOIN goals g ON g.goal_id = c.goal_id WHERE c.singleton = 1"""
            ).fetchone()
        if not row:
            return None
        record = GoalRecord.model_validate_json(row["record"])
        status = row["status"]
        if record.valid_until < utc_now() and status == "active":
            status = "expired"
        return record.model_copy(update={"status": status})

    def publish_effect(self, effect: Effect) -> PublishResult:
        with self._transaction() as connection:
            duplicate = connection.execute(
                """SELECT revision FROM effects WHERE effect_id = ? OR
                   (controller_id = ? AND idempotency_key = ?)""",
                (str(effect.effect_id), effect.controller_id, effect.idempotency_key),
            ).fetchone()
            if duplicate:
                return PublishResult(revision=duplicate["revision"], duplicate=True)
            current_generation = connection.execute(
                "SELECT MAX(deployment_generation) FROM effects WHERE controller_id = ?",
                (effect.controller_id,),
            ).fetchone()[0]
            should_enqueue = (
                current_generation is None
                or effect.deployment_generation >= int(current_generation)
            )
            if current_generation is not None and effect.deployment_generation > int(
                current_generation
            ):
                older = connection.execute(
                    """SELECT * FROM effects
                       WHERE controller_id = ? AND deployment_generation < ?
                       AND status = 'pending'""",
                    (effect.controller_id, effect.deployment_generation),
                ).fetchall()
                for old in older:
                    connection.execute(
                        "UPDATE effects SET status = 'rejected' WHERE effect_id = ?",
                        (old["effect_id"],),
                    )
                    transition_revision = self._next_revision(connection)
                    rejected = EffectRecord.model_validate_json(old["record"]).model_copy(
                        update={
                            "revision": transition_revision,
                            "status": EffectStatus.REJECTED,
                            "result": {"reason": "superseded deployment generation"},
                        }
                    )
                    self._event(
                        connection,
                        transition_revision,
                        "effect.rejected",
                        old["effect_id"],
                        _dump(rejected),
                    )
            revision = self._next_revision(connection)
            record = EffectRecord(
                **effect.model_dump(),
                revision=revision,
                status=(EffectStatus.PENDING if should_enqueue else EffectStatus.REJECTED),
                result=(None if should_enqueue else {"reason": "stale deployment generation"}),
            )
            encoded = _dump(record)
            connection.execute(
                """INSERT INTO effects(
                       effect_id, controller_id, deployment_generation, idempotency_key,
                       revision, status, claimed_by, result, record
                   ) VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?)""",
                (
                    str(effect.effect_id),
                    effect.controller_id,
                    effect.deployment_generation,
                    effect.idempotency_key,
                    revision,
                    record.status.value,
                    json.dumps(record.result) if record.result else None,
                    encoded,
                ),
            )
            category = "effect.published" if should_enqueue else "effect.rejected"
            self._event(connection, revision, category, str(effect.effect_id), encoded)
            return PublishResult(revision=revision, duplicate=False)

    def latest_effect(self, goal_id: UUID | None = None) -> EffectRecord | None:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM effects ORDER BY revision DESC"
            ).fetchall()
        selected = None
        for row in rows:
            candidate = EffectRecord.model_validate_json(row["record"])
            if goal_id is None or candidate.goal_id == goal_id:
                selected = (row, candidate)
                break
        if selected is None:
            return None
        row, record = selected
        result = json.loads(row["result"]) if row["result"] is not None else None
        return record.model_copy(
            update={
                "status": EffectStatus(row["status"]),
                "claimed_by": row["claimed_by"],
                "result": result,
            }
        )

    def claim_effect(self, executor_id: str) -> EffectRecord | None:
        now = utc_now()
        with self._transaction() as connection:
            rows = connection.execute(
                "SELECT * FROM effects WHERE status = 'pending' ORDER BY revision"
            ).fetchall()
            selected = None
            for row in rows:
                effect = EffectRecord.model_validate_json(row["record"])
                if effect.valid_until <= now:
                    connection.execute(
                        "UPDATE effects SET status = 'expired' WHERE effect_id = ?",
                        (row["effect_id"],),
                    )
                    revision = self._next_revision(connection)
                    expired = effect.model_copy(
                        update={"status": EffectStatus.EXPIRED, "revision": revision}
                    )
                    self._event(
                        connection,
                        revision,
                        "effect.expired",
                        row["effect_id"],
                        _dump(expired),
                    )
                    continue
                selected = (row, effect)
                break
            if selected is None:
                return None

            row, effect = selected
            connection.execute(
                "UPDATE effects SET status = 'claimed', claimed_by = ? WHERE effect_id = ?",
                (executor_id, row["effect_id"]),
            )
            revision = self._next_revision(connection)
            claimed = effect.model_copy(
                update={
                    "status": EffectStatus.CLAIMED,
                    "claimed_by": executor_id,
                    "revision": revision,
                }
            )
            self._event(
                connection, revision, "effect.claimed", row["effect_id"], _dump(claimed)
            )
            return claimed

    def complete_effect(self, effect_id: UUID, completion: EffectCompletion) -> EffectRecord:
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM effects WHERE effect_id = ?", (str(effect_id),)
            ).fetchone()
            if not row:
                raise NotFoundError(str(effect_id))
            if row["status"] != "claimed" or row["claimed_by"] != completion.executor_id:
                raise ConflictError("effect is not claimed by this executor")

            connection.execute(
                "UPDATE effects SET status = ?, result = ? WHERE effect_id = ?",
                (completion.status, json.dumps(completion.result), str(effect_id)),
            )
            revision = self._next_revision(connection)
            effect = EffectRecord.model_validate_json(row["record"]).model_copy(
                update={
                    "revision": revision,
                    "status": EffectStatus(completion.status),
                    "claimed_by": completion.executor_id,
                    "result": completion.result,
                }
            )
            self._event(
                connection,
                revision,
                f"effect.{completion.status}",
                str(effect_id),
                _dump(effect),
            )
            return effect
