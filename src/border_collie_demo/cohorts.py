"""Durable controller for bounded cohorts of independent Demo Runs."""

from __future__ import annotations

import asyncio
import json
import os
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .cohort_policy import (
    CohortPolicy,
    choose_fruit_sequence,
    decide_terminal_run,
    evaluate_home_clearance,
)
from .stage_demo import FruitMission, HomeScope, StageDemo


class CohortConflict(RuntimeError):
    """A new cohort cannot start while another cohort owns activation."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class CohortController:
    """Start, observe, and stop one durable cohort without bypassing StageDemo."""

    def __init__(self, demo: StageDemo, root: Path) -> None:
        self._demo = demo
        self._root = root
        self._lock = asyncio.Lock()
        self._task: asyncio.Task[None] | None = None
        self._current: dict[str, Any] | None = self._load_latest()
        self._stop_requested = False
        if self._current is not None and self._current.get("status") == "RUNNING":
            self._current["status"] = "INTERRUPTED"
            self._current["abort_reason"] = (
                "application restarted; cohorts are never resumed automatically"
            )
            self._current["ended_at_utc"] = _utc_now()
            self._persist()

    async def start(self, policy: CohortPolicy, qualified_fruits: list[str]) -> dict[str, Any]:
        sequence = choose_fruit_sequence(policy, qualified_fruits)
        async with self._lock:
            if self._task is not None and not self._task.done():
                raise CohortConflict("a cohort is already running")
            status = self._demo.status()
            if status.get("active_run_id") is not None:
                raise CohortConflict("a Demo Run is already active")
            if (status.get("mission") or {}).get("restart_required"):
                raise CohortConflict("restart required before another cohort")
            if not (status.get("activation") or {}).get("ready"):
                raise CohortConflict("activation readiness has not passed")
            cohort_id = str(uuid4())
            home_scope = HomeScope(f"cohort:{cohort_id}", "cohort")
            self._current = {
                "schema_version": 1,
                "cohort_id": cohort_id,
                "status": "RUNNING",
                "started_at_utc": _utc_now(),
                "ended_at_utc": None,
                "policy": policy.to_dict(),
                "fruit_sequence": sequence,
                "runs": [],
                "current_run_id": None,
                "stop_requested": False,
                "abort_reason": None,
                "home_scope": home_scope.to_dict(),
                "home": None,
            }
            self._stop_requested = False
            self._persist()
            self._task = asyncio.create_task(
                self._execute(policy), name=f"fruit-cohort-{cohort_id}"
            )
            return deepcopy(self._current)

    async def stop(self) -> dict[str, Any] | None:
        async with self._lock:
            current = self._current
            task = self._task
            if current is None or task is None or task.done():
                return None if current is None else deepcopy(current)
            self._stop_requested = True
            current["stop_requested"] = True
            self._persist()
        if self._demo.status().get("active_run_id") is not None:
            await self._demo.stop()
        await asyncio.gather(task, return_exceptions=True)
        return self.current()

    async def close(self) -> None:
        await self.stop()

    def current(self) -> dict[str, Any] | None:
        return None if self._current is None else deepcopy(self._current)

    def running(self) -> bool:
        return (
            self._current is not None
            and self._current.get("status") == "RUNNING"
            and self._task is not None
            and not self._task.done()
        )

    def get(self, cohort_id: str) -> dict[str, Any] | None:
        if not cohort_id or "/" in cohort_id or "\\" in cohort_id:
            return None
        path = self._root / f"{cohort_id}.json"
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None

    async def _execute(self, policy: CohortPolicy) -> None:
        assert self._current is not None
        cohort = self._current
        home_scope = HomeScope(**cohort["home_scope"])
        try:
            for number, fruit in enumerate(cohort["fruit_sequence"], start=1):
                if self._stop_requested:
                    self._finish("STOPPED", "operator stopped the cohort")
                    return
                activation_id = f"cohort:{cohort['cohort_id']}:run:{number}"
                try:
                    activation = await self._demo.activate(
                        FruitMission(
                            fruit,
                            "cohort",
                            activation_id,
                            home_scope=home_scope,
                        )
                    )
                except Exception as exc:  # noqa: BLE001 - activation is ambiguous
                    self._finish(
                        "ABORTED",
                        f"run {number} activation failed; no retry attempted: {exc}",
                    )
                    return
                if activation.idempotent_replay:
                    self._finish(
                        "ABORTED",
                        f"run {number} activation replayed; no automatic retry permitted",
                    )
                    return
                run_id = activation.run["run_id"]
                run_home = activation.run.get("home")
                if not isinstance(run_home, dict):
                    self._finish("ABORTED", f"run {number} has no stored cohort Home")
                    return
                if cohort["home"] is None:
                    cohort["home"] = deepcopy(run_home)
                elif cohort["home"] != run_home:
                    self._finish("ABORTED", f"run {number} changed cohort Home")
                    return
                cohort["current_run_id"] = run_id
                self._persist()
                try:
                    run = await self._demo.wait(run_id)
                except asyncio.CancelledError:
                    run = self._demo.result(run_id)
                    if self._stop_requested:
                        for _ in range(100):
                            if run.get("outcome") is not None:
                                break
                            await asyncio.sleep(0.01)
                            run = self._demo.result(run_id)
                if run.get("outcome") is None:
                    self._finish("ABORTED", f"run {number} did not become terminal")
                    return

                decision = decide_terminal_run(run, policy)
                run_tuning = run.get("run_tuning")
                search_tuning = (
                    run_tuning.get("search") if isinstance(run_tuning, dict) else None
                )
                stage_results = run.get("stage_results")
                turn_evidence = (
                    stage_results.get("turn_to_fruit")
                    if isinstance(stage_results, dict)
                    else None
                )
                record = {
                    "number": number,
                    "target_fruit": fruit,
                    "activation_id": activation_id,
                    "run_id": run_id,
                    "outcome": run.get("outcome"),
                    "reason": run.get("reason"),
                    "failed_phase": run.get("failed_phase"),
                    "final_safety_state": run.get("final_safety_state"),
                    "home_scope": run.get("home_scope"),
                    "home": run.get("home"),
                    "cohort_decision": decision,
                    "bearing_routing": {
                        "enabled": (
                            search_tuning.get("bearing_routing_enabled")
                            if isinstance(search_tuning, dict)
                            else None
                        ),
                        "used": (
                            turn_evidence.get("bearing_route_used")
                            if isinstance(turn_evidence, dict)
                            else None
                        ),
                        "fallback_reason": (
                            turn_evidence.get("bearing_route_fallback_reason")
                            if isinstance(turn_evidence, dict)
                            else None
                        ),
                        "planning_ms": (
                            turn_evidence.get("bearing_route_planning_ms")
                            if isinstance(turn_evidence, dict)
                            else None
                        ),
                    },
                }
                cohort["runs"].append(record)
                cohort["current_run_id"] = None
                self._persist()

                if self._stop_requested:
                    self._finish("STOPPED", "operator stopped the cohort")
                    return
                if decision["action"] == "STOP_COHORT":
                    self._finish("ABORTED", decision["detail"])
                    return

                clearance = evaluate_home_clearance(self._demo.status(), run_id)
                record["inter_run_clearance"] = clearance
                if not clearance["safe_to_continue"]:
                    decision["action"] = "STOP_COHORT"
                    decision["clearance_code"] = "HOME_CLEARANCE_BLOCKED"
                    decision["detail"] += "; fresh exact-run Home clearance failed"
                    self._persist()
                    self._finish("ABORTED", decision["detail"])
                    return
                decision["action"] = (
                    "CONTINUE"
                    if number < len(cohort["fruit_sequence"])
                    else "COHORT_COMPLETE"
                )
                decision["clearance_code"] = (
                    "HOME_CLEARANCE_PASSED"
                    if number < len(cohort["fruit_sequence"])
                    else "TARGET_RUN_COUNT_REACHED"
                )
                decision["detail"] += "; fresh exact-run Home clearance passed"
                self._persist()
            self._finish("COMPLETED", None)
        except Exception as exc:  # noqa: BLE001 - persist unexpected controller failure
            self._finish("ABORTED", f"cohort controller error: {exc}")

    def _finish(self, status: str, reason: str | None) -> None:
        assert self._current is not None
        self._current["status"] = status
        self._current["abort_reason"] = reason
        self._current["current_run_id"] = None
        self._current["ended_at_utc"] = _utc_now()
        self._persist()

    def _persist(self) -> None:
        if self._current is None:
            return
        self._root.mkdir(parents=True, exist_ok=True)
        path = self._root / f"{self._current['cohort_id']}.json"
        temporary = path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(self._current, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)

    def _load_latest(self) -> dict[str, Any] | None:
        try:
            paths = sorted(
                self._root.glob("*.json"), key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
        except OSError:
            return None
        for path in paths:
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
        return None
