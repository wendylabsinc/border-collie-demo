from __future__ import annotations

import json
import os
import shutil
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
from itertools import pairwise
from pathlib import Path
from time import monotonic
from typing import Any
from uuid import UUID, uuid4

from .evidence import EvidenceArtifact


class ActiveRunError(RuntimeError):
    """Raised when activation is requested while a Demo Run is active."""


class RunResultNotFound(KeyError):
    """Raised when a full Run Result identifier cannot be read."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class RunResultStore:
    """Append-only Demo Run journal with an atomically materialized result."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self._active_run_id: str | None = None
        self._active_recovery: tuple[str, str] | None = None

    @property
    def active_run_id(self) -> str | None:
        return self._active_run_id

    @property
    def active_recovery(self) -> dict[str, str] | None:
        if self._active_recovery is None:
            return None
        run_id, recovery_id = self._active_recovery
        return {"run_id": run_id, "recovery_id": recovery_id}

    def seal_interrupted_runs(self) -> list[dict[str, Any]]:
        if not self.root.exists():
            return []
        sealed: list[dict[str, Any]] = []
        for path in sorted(self.root.glob("*/result.json")):
            try:
                result = json.loads(path.read_text(encoding="utf-8"))
                run_id = str(UUID(result["run_id"]))
            except (KeyError, ValueError, json.JSONDecodeError, OSError):
                continue
            if result.get("outcome") is None:
                sealed.append(
                    self.seal(
                        run_id,
                        phase="failed",
                        outcome="FAILED",
                        reason="PROCESS_INTERRUPTED",
                        message="prior process ended before the Demo Run was sealed",
                        final_safety_state="UNKNOWN",
                    )
                )
                continue
            changed = False
            for recovery in result.get("recovery_attempts", []):
                if not isinstance(recovery, dict) or recovery.get("outcome") is not None:
                    continue
                recovery["outcome"] = "FAILED"
                recovery["reason"] = "PROCESS_INTERRUPTED"
                recovery["message"] = (
                    "prior process ended before failed-run Home recovery was sealed"
                )
                recovery["final_safety_state"] = "UNKNOWN"
                recovery["ended_at_utc"] = _utc_now()
                recovery["duration_s"] = None
                changed = True
            if changed:
                self._write_result(result)
                sealed.append(deepcopy(result))
        self._active_run_id = None
        self._active_recovery = None
        return sealed

    def start_run(
        self,
        *,
        target_fruit: str,
        activation_source: str,
        orientation_degrees: float = 0.0,
    ) -> dict[str, Any]:
        if self._active_run_id is not None:
            raise ActiveRunError("a Demo Run is already active")
        if self._active_recovery is not None:
            raise ActiveRunError("failed-run Home recovery is active")

        run_id = str(uuid4())
        started_utc = _utc_now()
        started_monotonic_s = monotonic()
        result: dict[str, Any] = {
            "schema_version": 1,
            "run_id": run_id,
            "target_fruit": target_fruit,
            "activation_source": activation_source,
            "orientation_degrees": float(orientation_degrees),
            "started_at_utc": started_utc,
            "started_monotonic_s": started_monotonic_s,
            "ended_at_utc": None,
            "duration_s": None,
            "outcome": None,
            "reason": None,
            "current_phase": "idle",
            "failed_phase": None,
            "message": "activation accepted; awaiting preflight",
            "final_safety_state": None,
            "events": [],
        }
        self.root.mkdir(parents=True, exist_ok=True)
        run_dir = self.root / run_id
        run_dir.mkdir()
        (run_dir / "snapshots").mkdir()
        self._append_event(
            result,
            phase="idle",
            reason="ACTIVATION_ACCEPTED",
            message="Demo Run persisted before preflight",
        )
        self._write_result(result)
        self._active_run_id = run_id
        return deepcopy(result)

    def start_recovery(
        self,
        run_id: str,
        *,
        confirmation: str,
    ) -> dict[str, Any]:
        if self._active_run_id is not None:
            raise ActiveRunError("a Demo Run is active")
        if self._active_recovery is not None:
            raise ActiveRunError("failed-run Home recovery is already active")
        result = self.get(run_id)
        if result.get("outcome") != "FAILED":
            raise ActiveRunError("only failed Demo Runs can be recovered")
        if result.get("recovery_attempts"):
            raise ActiveRunError("failed Demo Run already has a recovery attempt")

        recovery_id = str(uuid4())
        recovery = {
            "schema_version": 1,
            "recovery_id": recovery_id,
            "run_id": run_id,
            "confirmation": confirmation,
            "started_at_utc": _utc_now(),
            "started_monotonic_s": monotonic(),
            "ended_at_utc": None,
            "duration_s": None,
            "outcome": None,
            "reason": None,
            "message": "failed-run Home recovery accepted",
            "final_safety_state": None,
            "steps": [],
        }
        result.setdefault("recovery_attempts", []).append(recovery)
        self._write_result(result)
        self._active_recovery = (run_id, recovery_id)
        return deepcopy(recovery)

    def record_recovery_step(
        self,
        run_id: str,
        recovery_id: str,
        step: str,
        evidence: dict[str, Any],
    ) -> dict[str, Any]:
        result = self.get(run_id)
        recovery = self._find_recovery(result, recovery_id)
        if recovery.get("outcome") is not None:
            raise ActiveRunError("terminal Home recovery cannot be changed")
        recovery["steps"].append(
            {
                "sequence": len(recovery["steps"]) + 1,
                "step": step,
                "recorded_at_utc": _utc_now(),
                "seconds_since_start": monotonic()
                - float(recovery["started_monotonic_s"]),
                "evidence": deepcopy(evidence),
            }
        )
        recovery["message"] = f"{step} completed"
        self._write_result(result)
        return deepcopy(recovery)

    def seal_recovery(
        self,
        run_id: str,
        recovery_id: str,
        *,
        outcome: str,
        reason: str,
        message: str,
        final_safety_state: str,
        final_evidence: dict[str, Any],
    ) -> dict[str, Any]:
        result = self.get(run_id)
        recovery = self._find_recovery(result, recovery_id)
        if recovery.get("outcome") is not None:
            return deepcopy(recovery)
        recovery["outcome"] = outcome
        recovery["reason"] = reason
        recovery["message"] = message
        recovery["final_safety_state"] = final_safety_state
        recovery["final_evidence"] = deepcopy(final_evidence)
        recovery["ended_at_utc"] = _utc_now()
        recovery["duration_s"] = monotonic() - float(
            recovery["started_monotonic_s"]
        )
        self._write_result(result)
        if self._active_recovery == (run_id, recovery_id):
            self._active_recovery = None
        return deepcopy(recovery)

    def enter_phase(
        self,
        run_id: str,
        *,
        phase: str,
        reason: str,
        message: str,
    ) -> dict[str, Any]:
        result = self.get(run_id)
        if result["outcome"] is not None:
            raise ActiveRunError("terminal Demo Runs cannot be resumed")
        self._append_event(
            result,
            phase=phase,
            reason=reason,
            message=message,
        )
        result["current_phase"] = phase
        result["message"] = message
        self._write_result(result)
        return deepcopy(result)

    def record_preflight(
        self,
        run_id: str,
        report: dict[str, Any],
    ) -> dict[str, Any]:
        result = self.get(run_id)
        if result["outcome"] is not None:
            raise ActiveRunError("terminal Demo Runs cannot be changed")
        ready = bool(report.get("ready"))
        self._append_event(
            result,
            phase="preflight",
            reason="PREFLIGHT_PASSED" if ready else "PREFLIGHT_FAILED",
            message=(
                "all preflight checks passed"
                if ready
                else "one or more preflight checks failed"
            ),
        )
        result["preflight"] = deepcopy(report)
        result["message"] = result["events"][-1]["message"]
        self._write_result(result)
        return deepcopy(result)

    def record_home(
        self,
        run_id: str,
        home: dict[str, Any],
    ) -> dict[str, Any]:
        result = self.get(run_id)
        if result["outcome"] is not None:
            raise ActiveRunError("terminal Demo Runs cannot be changed")
        self._append_event(
            result,
            phase="capture_home",
            reason="HOME_CAPTURED",
            message="fresh robot-local position and heading captured as Home",
        )
        result["home"] = deepcopy(home)
        result["message"] = result["events"][-1]["message"]
        self._write_result(result)
        return deepcopy(result)

    def record_stage(
        self,
        run_id: str,
        phase: str,
        evidence: dict[str, Any],
    ) -> dict[str, Any]:
        result = self.get(run_id)
        if result["outcome"] is not None:
            raise ActiveRunError("terminal Demo Runs cannot be changed")
        self._append_event(
            result,
            phase=phase,
            reason=f"{phase.upper()}_COMPLETED",
            message=f"{phase} completed",
        )
        result.setdefault("stage_results", {})[phase] = deepcopy(evidence)
        result["message"] = result["events"][-1]["message"]
        self._write_result(result)
        return deepcopy(result)

    def record_artifacts(
        self,
        run_id: str,
        artifacts: list[EvidenceArtifact],
    ) -> dict[str, Any]:
        result = self.get(run_id)
        if result["outcome"] is not None:
            raise ActiveRunError("terminal Demo Runs cannot be changed")
        run_dir = self.root / str(UUID(run_id)) / "snapshots"
        descriptors: list[dict[str, Any]] = []
        for artifact in artifacts:
            destination = run_dir / artifact.filename
            temporary = run_dir / f".{artifact.filename}.tmp"
            with temporary.open("wb") as output:
                output.write(artifact.content)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, destination)
            descriptors.append(
                {
                    "filename": artifact.filename,
                    "relative_path": f"snapshots/{artifact.filename}",
                    "content_type": artifact.content_type,
                    "size_bytes": len(artifact.content),
                    "sha256": sha256(artifact.content).hexdigest(),
                }
            )
        result["artifacts"] = descriptors
        result["evidence_capture"] = {"available": True}
        self._write_result(result)
        return deepcopy(result)

    def record_evidence_unavailable(
        self,
        run_id: str,
        reason: str,
    ) -> dict[str, Any]:
        result = self.get(run_id)
        if result["outcome"] is not None:
            raise ActiveRunError("terminal Demo Runs cannot be changed")
        result["artifacts"] = []
        result["evidence_capture"] = {
            "available": False,
            "unavailable_reason": reason,
        }
        self._write_result(result)
        return deepcopy(result)

    def seal(
        self,
        run_id: str,
        *,
        phase: str,
        outcome: str,
        reason: str,
        message: str,
        final_safety_state: str,
        failed_phase: str | None = None,
        failure_details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        result = self.get(run_id)
        if result["outcome"] is not None:
            return result
        prior_phase = result["current_phase"]
        self._append_event(
            result,
            phase=phase,
            reason=reason,
            message=message,
        )
        result["current_phase"] = phase
        result["outcome"] = outcome
        result["reason"] = reason
        result["message"] = message
        result["final_safety_state"] = final_safety_state
        result["failed_phase"] = (
            failed_phase
            if failed_phase is not None
            else (prior_phase if outcome == "FAILED" else None)
        )
        if failure_details:
            result["failure_details"] = deepcopy(failure_details)
        stages = result.get("stage_results", {})
        home_distance = None
        heading_error = None
        for stage_name in ("restore_heading", "return_home", "turn_toward_home"):
            evidence = stages.get(stage_name)
            if not isinstance(evidence, dict):
                continue
            if home_distance is None and isinstance(
                evidence.get("home_distance_m"), (int, float)
            ):
                home_distance = evidence["home_distance_m"]
            if heading_error is None and isinstance(
                evidence.get("heading_error_rad"), (int, float)
            ):
                heading_error = evidence["heading_error_rad"]
        result["terminal_measurements"] = {
            "home_distance_m": home_distance,
            "heading_error_rad": heading_error,
        }
        result["ended_at_utc"] = _utc_now()
        result["duration_s"] = monotonic() - result["started_monotonic_s"]
        if outcome == "COMPLETED":
            result = self._compact_success(result)
            self._write_result(result)
            cleanup_errors = self._discard_success_details(run_id)
            if cleanup_errors:
                result["retention_cleanup"] = {
                    "complete": False,
                    "errors": cleanup_errors,
                }
                self._write_result(result)
        else:
            self._write_result(result)
        if self._active_run_id == run_id:
            self._active_run_id = None
        return deepcopy(result)

    def get(self, run_id: str) -> dict[str, Any]:
        try:
            normalized = str(UUID(run_id))
        except (ValueError, AttributeError) as exc:
            raise RunResultNotFound(run_id) from exc
        path = self.root / normalized / "result.json"
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            raise RunResultNotFound(run_id) from exc

    def list_results(self) -> list[dict[str, Any]]:
        if not self.root.exists():
            return []
        results: list[dict[str, Any]] = []
        for path in self.root.glob("*/result.json"):
            try:
                result = json.loads(path.read_text(encoding="utf-8"))
                UUID(result["run_id"])
            except (KeyError, ValueError, json.JSONDecodeError, OSError):
                continue
            results.append(result)
        return sorted(
            results,
            key=lambda result: result["started_at_utc"],
            reverse=True,
        )

    def artifact_path(
        self,
        run_id: str,
        filename: str,
    ) -> tuple[Path, str]:
        result = self.get(run_id)
        artifact = next(
            (
                item
                for item in result.get("artifacts", [])
                if isinstance(item, dict) and item.get("filename") == filename
            ),
            None,
        )
        if artifact is None:
            raise RunResultNotFound(filename)
        relative_path = artifact.get("relative_path")
        content_type = artifact.get("content_type")
        if not isinstance(relative_path, str) or not isinstance(content_type, str):
            raise RunResultNotFound(filename)
        run_dir = (self.root / str(UUID(run_id))).resolve()
        path = (run_dir / relative_path).resolve()
        if path.parent != (run_dir / "snapshots").resolve() or not path.is_file():
            raise RunResultNotFound(filename)
        return path, content_type

    def _append_event(
        self,
        result: dict[str, Any],
        *,
        phase: str,
        reason: str,
        message: str,
    ) -> None:
        event = {
            "sequence": len(result["events"]) + 1,
            "phase": phase,
            "previous_phase": (
                result["events"][-1]["phase"] if result["events"] else None
            ),
            "recorded_at_utc": _utc_now(),
            "seconds_since_start": monotonic() - result["started_monotonic_s"],
            "reason": reason,
            "message": message,
        }
        run_dir = self.root / result["run_id"]
        with (run_dir / "events.ndjson").open("a", encoding="utf-8") as journal:
            journal.write(json.dumps(event, separators=(",", ":")) + "\n")
            journal.flush()
            os.fsync(journal.fileno())
        result["events"].append(event)

    def _find_recovery(
        self, result: dict[str, Any], recovery_id: str
    ) -> dict[str, Any]:
        try:
            normalized = str(UUID(recovery_id))
        except (ValueError, AttributeError) as exc:
            raise RunResultNotFound(recovery_id) from exc
        recovery = next(
            (
                item
                for item in result.get("recovery_attempts", [])
                if isinstance(item, dict) and item.get("recovery_id") == normalized
            ),
            None,
        )
        if recovery is None:
            raise RunResultNotFound(recovery_id)
        return recovery

    def _compact_success(self, result: dict[str, Any]) -> dict[str, Any]:
        stages = result.get("stage_results") or {}
        orientation = stages.get("orient_for_run") or {}
        recognition = stages.get("find_fruit") or {}
        approach = stages.get("approach_fruit") or {}
        action = stages.get("sit_and_bark") or {}
        returned = stages.get("return_home") or {}
        restored = stages.get("restore_heading") or {}
        terminal = result.get("terminal_measurements") or {}

        durations: dict[str, float] = {}
        transitions = [
            (event.get("phase"), event.get("seconds_since_start"))
            for event in result.get("events", [])
            if event.get("phase") is not None
            and isinstance(event.get("seconds_since_start"), (int, float))
        ]
        for (phase, started), (_, ended) in pairwise(transitions):
            durations[str(phase)] = round(
                durations.get(str(phase), 0.0) + float(ended) - float(started),
                3,
            )

        key_values = {
            "completed_stages": list(stages),
            "stage_durations_s": durations,
            "motion_commands_sent": any(
                isinstance(evidence, dict)
                and evidence.get("motion_commands_sent") is True
                for evidence in stages.values()
            ),
            "measured_orientation_change_rad": orientation.get(
                "measured_yaw_change_rad"
            ),
            "recognition_label": recognition.get("label"),
            "recognition_confidence": recognition.get("confidence"),
            "recognition_stable_detections": recognition.get("stable_detections"),
            "arrival_confirmed": approach.get("arrival_confirmed"),
            "outbound_forward_pulses": approach.get("forward_pulse_count"),
            "final_push_mps": approach.get("final_push_mps"),
            "final_push_duration_s": approach.get("final_push_duration_s"),
            "bark_played": action.get("bark_played"),
            "requested_return_pulses": returned.get("requested_forward_pulses"),
            "replayed_return_pulses": returned.get("replayed_forward_pulses"),
            "home_distance_m": terminal.get("home_distance_m"),
            "heading_error_rad": terminal.get("heading_error_rad"),
            "position_tolerance_m": restored.get("position_tolerance_m"),
            "heading_tolerance_rad": restored.get("heading_tolerance_rad"),
        }
        return {
            "schema_version": 2,
            "record_type": "success_summary",
            "run_id": result["run_id"],
            "target_fruit": result["target_fruit"],
            "activation_source": result["activation_source"],
            "orientation_degrees": result.get("orientation_degrees", 0.0),
            "started_at_utc": result["started_at_utc"],
            "ended_at_utc": result["ended_at_utc"],
            "duration_s": result["duration_s"],
            "outcome": "COMPLETED",
            "reason": result["reason"],
            "current_phase": result["current_phase"],
            "message": result["message"],
            "failed_phase": None,
            "final_safety_state": result["final_safety_state"],
            "key_values": key_values,
        }

    def _discard_success_details(self, run_id: str) -> list[str]:
        run_dir = self.root / str(UUID(run_id))
        errors: list[str] = []
        events = run_dir / "events.ndjson"
        if events.is_file():
            try:
                events.unlink()
            except OSError as exc:
                errors.append(f"events.ndjson: {exc}")
        snapshots = run_dir / "snapshots"
        if snapshots.is_dir():
            try:
                shutil.rmtree(snapshots)
            except OSError as exc:
                errors.append(f"snapshots: {exc}")
        return errors

    def _write_result(self, result: dict[str, Any]) -> None:
        run_dir = self.root / result["run_id"]
        destination = run_dir / "result.json"
        temporary = run_dir / ".result.json.tmp"
        with temporary.open("w", encoding="utf-8") as output:
            json.dump(result, output, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
