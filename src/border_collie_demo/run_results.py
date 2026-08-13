from __future__ import annotations

import json
import os
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from time import monotonic
from typing import Any
from uuid import UUID, uuid4

from .black_box import RunBlackBox
from .evidence import EvidenceArtifact


class ActiveRunError(RuntimeError):
    """Raised when activation is requested while a Demo Run is active."""


class RunResultNotFound(KeyError):
    """Raised when a full Run Result identifier cannot be read."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class RunResultStore:
    """Append-only Demo Run journal with an atomically materialized result."""

    def __init__(self, root: Path, *, black_box: RunBlackBox | None = None) -> None:
        self.root = root.resolve()
        self.black_box = black_box or RunBlackBox(self.root)
        self._active_run_id: str | None = None

    @property
    def active_run_id(self) -> str | None:
        return self._active_run_id

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
            if result.get("outcome") is not None:
                continue
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
        self._active_run_id = None
        return sealed

    def start_run(
        self,
        *,
        target_fruit: str,
        activation_source: str,
        activation_id: str | None = None,
        run_tuning: dict[str, object] | None = None,
        search_experiment: dict[str, object] | None = None,
    ) -> dict[str, Any]:
        if self._active_run_id is not None:
            raise ActiveRunError("a Demo Run is already active")

        run_id = str(uuid4())
        started_utc = _utc_now()
        started_monotonic_s = monotonic()
        result: dict[str, Any] = {
            "schema_version": 1,
            "run_id": run_id,
            "target_fruit": target_fruit,
            "activation_source": activation_source,
            "activation_id": activation_id,
            "run_tuning": deepcopy(run_tuning),
            "search_experiment": deepcopy(search_experiment),
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
        self.black_box.record(
            run_id,
            "run_started",
            phase="idle",
            payload={
                "target_fruit": target_fruit,
                "activation_source": activation_source,
                "activation_id": activation_id,
                "run_tuning": deepcopy(run_tuning),
                "search_experiment": deepcopy(search_experiment),
            },
        )
        self._append_event(
            result,
            phase="idle",
            reason="ACTIVATION_ACCEPTED",
            message="Demo Run persisted before preflight",
        )
        self._write_result(result)
        self._active_run_id = run_id
        return deepcopy(result)

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
        self.black_box.record(
            run_id,
            "stage_result",
            phase=phase,
            payload=evidence,
        )
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

    def record_failure_epilogue(
        self,
        run_id: str,
        report: dict[str, object],
    ) -> dict[str, Any]:
        """Persist recovery evidence without changing the original outcome."""
        result = self.get(run_id)
        if result["outcome"] is not None:
            raise ActiveRunError("terminal Demo Runs cannot be changed")
        status = str(report.get("status") or "UNKNOWN")
        self._append_event(
            result,
            phase="failure_epilogue",
            reason=f"FAILURE_EPILOGUE_{status}",
            message=str(report.get("reason") or "failure epilogue completed"),
        )
        result["failure_epilogue"] = deepcopy(report)
        self.black_box.record(
            run_id,
            "failure_epilogue",
            phase="failure_epilogue",
            payload=dict(report),
        )
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
        epilogue = result.get("failure_epilogue")
        terminal_measurement = (
            epilogue.get("terminal_home_measurement")
            if isinstance(epilogue, dict)
            else None
        )
        if isinstance(terminal_measurement, dict) and isinstance(
            terminal_measurement.get("home_distance_m"), (int, float)
        ):
            home_distance = terminal_measurement["home_distance_m"]
        elif outcome == "COMPLETED":
            evidence = stages.get("return_home")
            if isinstance(evidence, dict) and isinstance(
                evidence.get("home_distance_m"), (int, float)
            ):
                home_distance = evidence["home_distance_m"]
        result["terminal_measurements"] = {
            "home_distance_m": home_distance,
            "heading_error_rad": heading_error,
        }
        result["ended_at_utc"] = _utc_now()
        result["duration_s"] = monotonic() - result["started_monotonic_s"]
        self.black_box.record(
            run_id,
            "run_sealed",
            phase=phase,
            payload={
                "outcome": outcome,
                "reason": reason,
                "failed_phase": result["failed_phase"],
                "final_safety_state": final_safety_state,
            },
        )
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
        self.black_box.record(
            result["run_id"],
            "mission_event",
            phase=phase,
            payload=event,
        )

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
