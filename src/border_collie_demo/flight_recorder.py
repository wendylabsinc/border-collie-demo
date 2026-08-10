"""Bounded, persistent flight recorder for cross-process failure evidence."""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from time import monotonic
from typing import Any
from uuid import uuid4

from .evidence import EvidenceArtifact


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class FlightRecorder:
    """Append telemetry durably and retain a bounded set of rotated segments."""

    def __init__(
        self,
        root: Path,
        *,
        segment_max_bytes: int = 2 * 1024 * 1024,
        maximum_segments: int = 5,
    ) -> None:
        if segment_max_bytes < 1024 or maximum_segments < 1:
            raise ValueError("flight recorder bounds are too small")
        self.root = root.resolve()
        self.segment_max_bytes = int(segment_max_bytes)
        self.maximum_segments = int(maximum_segments)
        self._lock = threading.Lock()
        self._sequence = 0
        self._previous_hash: str | None = None
        self.root.mkdir(parents=True, exist_ok=True)
        self._recover_tail()

    def record(
        self,
        kind: str,
        payload: dict[str, Any],
        *,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        if not kind.strip():
            raise ValueError("flight recorder event kind is required")
        with self._lock:
            self._sequence += 1
            event = {
                "event_id": str(uuid4()),
                "sequence": self._sequence,
                "kind": kind,
                "run_id": run_id,
                "recorded_at_utc": _utc_now(),
                "recorded_monotonic_s": monotonic(),
                "previous_event_sha256": self._previous_hash,
                "payload": payload,
            }
            canonical = json.dumps(
                event,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode()
            event["event_sha256"] = sha256(canonical).hexdigest()
            line = json.dumps(event, separators=(",", ":"), default=str) + "\n"
            self._rotate_if_needed(len(line.encode()))
            active = self.root / "active.ndjson"
            with active.open("a", encoding="utf-8") as output:
                output.write(line)
                output.flush()
                os.fsync(output.fileno())
            self._fsync_directory()
            self._previous_hash = str(event["event_sha256"])
            return dict(event)

    def snapshot(self) -> EvidenceArtifact:
        with self._lock:
            paths = [*sorted(self.root.glob("segment-*.ndjson")), self.root / "active.ndjson"]
            content = b"".join(path.read_bytes() for path in paths if path.is_file())
        return EvidenceArtifact(
            filename="flight-recorder.ndjson",
            content_type="application/x-ndjson",
            content=content,
        )

    def status(self) -> dict[str, object]:
        with self._lock:
            paths = [*self.root.glob("segment-*.ndjson"), self.root / "active.ndjson"]
            existing = [path for path in paths if path.is_file()]
            return {
                "sequence": self._sequence,
                "segments": len(existing),
                "bytes": sum(path.stat().st_size for path in existing),
                "maximum_segments": self.maximum_segments,
                "segment_max_bytes": self.segment_max_bytes,
            }

    def _rotate_if_needed(self, next_bytes: int) -> None:
        active = self.root / "active.ndjson"
        current = active.stat().st_size if active.is_file() else 0
        if current == 0 or current + next_bytes <= self.segment_max_bytes:
            return
        destination = self.root / (
            f"segment-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.ndjson"
        )
        os.replace(active, destination)
        segments = sorted(self.root.glob("segment-*.ndjson"))
        while len(segments) > self.maximum_segments:
            segments.pop(0).unlink()
        self._fsync_directory()

    def _recover_tail(self) -> None:
        paths = [*sorted(self.root.glob("segment-*.ndjson")), self.root / "active.ndjson"]
        for path in paths:
            if not path.is_file():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    event = json.loads(line)
                    sequence = int(event["sequence"])
                    event_hash = str(event["event_sha256"])
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    continue
                if sequence >= self._sequence:
                    self._sequence = sequence
                    self._previous_hash = event_hash

    def _fsync_directory(self) -> None:
        descriptor = os.open(self.root, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def terminal_evidence_bundle(
    recorder: FlightRecorder,
    external_capture: Any | None,
) -> list[EvidenceArtifact]:
    artifacts: list[EvidenceArtifact] = []
    warnings: list[str] = []
    try:
        artifacts.append(recorder.snapshot())
    except Exception as exc:  # noqa: BLE001 - evidence cannot mask safety
        warnings.append(f"flight recorder snapshot failed: {exc}")
    if external_capture is not None:
        try:
            artifacts.extend(external_capture())
        except Exception as exc:  # noqa: BLE001 - preserve local evidence
            warnings.append(f"media evidence capture failed: {exc}")
    else:
        warnings.append("media evidence adapter is not configured")
    if warnings:
        artifacts.append(
            EvidenceArtifact(
                filename="evidence-capture-warnings.json",
                content_type="application/json",
                content=(json.dumps({"warnings": warnings}, indent=2) + "\n").encode(),
            )
        )
    if not artifacts:
        raise RuntimeError("no terminal evidence could be captured")
    return artifacts
