"""Append-only, fsynced black-box timeline for each Demo Run."""

from __future__ import annotations

import json
import os
import queue
import threading
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Any
from uuid import UUID

from .operator_logging import OperatorEventSink


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class RunBlackBox:
    """Record ordered diagnostic facts without retaining camera images."""

    def __init__(
        self,
        root: Path,
        *,
        operator_sink: OperatorEventSink | None = None,
    ) -> None:
        self.root = root.resolve()
        self._operator_sink = operator_sink
        self._lock = threading.Lock()
        self._sequences: dict[str, int] = {}
        self._closed = False
        self._last_error: str | None = None
        self._last_operator_error: str | None = None
        self._pending: queue.Queue[tuple[Path, dict[str, Any]] | None] = queue.Queue()
        self._writer = threading.Thread(
            target=self._write_loop,
            name="border-collie-black-box",
            daemon=True,
        )
        self._writer.start()

    def record(
        self,
        run_id: str,
        kind: str,
        *,
        phase: str | None,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        normalized = str(UUID(run_id))
        if not kind.strip():
            raise ValueError("black-box event kind is required")
        with self._lock:
            path = self._path(normalized)
            sequence = self._sequences.get(normalized)
            if sequence is None:
                self.flush()
                sequence = self._last_sequence(path)
            sequence += 1
            event = {
                "schema_version": 1,
                "sequence": sequence,
                "recorded_at_utc": _utc_now(),
                "recorded_monotonic_s": monotonic(),
                "kind": kind.strip(),
                "phase": phase,
                "payload": deepcopy(payload),
            }
            self._sequences[normalized] = sequence
        self._pending.put((path, event))
        if self._operator_sink is not None:
            try:
                self._operator_sink.emit(normalized, deepcopy(event))
            except Exception as exc:  # noqa: BLE001 - logs never affect safety
                self._last_operator_error = str(exc)
        return deepcopy(event)

    def read(self, run_id: str) -> list[dict[str, Any]]:
        self.flush()
        path = self._path(str(UUID(run_id)))
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return []
        return [json.loads(line) for line in lines if line.strip()]

    def path(self, run_id: str) -> Path:
        self.flush()
        path = self._path(str(UUID(run_id))).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def flush(self) -> None:
        self._pending.join()

    def close(self) -> None:
        if self._closed:
            return
        self.flush()
        self._pending.put(None)
        self._writer.join(timeout=2.0)
        self._closed = True

    def _write_loop(self) -> None:
        while True:
            item = self._pending.get()
            try:
                if item is None:
                    return
                path, event = item
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as output:
                    output.write(json.dumps(event, separators=(",", ":")) + "\n")
                    output.flush()
                    os.fsync(output.fileno())
            except Exception as exc:  # noqa: BLE001 - recorder cannot stop motion
                self._last_error = str(exc)
            finally:
                self._pending.task_done()

    def _path(self, run_id: str) -> Path:
        return self.root / run_id / "black-box.ndjson"

    @staticmethod
    def _last_sequence(path: Path) -> int:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return 0
        for line in reversed(lines):
            if not line.strip():
                continue
            try:
                return int(json.loads(line)["sequence"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
        return 0
