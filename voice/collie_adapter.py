"""Small, explicit voice-command surface for the Border Collie demo."""

from __future__ import annotations

import json
import re
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class VoiceIntent:
    action: str
    target_fruit: Optional[str] = None


def interpret_command(text: str) -> Optional[VoiceIntent]:
    """Interpret only the deliberately supported stage phrases."""
    normalized = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    if normalized in {"stop", "stop demo", "stop the demo"}:
        return VoiceIntent(action="stop_demo")

    words = set(normalized.split())
    fruits = set()
    if words.intersection({"pear", "pears", "pair", "pairs"}):
        fruits.add("pear")
    if words.intersection({"apple", "apples"}):
        fruits.add("apple")
    if words.intersection({"banana", "bananas"}):
        fruits.add("banana")
    if len(fruits) != 1:
        return None

    # Once the wake word has opened the command window, the fruit itself is
    # the intent. This deliberately tolerates arbitrary surrounding words so
    # imperfect ASR phrasing still selects the requested fruit.
    return VoiceIntent(action="activate_demo", target_fruit=fruits.pop())


class BorderCollieAdapter:
    """Dispatch allowlisted intents through the demo's existing HTTP safety gates."""

    def __init__(
        self,
        base_url: str,
        timeout_s: float = 3.0,
        expected_build_label: str | None = None,
        expected_release_id: str | None = None,
        expected_config_schema: int | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.expected_build_label = (expected_build_label or "").strip() or None
        self.expected_release_id = (expected_release_id or "").strip() or None
        self.expected_config_schema = expected_config_schema
        self._armed = False
        self._lock = threading.Lock()

    @property
    def armed(self) -> bool:
        with self._lock:
            return self._armed

    def arm(self) -> dict:
        readiness = self.readiness()
        if not readiness["ready"]:
            raise RuntimeError(readiness["detail"])
        with self._lock:
            self._armed = True
        return readiness

    def disarm(self) -> None:
        with self._lock:
            self._armed = False

    def readiness(self) -> dict:
        """Read the clean demo's authoritative activation boundary."""
        try:
            status = self._get("/api/status")
        except RuntimeError as exc:
            return {
                "ready": False,
                "detail": str(exc),
                "url": self.base_url,
                "build_label": None,
                "release_id": None,
                "config_schema": None,
                "runtime_mode": None,
                "active_run_id": None,
                "blockers": [str(exc)],
            }

        build_label = status.get("build_label")
        release = status.get("release") or {}
        release_id = release.get("release_id")
        config_schema = release.get("config_schema")
        runtime_mode = status.get("runtime_mode")
        active_run_id = status.get("active_run_id")
        activation = status.get("activation") or {}
        mission = status.get("mission") or {}
        blockers = [
            str(item.get("detail") or item.get("name") or "unknown blocker")
            for item in activation.get("blockers") or []
            if isinstance(item, dict)
        ]
        if not activation.get("ready") and not blockers:
            blockers.append("dog activation is not ready")
        if self.expected_build_label and build_label != self.expected_build_label:
            blockers.insert(
                0,
                f"expected dog build {self.expected_build_label!r}, got {build_label!r}",
            )
        if self.expected_release_id and release_id != self.expected_release_id:
            blockers.insert(
                0,
                f"expected dog release {self.expected_release_id!r}, got {release_id!r}",
            )
        if (
            self.expected_config_schema is not None
            and config_schema != self.expected_config_schema
        ):
            blockers.insert(
                0,
                "expected dog config schema "
                f"{self.expected_config_schema}, got {config_schema!r}",
            )
        if runtime_mode != "production":
            blockers.insert(0, f"dog runtime must be 'production', got {runtime_mode!r}")
        if active_run_id:
            blockers.insert(0, f"Demo Run {active_run_id} is already active")
        if mission.get("restart_required"):
            blockers.insert(0, "physical remote takeover is latched; restart required")
        ready = bool(activation.get("ready")) and not blockers
        return {
            "ready": ready,
            "detail": "dog mission API is ready" if ready else "; ".join(blockers),
            "url": self.base_url,
            "build_label": build_label,
            "release_id": release_id,
            "config_schema": config_schema,
            "runtime_mode": runtime_mode,
            "active_run_id": active_run_id,
            "blockers": blockers,
        }

    def dispatch(self, text: str) -> dict:
        intent = interpret_command(text)
        if intent is None:
            return {"calls": [], "error": "unsupported voice command"}
        if not self.armed:
            return {
                "calls": [],
                "error": "voice actions are disarmed; review the transcript and arm them in the UI",
            }

        if intent.action == "activate_demo":
            readiness = self.readiness()
            if not readiness["ready"]:
                return {"calls": [], "error": readiness["detail"]}
            payload = {"target_fruit": intent.target_fruit, "activation_source": "voice"}
            response = self._post("/api/run", payload)
            run = response.get("run") or {}
            if (
                run.get("target_fruit") != intent.target_fruit
                or run.get("activation_source") != "voice"
                or not run.get("run_id")
            ):
                raise RuntimeError("Border Collie returned an invalid Demo Run response")
            return {
                "calls": [{
                    "tool": "activate_demo",
                    "args": payload,
                    "result": _response_summary(response),
                    "run_id": run["run_id"],
                    "phase": run.get("current_phase"),
                }]
            }

        response = self._post("/api/stop", {})
        return {
            "calls": [{
                "tool": "request_stop",
                "args": {},
                "result": _response_summary(response),
            }]
        }

    def _post(self, path: str, payload: dict) -> dict:
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                body = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Border Collie rejected the command ({exc.code}): {detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            raise RuntimeError(f"Border Collie API is unavailable: {reason}") from exc
        if not body:
            return {}
        try:
            response = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Border Collie response was not valid JSON") from exc
        if not isinstance(response, dict):
            raise RuntimeError("Border Collie response was not a JSON object")
        return response

    def _get(self, path: str) -> dict:
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            headers={"Accept": "application/json"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                body = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Border Collie status failed ({exc.code}): {detail}"
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            raise RuntimeError(f"Border Collie API is unavailable: {reason}") from exc
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Border Collie status was not valid JSON") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("Border Collie status was not a JSON object")
        return payload


def _response_summary(response: dict) -> str:
    run = response.get("run")
    if isinstance(run, dict) and run.get("run_id"):
        outcome = run.get("outcome")
        if outcome:
            return f"run {run['run_id']}: {outcome} ({run.get('reason') or 'no reason'})"
        return f"run {run['run_id']}: {run.get('current_phase') or 'accepted'}"
    for key in ("message", "status", "state", "run_id"):
        value = response.get(key)
        if value is not None:
            return str(value)
    return "accepted"
