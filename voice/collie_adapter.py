"""Small, explicit voice-command surface for the Border Collie demo."""

from __future__ import annotations

import json
import re
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass


#: Motion-qualified fruits.  Banana is deliberately excluded; the demo API
#: answers 409 for it.
QUALIFIED_FRUITS = ("apple", "mango", "pear")

#: The "full demo" cohort mirrors the web UI's "Run Apple + Mango + Pear once"
#: button (``id="start-three-fruit"`` in ``web/index.html``).  Keep these values
#: in step with that handler so the voice path and the button stay identical.
FULL_DEMO_RUNS = 3
FULL_DEMO_FRUIT_SUBSET = ("apple", "mango", "pear")
FULL_DEMO_SEED = 20260813
FULL_DEMO_TOLERATED_FAILURES = (
    {"reason": "TARGET_RECOGNITION_FAILURE"},
    {"reason": "TARGET_LOST"},
    {"reason": "ARRIVAL_FAILURE"},
    {"reason": "ACTION_FAILURE"},
    {"failed_phase": "turn_to_fruit"},
    {"failed_phase": "find_fruit"},
    {"failed_phase": "approach_fruit"},
    {"failed_phase": "arrived"},
    {"failed_phase": "sit_and_bark"},
    {"failed_phase": "stand"},
)
FULL_DEMO_TUNING = {
    "search": {"yaw_rps": 0.8},
    "home": {"align_yaw_rps": 0.8},
}

#: Words an operator naturally puts in front of a stage command.  Stripping
#: stops at the first non-filler, so "ok so the mango" never reduces to "mango".
_LEAD_FILLERS = frozenset({
    "hey", "wendy", "ok", "okay", "please", "now",
    "start", "run", "do", "let", "lets", "s", "the", "a",
})
_TRAIL_FILLERS = frozenset({"please", "now", "thanks", "thank", "you"})

#: Accepted spellings of the 3-run cohort command, matched against the *whole*
#: utterance (after filler stripping) rather than searched for inside it.
_FULL_DEMO_PHRASES = frozenset({
    "full demo",
    "fulldemo",
    "full demos",
    "full demo run",
    "full demo runs",
    "full demo cohort",
    "full fruit demo",
    "full three fruit demo",
})

#: Accepted spellings of the single Mango Demo Run, matched the same way.
#: "man go" is included because Parakeet splits the word; it is only safe
#: because the match is anchored to the entire utterance.
_MANGO_PHRASES = frozenset({
    "mango",
    "mangos",
    "mangoes",
    "man go",
    "mango run",
    "mango demo",
    "mango demo run",
    "one mango",
    "single mango",
})


@dataclass(frozen=True)
class VoiceIntent:
    action: str
    target_fruit: str | None = None


def _canonical_words(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).split()


def _core_phrase(words: list[str]) -> str:
    """Drop bounded leading/trailing filler and return what is left."""
    start = 0
    while start < len(words) and words[start] in _LEAD_FILLERS:
        start += 1
    end = len(words)
    while end > start and words[end - 1] in _TRAIL_FILLERS:
        end -= 1
    return " ".join(words[start:end])


def _mentions_full_demo(words: list[str]) -> bool:
    """True when the operator said "full demo" anywhere in the utterance."""
    if "fulldemo" in words:
        return True
    return any(first == "full" and second == "demo"
               for first, second in zip(words, words[1:]))


def interpret_command(text: str) -> VoiceIntent | None:
    """Interpret only the deliberately supported stage phrases."""
    words = _canonical_words(text)
    normalized = " ".join(words)
    if normalized in {"stop", "stop demo", "stop the demo"}:
        return VoiceIntent(action="stop_demo")

    core = _core_phrase(words)

    # The whole-cohort command is resolved first, so a fruit named inside it can
    # never win the utterance.
    if core in _FULL_DEMO_PHRASES:
        return VoiceIntent(action="full_demo")

    # "full demo" said as part of a longer sentence ("run the full demo and find
    # the mango") is ambiguous.  Refuse it outright rather than falling through
    # to the single-fruit rules and starting the wrong thing on a real robot.
    if _mentions_full_demo(words):
        return None

    # Bare "mango" is a Demo Run only as the entire utterance.  Anchoring is what
    # keeps the word harmless inside ordinary speech ("I like mango").
    if core in _MANGO_PHRASES:
        return VoiceIntent(action="activate_demo", target_fruit="mango")

    unique = set(words)
    fruits = set()
    if unique.intersection({"pear", "pears", "pair", "pairs"}):
        fruits.add("pear")
    if unique.intersection({"apple", "apples"}):
        fruits.add("apple")
    if unique.intersection({"mango", "mangoes", "mangos"}):
        fruits.add("mango")
    if len(fruits) != 1:
        return None

    # A fruit name alone is not motion intent.  Keep this allowlist deliberately
    # small so ordinary stage conversation about fruit cannot activate Woof.
    if not unique.intersection({"find", "follow", "locate", "search", "seek", "go"}):
        return None

    return VoiceIntent(action="activate_demo", target_fruit=fruits.pop())


def display_command(intent: VoiceIntent) -> str:
    """Short operator-facing label for a recognised intent."""
    if intent.action == "activate_demo":
        return intent.target_fruit or "run"
    if intent.action == "full_demo":
        return "full demo"
    return "stop"


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

    def dispatch(self, text: str, *, activation_id: str | None = None) -> dict:
        intent = interpret_command(text)
        if intent is None:
            return {"calls": [], "error": "unsupported voice command"}
        if not self.armed:
            return {
                "calls": [],
                "error": "voice actions are disarmed; review the transcript and arm them in the UI",
            }

        if intent.action == "activate_demo":
            durable_id = (activation_id or "").strip()
            if not durable_id:
                raise ValueError("activation_id is required for a voice Demo Run")
            readiness = self.readiness()
            if not readiness["ready"]:
                return {"calls": [], "error": readiness["detail"]}
            payload = {
                "target_fruit": intent.target_fruit,
                "activation_source": "voice",
                "activation_id": durable_id,
            }
            response = self._post("/api/run", payload)
            run = response.get("run") or {}
            if (
                run.get("target_fruit") != intent.target_fruit
                or run.get("activation_source") != "voice"
                or run.get("activation_id") != durable_id
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

        if intent.action == "full_demo":
            readiness = self.readiness()
            if not readiness["ready"]:
                return {"calls": [], "error": readiness["detail"]}
            payload = {
                "runs": FULL_DEMO_RUNS,
                "randomized": True,
                "target_fruit": None,
                "fruit_subset": list(FULL_DEMO_FRUIT_SUBSET),
                "seed": FULL_DEMO_SEED,
                "tolerated_failures": [dict(item) for item in FULL_DEMO_TOLERATED_FAILURES],
                "tuning": {group: dict(values) for group, values in FULL_DEMO_TUNING.items()},
            }
            response = self._post("/api/cohorts", payload)
            cohort = response.get("cohort") or {}
            if not cohort.get("cohort_id"):
                raise RuntimeError("Border Collie returned an invalid cohort response")
            return {
                "calls": [{
                    "tool": "start_full_demo",
                    "args": payload,
                    "result": _response_summary(response),
                    "cohort_id": cohort["cohort_id"],
                    "fruit_sequence": cohort.get("fruit_sequence"),
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
            raise RuntimeError(_http_error_message(exc)) from exc
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
            raise TypeError("Border Collie response was not a JSON object")
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
            raise TypeError("Border Collie status was not a JSON object")
        return payload


#: Operator-facing prefixes for the two refusals the demo API can return.  These
#: reach the page as ``BLOCKED — <message>`` via the adapter's existing error
#: path, so a refused command is never silently dropped.
_STATUS_PREFIXES = {
    409: "Border Collie refused the command (409 conflict)",
    423: (
        "Border Collie is locked (423; physical remote takeover is latched, "
        "restart the demo)"
    ),
}


def _http_error_message(exc: urllib.error.HTTPError) -> str:
    raw = exc.read().decode("utf-8", errors="replace")
    detail = raw
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict) and parsed.get("detail"):
        detail = str(parsed["detail"])
    prefix = _STATUS_PREFIXES.get(
        exc.code, f"Border Collie rejected the command ({exc.code})"
    )
    return f"{prefix}: {detail.strip()}"


def _response_summary(response: dict) -> str:
    cohort = response.get("cohort")
    if isinstance(cohort, dict) and cohort.get("cohort_id"):
        sequence = ", ".join(cohort.get("fruit_sequence") or []) or "no sequence"
        status = cohort.get("status") or "accepted"
        return f"cohort {cohort['cohort_id']}: {status} ({sequence})"

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
