"""Physical Go2 Start-button activation behind one evidence-rich seam.

Pressing Start on the Go2 controller requests exactly one three-fruit cohort:
one Apple, one Mango and one Pear Demo Run, in a seeded random order.  The
adapter owns decoding, neutral-before-arm behavior, rising-edge deduplication
and black-box attribution.  It never sends a robot command and never builds a
cohort policy of its own: the injected ``start_cohort`` callable is the same
``CohortController.start`` path the "Run Apple + Mango + Pear once" UI button
drives through ``POST /api/cohorts``, so preflight, exact-zero disarm, the
Remote Takeover latch and the per-run Home clearance gate all still apply.
"""

from __future__ import annotations

import asyncio
import inspect
import math
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol

from .black_box import RunBlackBox


START_BUTTON_MASK = 1 << 2
START_BUTTON_NAME = "start"


class ControllerStartRefused(RuntimeError):
    """The Start edge was understood but refused before touching the cohort.

    ``disposition`` mirrors the HTTP guard that would have rejected the same
    request: ``takeover_latched`` is the 423 case, ``active_cohort`` and
    ``active_run`` are the 409 cases.
    """

    def __init__(self, disposition: str, detail: str) -> None:
        super().__init__(detail)
        self.disposition = disposition
        self.detail = detail


@dataclass(frozen=True)
class ControllerSample:
    """One received Unitree LowState controller sample with source evidence."""

    source: str
    source_sequence: int
    received_monotonic_s: float
    wireless_remote: bytes

    def __post_init__(self) -> None:
        source = self.source.strip()
        if not source:
            raise ValueError("controller sample source is required")
        if isinstance(self.source_sequence, bool) or self.source_sequence < 0:
            raise ValueError("controller source sequence must be non-negative")
        received = float(self.received_monotonic_s)
        if not math.isfinite(received) or received < 0.0:
            raise ValueError("controller receive time must be finite and non-negative")
        remote = bytes(self.wireless_remote)
        if len(remote) < 4:
            raise ValueError("wireless_remote must contain at least four bytes")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "received_monotonic_s", received)
        object.__setattr__(self, "wireless_remote", remote)


@dataclass(frozen=True)
class ControllerDecision:
    """Observable outcome of one controller sample."""

    disposition: str
    activation_id: str | None = None
    cohort_id: str | None = None
    detail: str | None = None


StartCohort = Callable[[], Awaitable[dict[str, Any]]]
ActiveRunId = Callable[[], str | None]
ControllerObserver = Callable[[ControllerSample], Awaitable[object] | object]


class ControllerStartSource(Protocol):
    async def start(self, observer: ControllerObserver) -> None: ...

    async def close(self) -> None: ...

    def status(self) -> dict[str, object]: ...


class ControllerStartAdapter:
    """Turn one verified Start rising edge into at most one three-fruit cohort.

    The first observed state must be released.  Consequently a controller held
    across process or robot boot cannot cause motion when Wendy restores the
    app.  Repeated DDS delivery and a held button cannot submit a second
    cohort; a release followed by a fresh press is a new request, but it is
    still refused while a cohort or Demo Run owns activation.
    """

    def __init__(
        self,
        *,
        start_cohort: StartCohort,
        black_box: RunBlackBox,
        active_run_id: ActiveRunId,
        dedupe_window: int = 128,
        event_history: int = 16,
    ) -> None:
        if dedupe_window < 2:
            raise ValueError("controller dedupe window must be at least two samples")
        if event_history < 1:
            raise ValueError("controller event history must retain at least one edge")
        self._start_cohort = start_cohort
        self._black_box = black_box
        self._active_run_id = active_run_id
        self._lock = asyncio.Lock()
        self._armed_after_release = False
        self._pressed = False
        self._seen_order: deque[tuple[str, int]] = deque()
        self._seen: set[tuple[str, int]] = set()
        self._dedupe_window = dedupe_window
        self._last_decision = "waiting_for_release"
        self._last_source_sequence: int | None = None
        self._accepted_edges = 0
        self._last_cohort_id: str | None = None
        self._last_error: str | None = None
        # Accepted cohorts have no Run Result yet, so the black box cannot own
        # their attribution.  Keep a bounded edge history readable from status.
        self._events: deque[dict[str, Any]] = deque(maxlen=event_history)

    async def observe(self, sample: ControllerSample) -> ControllerDecision:
        """Consume one sample; only a neutral-armed Start edge can activate."""

        async with self._lock:
            identity = (sample.source, sample.source_sequence)
            if identity in self._seen:
                return self._decision("duplicate")
            self._remember(identity)
            self._last_source_sequence = sample.source_sequence

            button_word = int.from_bytes(sample.wireless_remote[2:4], "little")
            pressed = bool(button_word & START_BUTTON_MASK)
            evidence: dict[str, Any] = {
                "button": START_BUTTON_NAME,
                "button_mask": START_BUTTON_MASK,
                "button_word": button_word,
                "received_monotonic_s": sample.received_monotonic_s,
                "source": sample.source,
                "source_sequence": sample.source_sequence,
                "start_pressed": pressed,
            }

            if not self._armed_after_release:
                self._pressed = pressed
                if pressed:
                    return self._decision("startup_held")
                self._armed_after_release = True
                return self._decision("armed")

            if not pressed:
                self._pressed = False
                return self._decision("released")
            if self._pressed:
                return self._decision("held")

            self._pressed = True
            activation_id = (
                f"go2-controller-start:{sample.source}:{sample.source_sequence}"
            )
            try:
                cohort = await self._start_cohort()
            except ControllerStartRefused as exc:
                return self._refuse(
                    evidence, activation_id, exc.disposition, exc.detail
                )
            except Exception as exc:  # noqa: BLE001 - refuse, never crash the app
                return self._refuse(evidence, activation_id, "rejected", str(exc))

            cohort_id = str(cohort.get("cohort_id"))
            payload = {
                **evidence,
                "activation_id": activation_id,
                "cohort_id": cohort_id,
                "disposition": "accepted",
                "fruit_sequence": list(cohort.get("fruit_sequence") or []),
                "runs": (cohort.get("policy") or {}).get("runs"),
                "seed": (cohort.get("policy") or {}).get("seed"),
            }
            self._accepted_edges += 1
            self._last_cohort_id = cohort_id
            self._last_error = None
            self._events.append(payload)
            return self._decision(
                "accepted", activation_id=activation_id, cohort_id=cohort_id
            )

    def status(self) -> dict[str, object]:
        return {
            "ready": self._armed_after_release,
            "detail": (
                "waiting for Start release before controller activation"
                if not self._armed_after_release
                else "Start rising edge is armed for one Apple + Mango + Pear cohort"
            ),
            "button": START_BUTTON_NAME,
            "button_mask": START_BUTTON_MASK,
            "last_disposition": self._last_decision,
            "last_source_sequence": self._last_source_sequence,
            "accepted_edges": self._accepted_edges,
            "last_cohort_id": self._last_cohort_id,
            "last_error": self._last_error,
            "recent_edges": [dict(event) for event in self._events],
        }

    def _refuse(
        self,
        evidence: dict[str, Any],
        activation_id: str,
        disposition: str,
        detail: str,
    ) -> ControllerDecision:
        self._last_error = detail
        payload = {
            **evidence,
            "activation_id": activation_id,
            "disposition": disposition,
            "error": detail,
        }
        self._events.append(payload)
        run_id = self._active_run_id()
        if run_id is not None:
            try:
                self._black_box.record(
                    run_id,
                    "controller_start",
                    phase="controller_input",
                    payload=payload,
                )
            except Exception as exc:  # noqa: BLE001 - attribution is best effort
                self._last_error = f"{detail}; black-box attribution failed: {exc}"
        return self._decision(
            disposition, activation_id=activation_id, detail=detail
        )

    def _decision(
        self,
        disposition: str,
        *,
        activation_id: str | None = None,
        cohort_id: str | None = None,
        detail: str | None = None,
    ) -> ControllerDecision:
        self._last_decision = disposition
        return ControllerDecision(disposition, activation_id, cohort_id, detail)

    def _remember(self, identity: tuple[str, int]) -> None:
        while len(self._seen_order) >= self._dedupe_window:
            expired = self._seen_order.popleft()
            self._seen.discard(expired)
        self._seen_order.append(identity)
        self._seen.add(identity)


class Go2ControllerStartSource:
    """Read-only DDS source for the Go2 controller's LowState payload.

    DDS callback threading, Unitree dynamic imports, and shutdown draining stay
    behind this source.  It emits immutable ``ControllerSample`` values and has
    no reference to motion clients or mission policy.
    """

    def __init__(
        self,
        *,
        topic: str = "rt/lf/lowstate",
        subscriber_factory: Callable[[str, object], object] | None = None,
        message_type: object | None = None,
        monotonic_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        topic = topic.strip()
        if not topic:
            raise ValueError("controller LowState topic is required")
        self.topic = topic
        self._subscriber_factory = subscriber_factory
        self._message_type = message_type
        self._clock = monotonic_clock
        self._subscriber: object | None = None
        self._observer: ControllerObserver | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._tasks: set[asyncio.Task[object]] = set()
        self._accepting = False
        self._received_samples = 0
        self._last_source_sequence: int | None = None
        self._last_error: str | None = None

    async def start(self, observer: ControllerObserver) -> None:
        if self._subscriber is not None:
            return
        self._loop = asyncio.get_running_loop()
        self._observer = observer
        try:
            factory, message_type = self._resolve_unitree_types()
            subscriber = factory(self.topic, message_type)
            subscriber.Init(self._on_lowstate, 10)
        except Exception as exc:  # noqa: BLE001 - Unitree DDS is dynamically typed
            self._last_error = f"controller subscriber failed: {exc}"
            self._accepting = False
            return
        self._subscriber = subscriber
        self._accepting = True
        self._last_error = None

    async def close(self) -> None:
        subscriber, self._subscriber = self._subscriber, None
        if subscriber is not None:
            try:
                subscriber.Close()
            except Exception as exc:  # noqa: BLE001 - shutdown is best effort
                self._last_error = f"controller subscriber close failed: {exc}"
        # A callback already handed to the event loop must finish attribution
        # before the app closes its Run Result/black-box writers.
        await asyncio.sleep(0)
        self._accepting = False
        if self._tasks:
            await asyncio.gather(*tuple(self._tasks), return_exceptions=True)
        self._observer = None
        self._loop = None

    def status(self) -> dict[str, object]:
        return {
            "connected": self._subscriber is not None and self._accepting,
            "topic": self.topic,
            "received_samples": self._received_samples,
            "last_source_sequence": self._last_source_sequence,
            "last_error": self._last_error,
        }

    def _resolve_unitree_types(self) -> tuple[Callable[[str, object], object], object]:
        factory = self._subscriber_factory
        message_type = self._message_type
        if factory is None:
            from unitree_sdk2py.core.channel import ChannelSubscriber

            factory = ChannelSubscriber
        if message_type is None:
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_

            message_type = LowState_
        return factory, message_type

    def _on_lowstate(self, message: object) -> None:
        try:
            sample = ControllerSample(
                source=self.topic,
                source_sequence=int(getattr(message, "tick")),
                received_monotonic_s=self._clock(),
                wireless_remote=bytes(getattr(message, "wireless_remote")),
            )
        except Exception as exc:  # noqa: BLE001 - malformed DDS data is evidence
            self._last_error = f"invalid controller sample: {exc}"
            return
        self._received_samples += 1
        self._last_source_sequence = sample.source_sequence
        loop = self._loop
        if loop is None or not self._accepting:
            return
        loop.call_soon_threadsafe(self._dispatch, sample)

    def _dispatch(self, sample: ControllerSample) -> None:
        if not self._accepting or self._observer is None:
            return
        try:
            result = self._observer(sample)
        except Exception as exc:  # noqa: BLE001 - input never crashes the app
            self._last_error = f"controller observer failed: {exc}"
            return
        if not inspect.isawaitable(result):
            return
        task = asyncio.create_task(result, name=f"go2-start-{sample.source_sequence}")
        self._tasks.add(task)
        task.add_done_callback(self._task_done)

    def _task_done(self, task: asyncio.Task[object]) -> None:
        self._tasks.discard(task)
        if task.cancelled():
            return
        try:
            error = task.exception()
        except Exception as exc:  # noqa: BLE001 - diagnostic only
            self._last_error = f"controller activation failed: {exc}"
            return
        if error is not None:
            self._last_error = f"controller activation failed: {error}"
