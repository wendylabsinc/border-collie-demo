"""Physical Go2 controller input behind one evidence-rich seam.

The adapter turns raw ``LowState_.wireless_remote`` frames into exactly two
operator intents, and never sends a robot command of its own:

* **Idle** -- three deliberate Start presses inside a bounded window request
  one three-fruit cohort: one Apple, one Mango and one Pear Demo Run, in a
  seeded random order.  The injected ``start_cohort`` callable is the same
  ``CohortController.start`` path the "Run Apple + Mango + Pear once" UI button
  drives through ``POST /api/cohorts``, so preflight, exact-zero disarm, the
  Remote Takeover latch and the per-run Home clearance gate all still apply.
* **Run or cohort active** -- *any* freshly touched control, button or stick,
  stops immediately through the injected ``stop_active`` callable, which is the
  same safe path ``POST /api/stop`` uses.  Stopping is never rate-limited,
  never counted and never debounced: one input, one stop.

The adapter owns decoding, neutral-before-arm behavior, rising-edge
deduplication, the press counter and black-box attribution.  It owns no motion
client and no writer.
"""

from __future__ import annotations

import asyncio
import inspect
import math
import struct
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol

from .black_box import RunBlackBox
from .models import RemoteInput


# ---------------------------------------------------------------------------
# wireless_remote frame layout
#
# Verified against the pinned SDK (unitree_sdk2_python @ e4cd91f, see
# build.stagefile.yaml) and Unitree's own xRockerBtnDataStruct:
#
#     uint8_t head[2];      // bytes  0..1
#     uint16  btn;          // bytes  2..3   little-endian button bitfield
#     float   lx;           // bytes  4..7
#     float   rx;           // bytes  8..11
#     float   ry;           // bytes 12..15
#     float   L2;           // bytes 16..19  analog trigger, SDK marks unused
#     float   ly;           // bytes 20..23
#     uint8_t idle[16];     // bytes 24..39
#
# Note ``ly`` is at offset 20, not 16: the analog L2 float sits between ry and
# ly. The L2 *button* bit is still watched, so squeezing L2 is not missed; only
# its analog channel is skipped, because the SDK itself labels that channel a
# placeholder and its resting value is not guaranteed to be zero.
# ---------------------------------------------------------------------------
BUTTON_WORD_OFFSET = 2
DECODABLE_FRAME_BYTES = 24
WIRELESS_REMOTE_BYTES = 40

BUTTON_NAMES: tuple[str, ...] = (
    "R1", "L1", "start", "select", "R2", "L2", "F1", "F2",
    "A", "B", "X", "Y", "up", "right", "down", "left",
)
STICK_AXES: tuple[tuple[str, int], ...] = (
    ("left_stick_x", 4),
    ("right_stick_x", 8),
    ("right_stick_y", 12),
    ("left_stick_y", 20),
)

START_BUTTON_MASK = 1 << 2
START_BUTTON_NAME = "start"

# Stick displacement that counts as a deliberate touch, in normalized units
# where full deflection is 1.0.
#
# Unitree's own SDK applies a 0.01 dead zone to these axes, which is its
# implied noise floor, and a sibling project tuned a 0.12 resting band against
# real handheld-controller hardware. 0.15 sits above the largest measured
# resting drift with margin, so a controller lying on a table never stops a
# run, while a deliberate nudge -- which drives an axis to 0.5..1.0 almost
# immediately -- crosses it on the first sample. That margin is what lets the
# stop stay single-sample and instant instead of needing a debounce that would
# delay it.
STICK_DEADZONE = 0.15

# Deliberate presses required to launch a cohort. One press was too easy to
# trigger by brushing the controller.
START_PRESS_COUNT = 3

# All the presses must land inside this window, measured from the first one.
# Three deliberate presses take roughly one to two seconds, so 5.0 s tolerates
# a hesitant operator and a dropped DDS sample without letting stale presses
# accumulate: a press now and a press an hour later can never combine, and two
# accidental brushes minutes apart cannot arm the demo.
START_SEQUENCE_WINDOW_S = 5.0

# Continuous neutral required before a latched Remote Takeover is released.
# A stick swept between extremes passes through center in far less than this,
# so mid-takeover transits never clear the latch, while an operator who has
# actually put the controller down waits only a beat.
TAKEOVER_RELEASE_HOLD_S = 2.0


@dataclass(frozen=True)
class ControllerFrame:
    """One decoded controller frame: which controls the operator is touching."""

    button_word: int
    buttons: tuple[str, ...]
    axes: tuple[tuple[str, float], ...]
    displaced_axes: tuple[str, ...]

    @property
    def controls(self) -> frozenset[str]:
        return frozenset(self.buttons) | frozenset(self.displaced_axes)

    @property
    def neutral(self) -> bool:
        return not self.buttons and not self.displaced_axes

    def to_evidence(self) -> dict[str, Any]:
        return {
            "button_word": self.button_word,
            "buttons_pressed": list(self.buttons),
            "axes": {name: value for name, value in self.axes},
            "axes_displaced": list(self.displaced_axes),
            "neutral": self.neutral,
        }


def decode_frame(
    wireless_remote: bytes, *, deadzone: float = STICK_DEADZONE
) -> ControllerFrame:
    """Decode buttons and stick axes from one wireless_remote payload."""

    button_word = int.from_bytes(
        wireless_remote[BUTTON_WORD_OFFSET : BUTTON_WORD_OFFSET + 2], "little"
    )
    buttons = tuple(
        name for index, name in enumerate(BUTTON_NAMES) if button_word & (1 << index)
    )
    axes: list[tuple[str, float]] = []
    displaced: list[str] = []
    for name, offset in STICK_AXES:
        (value,) = struct.unpack_from("<f", wireless_remote, offset)
        value = float(value)
        if not math.isfinite(value):
            # A corrupt float is evidence, not an operator input. Reporting it
            # as a displacement would stop every run the moment the link
            # glitched, so it is recorded and treated as centered.
            axes.append((name, 0.0))
            continue
        axes.append((name, value))
        if abs(value) > deadzone:
            displaced.append(name)
    return ControllerFrame(
        button_word=button_word,
        buttons=buttons,
        axes=tuple(axes),
        displaced_axes=tuple(displaced),
    )


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
        if len(remote) < DECODABLE_FRAME_BYTES:
            # Buttons alone need four bytes, but the stick axes run to offset
            # 24. A short frame cannot prove the sticks are centered, so it is
            # rejected rather than decoded as neutral.
            raise ValueError(
                "wireless_remote must contain at least "
                f"{DECODABLE_FRAME_BYTES} bytes"
            )
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
StopActive = Callable[[RemoteInput], Awaitable[dict[str, Any]]]
ActiveRunId = Callable[[], str | None]
IsBusy = Callable[[], bool]
IsTakeoverLatched = Callable[[], bool]
ReleaseTakeover = Callable[[], None]
ControllerObserver = Callable[[ControllerSample], Awaitable[object] | object]


class ControllerStartSource(Protocol):
    async def start(self, observer: ControllerObserver) -> None: ...

    async def close(self) -> None: ...

    def status(self) -> dict[str, object]: ...


class ControllerStartAdapter:
    """Turn physical controller input into a cohort request or an instant stop.

    Two intents, kept unambiguous by what the robot is doing at the time:

    * **Idle** -- ``START_PRESS_COUNT`` distinct Start rising edges inside
      ``START_SEQUENCE_WINDOW_S`` request one three-fruit cohort. A held button
      is one press, not many. The counter resets when the window lapses, when
      any other control is touched, on a stop, and after a launch.
    * **Run or cohort active** -- any freshly touched control stops the run at
      once through the safe stop path. Start has no special status here: while
      something is running it stops, and it never counts toward a new sequence.

    The first observed state must be fully neutral -- every button released and
    every stick centered. Consequently a controller held, or a stick leaned on,
    across process or robot boot cannot cause motion or a spurious stop when
    Wendy restores the app.
    """

    def __init__(
        self,
        *,
        start_cohort: StartCohort,
        black_box: RunBlackBox,
        active_run_id: ActiveRunId,
        stop_active: StopActive | None = None,
        is_busy: IsBusy | None = None,
        takeover_latched: IsTakeoverLatched | None = None,
        release_takeover: ReleaseTakeover | None = None,
        stick_deadzone: float = STICK_DEADZONE,
        start_press_count: int = START_PRESS_COUNT,
        start_sequence_window_s: float = START_SEQUENCE_WINDOW_S,
        takeover_release_hold_s: float = TAKEOVER_RELEASE_HOLD_S,
        dedupe_window: int = 128,
        event_history: int = 16,
    ) -> None:
        if dedupe_window < 2:
            raise ValueError("controller dedupe window must be at least two samples")
        if event_history < 1:
            raise ValueError("controller event history must retain at least one edge")
        if not 0.0 < stick_deadzone < 1.0:
            raise ValueError("stick deadzone must be within 0..1 exclusive")
        if start_press_count < 1:
            raise ValueError("at least one Start press is required")
        if start_sequence_window_s <= 0.0:
            raise ValueError("the Start sequence window must be positive")
        if takeover_release_hold_s < 0.0:
            raise ValueError("the takeover release hold must not be negative")
        self._start_cohort = start_cohort
        self._stop_active = stop_active
        self._black_box = black_box
        self._active_run_id = active_run_id
        self._is_busy = (
            is_busy if is_busy is not None else (lambda: active_run_id() is not None)
        )
        self._takeover_latched = takeover_latched or (lambda: False)
        self._release_takeover = release_takeover
        self._stick_deadzone = stick_deadzone
        self._start_press_count = start_press_count
        self._start_sequence_window_s = start_sequence_window_s
        self._takeover_release_hold_s = takeover_release_hold_s
        self._lock = asyncio.Lock()
        # Dropped rather than queued: LowState arrives far faster than a stop
        # completes, so waiting on the lock would pile up thousands of tasks
        # behind one stop. The stop is already under way; the samples add
        # nothing.
        self._in_flight = False
        self._armed_after_release = False
        self._active_controls: frozenset[str] = frozenset()
        self._press_times: list[float] = []
        self._neutral_since: float | None = None
        self._seen_order: deque[tuple[str, int]] = deque()
        self._seen: set[tuple[str, int]] = set()
        self._dedupe_window = dedupe_window
        self._last_decision = "waiting_for_release"
        self._last_source_sequence: int | None = None
        self._accepted_edges = 0
        self._takeover_stops = 0
        self._last_cohort_id: str | None = None
        self._last_error: str | None = None
        # Accepted cohorts have no Run Result yet, so the black box cannot own
        # their attribution.  Keep a bounded edge history readable from status.
        self._events: deque[dict[str, Any]] = deque(maxlen=event_history)

    async def observe(self, sample: ControllerSample) -> ControllerDecision:
        """Consume one sample and take at most one action."""

        if self._in_flight:
            return self._decision("busy")
        async with self._lock:
            identity = (sample.source, sample.source_sequence)
            if identity in self._seen:
                return self._decision("duplicate")
            self._remember(identity)
            self._last_source_sequence = sample.source_sequence

            frame = decode_frame(
                sample.wireless_remote, deadzone=self._stick_deadzone
            )
            previous, controls = self._active_controls, frame.controls
            self._active_controls = controls
            newly_touched = controls - previous
            now = sample.received_monotonic_s

            # 1. Neutral before arm. Anything held as the app boots is inert:
            #    it can neither start a demo nor be mistaken for a takeover.
            if not self._armed_after_release:
                if not frame.neutral:
                    return self._decision("startup_held")
                self._armed_after_release = True
                self._neutral_since = now
                return self._decision("armed")

            # 2. A latched takeover owns the robot until the operator lets go.
            if self._takeover_latched():
                return self._observe_while_latched(frame, now)

            # 3. Something is running: any fresh touch stops it, instantly.
            if self._is_busy():
                self._press_times.clear()
                if newly_touched:
                    return await self._stop_for_input(sample, frame, newly_touched)
                return self._decision("held" if controls else "idle")

            # 4. Idle: only Start, pressed alone, counts toward a launch.
            return await self._observe_while_idle(sample, frame, previous, now)

    def _observe_while_latched(
        self, frame: ControllerFrame, now: float
    ) -> ControllerDecision:
        """Hold everything until the controller has been released and settled."""

        if not frame.neutral:
            self._neutral_since = None
            return self._decision("takeover_latched")
        if self._neutral_since is None:
            self._neutral_since = now
            return self._decision("takeover_latched")
        if now - self._neutral_since < self._takeover_release_hold_s:
            return self._decision("takeover_latched")
        if self._release_takeover is None:
            return self._decision("takeover_latched")
        try:
            self._release_takeover()
        except Exception as exc:  # noqa: BLE001 - never crash on the input path
            self._last_error = f"remote takeover release failed: {exc}"
            return self._decision("takeover_latched")
        self._press_times.clear()
        self._last_error = None
        self._events.append(
            {
                **frame.to_evidence(),
                "disposition": "takeover_released",
                "neutral_hold_s": self._takeover_release_hold_s,
                "received_monotonic_s": now,
            }
        )
        return self._decision("takeover_released")

    async def _observe_while_idle(
        self,
        sample: ControllerSample,
        frame: ControllerFrame,
        previous: frozenset[str],
        now: float,
    ) -> ControllerDecision:
        expired = self._expire_start_sequence(now)
        controls = frame.controls

        if frame.neutral:
            if expired:
                return self._decision("start_sequence_expired")
            return self._decision("released" if previous else "idle")

        # A stick leaned on, or any other button, is not a Start press. It
        # clears the sequence so a half-counted launch cannot hide behind it.
        if controls != frozenset({START_BUTTON_NAME}):
            self._press_times.clear()
            return self._decision("other_input")

        if START_BUTTON_NAME in previous:
            return self._decision("held")

        self._press_times.append(now)
        if len(self._press_times) < self._start_press_count:
            return self._decision("start_press_recorded")
        return await self._launch_cohort(sample, frame)

    async def _launch_cohort(
        self, sample: ControllerSample, frame: ControllerFrame
    ) -> ControllerDecision:
        self._press_times.clear()
        activation_id = f"go2-controller-start:{sample.source}:{sample.source_sequence}"
        evidence: dict[str, Any] = {
            **frame.to_evidence(),
            "button": START_BUTTON_NAME,
            "button_mask": START_BUTTON_MASK,
            "presses_required": self._start_press_count,
            "received_monotonic_s": sample.received_monotonic_s,
            "source": sample.source,
            "source_sequence": sample.source_sequence,
            "start_pressed": True,
        }
        self._in_flight = True
        try:
            cohort = await self._start_cohort()
        except ControllerStartRefused as exc:
            return self._refuse(evidence, activation_id, exc.disposition, exc.detail)
        except Exception as exc:  # noqa: BLE001 - refuse, never crash the app
            return self._refuse(evidence, activation_id, "rejected", str(exc))
        finally:
            self._in_flight = False

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

    async def _stop_for_input(
        self,
        sample: ControllerSample,
        frame: ControllerFrame,
        newly_touched: frozenset[str],
    ) -> ControllerDecision:
        """Stop the active run through the injected safe path. No motion here."""

        control = ",".join(sorted(newly_touched))
        remote_input = RemoteInput(
            source=sample.source,
            control=control,
            received_monotonic_s=sample.received_monotonic_s,
        )
        payload = {
            **frame.to_evidence(),
            "control": control,
            "disposition": "takeover_stop",
            "received_monotonic_s": sample.received_monotonic_s,
            "source": sample.source,
            "source_sequence": sample.source_sequence,
        }
        # Attributed before the stop runs, because the stop seals the Run
        # Result this belongs to.
        run_id = self._active_run_id()
        if run_id is not None:
            try:
                self._black_box.record(
                    run_id,
                    "controller_input",
                    phase="remote_takeover",
                    payload=payload,
                )
            except Exception as exc:  # noqa: BLE001 - attribution is best effort
                self._last_error = f"black-box attribution failed: {exc}"

        if self._stop_active is None:
            self._last_error = "controller stop path is not connected"
            self._events.append({**payload, "disposition": "takeover_unavailable"})
            return self._decision(
                "takeover_unavailable", detail=self._last_error
            )

        self._in_flight = True
        try:
            await self._stop_active(remote_input)
        except Exception as exc:  # noqa: BLE001 - a failed stop must stay visible
            self._last_error = f"controller stop failed: {exc}"
            self._events.append(
                {**payload, "disposition": "takeover_stop_failed", "error": str(exc)}
            )
            return self._decision("takeover_stop_failed", detail=str(exc))
        finally:
            self._in_flight = False
            self._neutral_since = None

        self._takeover_stops += 1
        self._last_error = None
        self._events.append(payload)
        return self._decision("takeover_stop", detail=control)

    def _expire_start_sequence(self, now: float) -> bool:
        """Drop a partial press sequence once its window has lapsed."""
        if not self._press_times:
            return False
        if now - self._press_times[0] <= self._start_sequence_window_s:
            return False
        self._press_times.clear()
        return True

    def status(self) -> dict[str, object]:
        recorded = len(self._press_times)
        latched = bool(self._takeover_latched())
        if not self._armed_after_release:
            detail = "waiting for a neutral controller before activation"
        elif latched:
            detail = (
                "physical remote takeover is latched; release the controller "
                f"for {self._takeover_release_hold_s:g} s to hand control back"
            )
        elif self._is_busy():
            detail = "a run is active; any controller input stops it immediately"
        else:
            detail = (
                f"{recorded} of {self._start_press_count} Start presses recorded; "
                f"press Start {self._start_press_count - recorded} more time(s) "
                "for one Apple + Mango + Pear cohort"
            )
        return {
            "ready": self._armed_after_release,
            "detail": detail,
            "button": START_BUTTON_NAME,
            "button_mask": START_BUTTON_MASK,
            "start_presses_required": self._start_press_count,
            "start_presses_recorded": recorded,
            "start_press_progress": f"{recorded} of {self._start_press_count}",
            "start_sequence_window_s": self._start_sequence_window_s,
            "stick_deadzone": self._stick_deadzone,
            "watched_axes": [name for name, _ in STICK_AXES],
            "takeover": {
                "latched": latched,
                "stops": self._takeover_stops,
                "release_hold_s": self._takeover_release_hold_s,
                "stop_path_connected": self._stop_active is not None,
            },
            "active_controls": sorted(self._active_controls),
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
