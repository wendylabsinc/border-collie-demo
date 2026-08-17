"""Bounded microphone-session reconnection."""

from __future__ import annotations

from collections.abc import Callable

# Why a microphone attempt failed, coarse enough to be stable and specific
# enough to act on. Ordered most specific first; the first match wins.
_FAILURE_SIGNATURES: tuple[tuple[str, tuple[str, ...]], ...] = (
    # Present and streaming, but carrying no signal at all: powered off, muted,
    # or the transmitter is not paired to the receiver.
    ("silent", ("only digital silence",)),
    # Nothing matched the configured spec. On stage this is the common one: the
    # receiver is not plugged in yet.
    ("absent", ("no microphone matched",)),
    # Enumerated but the device would not open: claimed by another process, or
    # an unsupported rate/channel count.
    ("open_failed", ("capture failed to start",)),
    # Frames stopped arriving from a stream that had been running: unplugged
    # mid-session, or the driver went away.
    ("disconnected", ("no microphone frame received",)),
    # The host audio backend itself could not be queried.
    ("discovery_failed", ("microphone discovery failed",)),
)


def classify_microphone_failure(error: str) -> str:
    """Bucket a microphone failure message into a stable kind.

    Kept as a pure function on the message rather than the exception type: the
    supervisor only ever sees the string the session recorded, and the strings
    are the same ones an operator reads in the logs.
    """
    low = (error or "").casefold()
    for kind, needles in _FAILURE_SIGNATURES:
        if any(needle in low for needle in needles):
            return kind
    return "other"


def run_microphone_session_loop(
    *,
    session: Callable[[], None],
    state: dict[str, object],
    retry_interval_s: float,
    sleep: Callable[[float], None],
    should_stop: Callable[[], bool] = lambda: False,
    on_retry: Callable[[str], None] = lambda _error: None,
) -> None:
    """Restart a complete discovery/capture session after every return.

    The session owns the specific hardware error.  This supervisor owns only
    bounded restart cadence and counters, and keeps readiness fail-closed
    between attempts.
    """

    if not 0.25 <= retry_interval_s <= 30.0:
        raise ValueError("retry_interval_s must stay within 0.25..30.0 seconds")

    state.setdefault("attempts", 0)
    state.setdefault("retry_count", 0)

    while not should_stop():
        state["attempts"] = int(state["attempts"]) + 1
        try:
            session()
        except Exception as exc:  # noqa: BLE001 - last-resort thread boundary
            state.update(
                ready=False,
                error=f"microphone session failed unexpectedly: {exc}",
            )

        if should_stop():
            return

        if state.get("ready"):
            state.update(ready=False, error="microphone capture stopped")
        error = str(state.get("error") or "microphone session stopped")
        state["retry_count"] = int(state["retry_count"]) + 1
        on_retry(error)
        sleep(retry_interval_s)
