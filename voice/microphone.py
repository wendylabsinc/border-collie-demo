"""Bounded microphone-session reconnection."""

from __future__ import annotations

from collections.abc import Callable


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
