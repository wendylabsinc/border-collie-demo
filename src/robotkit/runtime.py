"""Common process lifecycle helpers; no domain state lives here."""

from __future__ import annotations

import logging
import os
import signal
import socket
import time
from collections.abc import Callable


def configure_logging() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def instance_id() -> str:
    return os.getenv("ROBOTKIT_INSTANCE_ID", socket.gethostname())


def deployment_generation() -> int:
    return int(os.getenv("ROBOTKIT_DEPLOYMENT_GENERATION", "0"))


def world_state_url() -> str:
    explicit = os.getenv("WORLD_STATE_URL")
    if explicit:
        return explicit
    device = os.getenv("WENDY_DEVICE_HOSTNAME")
    return f"http://{device}:8080" if device else "http://world-state:8080"


def run_loop(step: Callable[[], None], interval_seconds: float) -> None:
    stopping = False

    def stop(*_: object) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    while not stopping:
        started = time.monotonic()
        try:
            step()
        except Exception:
            logging.getLogger("robotkit.runtime").exception("worker iteration failed")
        remaining = interval_seconds - (time.monotonic() - started)
        if remaining > 0:
            time.sleep(remaining)
