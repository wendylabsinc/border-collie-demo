"""Supervised fruit-soak harness: randomized qualified-fruit Demo Runs.

Runs from the operator's machine against a deployed demo and records every
result. It never selects a fruit the deployed build does not qualify: the
sequence is drawn from the live ``/api/fruits`` qualified list, so the same
harness is valid on base (pear/apple/banana) and edge builds.

Beyond outcomes, each run records an in-run telemetry series sampled every
poll tick and aggregated per stage:

- fruit-detection confidence and bounding-box bottom ratio ("how close we
  thought") from the perception sidecar, tagged with the stage in play
- app-poll HTTP latency and errors, which double as the Wi-Fi cutout/slowdown
  record from the operator vantage
- device temperatures, when a ``--temp-url`` or ``--temp-cmd`` source is given
- voice-dongle visibility (``wendy device audio list``) and device Wi-Fi
  status before each run, when ``--device-probes`` is enabled

An operator must supervise the robot for the whole session. The harness is a
trigger-and-recorder; every safety behavior (preflight, watchdogs, takeover,
fail-closed camera rules) belongs to the deployed application, and a latched
restart-required state aborts the session immediately.

Usage:
    python3 scripts/fruit_soak.py --host 192.168.0.107 --runs 10 --seed 20260810 \
        --expected-build-label "base-soak-v2-orientation (demo/base)" \
        --expected-fruits apple banana pear \
        --note "apple at 94in; banana and pear at 84in" \
        --dongle-match "DJI MIC MINI" --device-probes
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

try:  # package import (tests) or direct script execution
    from scripts.stage_scorecard import score_session
except ImportError:  # pragma: no cover - script-invocation path
    from stage_scorecard import score_session

SCHEMA_VERSION = 4
POLL_INTERVAL_S = 1.5
READY_TIMEOUT_S = 90.0
RUN_TIMEOUT_S = 180.0
COOLDOWN_S = 10.0


class HarnessAbort(RuntimeError):
    """Session-ending condition that is not an individual run failure."""


class ApiClient:
    """Minimal JSON client for the demo app and its perception sidecar."""

    def __init__(self, base_url: str, sidecar_url: str, timeout_s: float = 8.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.sidecar_url = sidecar_url.rstrip("/")
        self.timeout_s = timeout_s

    def _request(self, url: str, method: str = "GET", payload: dict | None = None) -> dict:
        body = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(
            url, data=body, method=method, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            return json.loads(response.read())

    def status(self) -> dict:
        return self._request(f"{self.base_url}/api/status")

    def timed_status(self) -> tuple[dict | None, float, str | None]:
        """Status plus round-trip latency; errors are data, not exceptions."""
        started = time.monotonic()
        try:
            payload = self.status()
            return payload, (time.monotonic() - started) * 1000.0, None
        except Exception as exc:  # noqa: BLE001 - sampled link failures are data
            return None, (time.monotonic() - started) * 1000.0, str(exc)

    def sidecar_status(self) -> tuple[dict | None, str | None]:
        try:
            return self._request(f"{self.sidecar_url}/status"), None
        except Exception as exc:  # noqa: BLE001 - sampled link failures are data
            return None, str(exc)

    def fruits(self) -> dict:
        return self._request(f"{self.base_url}/api/fruits")

    def activate(self, fruit: str, orientation_degrees: float = 0.0) -> dict:
        return self._request(
            f"{self.base_url}/api/run",
            "POST",
            {
                "target_fruit": fruit,
                "orientation_degrees": orientation_degrees,
            },
        )

    def result(self, run_id: str) -> dict:
        return self._request(f"{self.base_url}/api/results/{run_id}")

    def camera_frame(self) -> bytes:
        request = urllib.request.Request(f"{self.base_url}/api/camera/frame.jpg")
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            return response.read()

    def stop(self) -> dict:
        return self._request(f"{self.base_url}/api/stop", "POST")


class TempSource:
    """Optional temperature reader: an HTTP JSON URL, a command, or the
    device's thermal zones through ``wendy device top`` (agent mode)."""

    def __init__(
        self,
        url: str | None = None,
        command: str | None = None,
        agent: str | None = None,
    ) -> None:
        self.url = url
        self.command = command
        self.agent = agent

    @property
    def enabled(self) -> bool:
        return bool(self.url or self.command or self.agent)

    def read(self) -> tuple[dict | None, str | None]:
        try:
            if self.agent:
                completed = subprocess.run(
                    ["wendy", "device", "top", "--device", self.agent, "--json"],
                    capture_output=True,
                    timeout=20.0,
                    text=True,
                )
                if completed.returncode != 0:
                    return None, completed.stderr.strip()[:200] or "wendy top failed"
                zones = (
                    json.loads(completed.stdout).get("host", {}).get("thermalZones")
                    or []
                )
                temps = {
                    z["name"]: z["tempC"]
                    for z in zones
                    if isinstance(z.get("tempC"), (int, float))
                }
                return (temps or None), (None if temps else "no thermal zones reported")
            if self.url:
                with urllib.request.urlopen(self.url, timeout=4.0) as response:
                    return json.loads(response.read()), None
            if self.command:
                completed = subprocess.run(
                    self.command, shell=True, capture_output=True, timeout=8.0, text=True
                )
                if completed.returncode != 0:
                    return None, completed.stderr.strip() or "temp command failed"
                return json.loads(completed.stdout), None
        except Exception as exc:  # noqa: BLE001 - sampled probe failures are data
            return None, str(exc)
        return None, None


class ThreadedTempSampler:
    """Samples a slow TempSource on its own thread so the poll loop never
    blocks on it. Runs across the whole session, so cooldown-period cooling
    between runs is captured too. ``latest()`` falls back to a synchronous
    read when the thread was never started (tests, one-shot use)."""

    def __init__(self, source: TempSource, interval_s: float = 10.0) -> None:
        self.source = source
        self.interval_s = interval_s
        self._lock = threading.Lock()
        self._latest: tuple[dict | None, str | None, float | None] = (None, None, None)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def enabled(self) -> bool:
        return self.source.enabled

    def _read_once(self) -> None:
        temps, error = self.source.read()
        with self._lock:
            self._latest = (temps, error, time.monotonic())

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._read_once()
            self._stop.wait(self.interval_s)

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def latest(self) -> tuple[dict | None, str | None, float | None]:
        if self.enabled and self._thread is None and self._latest[2] is None:
            self._read_once()
        with self._lock:
            temps, error, read_at = self._latest
        age_s = None if read_at is None else time.monotonic() - read_at
        return temps, error, age_s


class DeviceProbe:
    """Pre/post-run device checks through the wendy CLI (dongle, Wi-Fi)."""

    def __init__(self, agent: str, enabled: bool) -> None:
        self.agent = agent
        self.enabled = enabled and shutil.which("wendy") is not None

    def _cli_json(self, *args: str) -> dict | list | None:
        try:
            completed = subprocess.run(
                ["wendy", *args, "--device", self.agent, "--json"],
                capture_output=True,
                timeout=20.0,
                text=True,
            )
            if completed.returncode != 0:
                return {"error": completed.stderr.strip()[:300] or "wendy CLI failed"}
            return json.loads(completed.stdout)
        except Exception as exc:  # noqa: BLE001 - probe failures are data
            return {"error": str(exc)[:300]}

    def audio_devices(self) -> dict | list | None:
        return self._cli_json("device", "audio", "list") if self.enabled else None

    def wifi_status(self) -> dict | list | None:
        return self._cli_json("device", "wifi", "status") if self.enabled else None

    def usb_devices(self) -> list | dict | None:
        """USB device descriptions from the hardware capability list. The
        voice dongle enumerates here (not in the ALSA audio list), and so
        does the Wi-Fi adapter."""
        if not self.enabled:
            return None
        listing = self._cli_json("device", "hardware", "list")
        if isinstance(listing, dict):  # error payload
            return listing
        return [
            entry.get("description")
            for entry in listing or []
            if isinstance(entry, dict) and entry.get("category") == "usb"
        ]


def dongle_check(sources: dict | None, match: str | None) -> dict:
    """Record voice-dongle visibility across the device's USB and audio lists."""
    if sources is None:
        return {"checked": False, "visible": None, "detail": "device probes disabled"}
    errors = {
        name: payload["error"]
        for name, payload in sources.items()
        if isinstance(payload, dict) and "error" in payload
    }
    if errors and len(errors) == len(sources):
        return {"checked": False, "visible": None, "detail": "; ".join(errors.values())}
    result: dict = {"checked": True, **sources}
    if errors:
        result["probe_errors"] = errors
    if match:
        result["match"] = match
        rendered = json.dumps(
            {k: v for k, v in sources.items() if k not in errors}
        )
        result["visible"] = match.casefold() in rendered.casefold()
    return result


def draw_fruit_sequence(qualified: list[str], runs: int, seed: int) -> list[str]:
    """Build a seeded, balanced sequence with a reproducibly random order.

    Balance matters for acceptance: ten independent choices can schedule one
    of three fruits only once, making the scorecard's two-successes-per-fruit
    criterion impossible before the robot moves. Every fruit therefore gets
    either ``runs // fruit_count`` or one additional attempt, while the extra
    slots and final order remain seeded and random.
    """
    if not qualified:
        raise HarnessAbort("deployed build reports no qualified fruits")
    if runs <= 0:
        raise HarnessAbort("run count must be greater than zero")
    fruits = sorted(set(qualified))
    rng = random.Random(seed)
    repetitions, remainder = divmod(runs, len(fruits))
    sequence = fruits * repetitions
    sequence.extend(rng.sample(fruits, remainder))
    rng.shuffle(sequence)
    return sequence


def draw_orientation_sequence(runs: int, seed: int) -> list[int]:
    """Draw reproducible integer headings spanning the full [0, 360) circle."""
    if runs <= 0:
        raise HarnessAbort("run count must be greater than zero")
    rng = random.Random(seed ^ 0xC0111E)
    return [rng.randrange(360) for _ in range(runs)]


def take_sample(
    client: ApiClient,
    target_fruit: str,
    temp_sampler: "ThreadedTempSampler | None",
    *,
    clock=time.monotonic,
) -> dict:
    """One telemetry tick: stage, confidence, proximity, link, temperature."""
    status, latency_ms, app_error = client.timed_status()
    mission = (status or {}).get("mission") or {}
    sample: dict = {
        "t_monotonic_s": clock(),
        "phase": mission.get("phase"),
        "app_latency_ms": round(latency_ms, 1),
        "app_error": app_error,
    }
    sidecar, sidecar_error = client.sidecar_status()
    if sidecar_error:
        sample["sidecar_error"] = sidecar_error
    elif sidecar is not None:
        detection = sidecar.get("detection") or {}
        source = sidecar.get("source") or {}
        height = source.get("height")
        bbox = detection.get("bbox_xyxy")
        sample["detection_label"] = detection.get("label")
        sample["confidence"] = detection.get("confidence")
        sample["inference_s"] = detection.get("inference_s")
        sample["target_matches"] = detection.get("label") == target_fruit
        if bbox and height:
            sample["bbox_bottom_ratio"] = round(float(bbox[3]) / float(height), 4)
        route = detection.get("route")
        if route is not None:
            sample["route"] = route
    if temp_sampler is not None and temp_sampler.enabled:
        temps, temp_error, age_s = temp_sampler.latest()
        if temps is not None:
            sample["temps"] = temps
            if age_s is not None:
                sample["temps_age_s"] = round(age_s, 1)
        if temp_error:
            sample["temp_error"] = temp_error
    return sample


def _stats(values: list[float]) -> dict | None:
    if not values:
        return None
    return {
        "min": round(min(values), 4),
        "mean": round(sum(values) / len(values), 4),
        "max": round(max(values), 4),
        "samples": len(values),
    }


def aggregate_stage_telemetry(samples: list[dict]) -> dict:
    """Group the sample series by stage and reduce each stage's signals."""
    stages: dict[str, dict] = {}
    for sample in samples:
        phase = sample.get("phase") or "unknown"
        bucket = stages.setdefault(
            phase,
            {
                "confidences": [],
                "bottom_ratios": [],
                "latencies": [],
                "errors": 0,
                "temp_series": {},
            },
        )
        if sample.get("target_matches") and sample.get("confidence") is not None:
            bucket["confidences"].append(float(sample["confidence"]))
        if sample.get("target_matches") and sample.get("bbox_bottom_ratio") is not None:
            bucket["bottom_ratios"].append(float(sample["bbox_bottom_ratio"]))
        bucket["latencies"].append(float(sample["app_latency_ms"]))
        if sample.get("app_error"):
            bucket["errors"] += 1
        for zone, value in (sample.get("temps") or {}).items():
            if isinstance(value, (int, float)):
                bucket["temp_series"].setdefault(zone, []).append(float(value))

    aggregated: dict[str, dict] = {}
    for phase, bucket in stages.items():
        aggregated[phase] = {
            "target_confidence": _stats(bucket["confidences"]),
            "bbox_bottom_ratio": _stats(bucket["bottom_ratios"]),
            "app_latency_ms": _stats(bucket["latencies"]),
            "app_errors": bucket["errors"],
            "temps_c": {
                zone: _stats(series) for zone, series in bucket["temp_series"].items()
            }
            or None,
        }
    return aggregated


def summarize_network(samples: list[dict]) -> dict:
    """Session-facing Wi-Fi verdict: cutouts (errors) and slowdowns (latency)."""
    latencies = sorted(s["app_latency_ms"] for s in samples if s.get("app_error") is None)
    errors = [
        {"t_monotonic_s": s["t_monotonic_s"], "error": s["app_error"]}
        for s in samples
        if s.get("app_error")
    ]
    p95 = latencies[max(0, int(len(latencies) * 0.95) - 1)] if latencies else None
    return {
        "poll_count": len(samples),
        "error_count": len(errors),
        "errors": errors,
        "latency_ms": _stats(list(latencies)),
        "latency_p95_ms": p95,
        "cut_out": bool(errors),
    }


def compute_stage_durations(events: list[dict] | None) -> dict[str, float]:
    """Per-stage wall time from the run's phase-transition events.

    Acquisition time (turn/search stages) is the most lighting-sensitive
    number in a run, so it gets its own record instead of hiding inside the
    total duration.
    """
    if not events:
        return {}
    transitions = [
        (e.get("phase"), e.get("seconds_since_start"))
        for e in events
        if e.get("phase") is not None and e.get("seconds_since_start") is not None
    ]
    durations: dict[str, float] = {}
    for (phase, started), (_, ended) in zip(transitions, transitions[1:]):
        durations[phase] = round(durations.get(phase, 0.0) + (ended - started), 3)
    return durations


def capture_lighting_frame(
    client: ApiClient, frames_dir: Path, number: int
) -> dict:
    """Save one camera frame at run start as the run's lighting evidence."""
    try:
        payload = client.camera_frame()
    except Exception as exc:  # noqa: BLE001 - evidence failures are data
        return {"error": str(exc)}
    frames_dir.mkdir(parents=True, exist_ok=True)
    path = frames_dir / f"run-{number:02d}-start.jpg"
    path.write_bytes(payload)
    return {"path": str(path), "bytes": len(payload)}


def summarize_run(
    run: dict,
    fruit: str,
    number: int,
    orientation_degrees: float = 0.0,
) -> dict:
    terminal = run.get("terminal_measurements") or {}
    return {
        "number": number,
        "target_fruit": fruit,
        "orientation_degrees": float(
            run.get("orientation_degrees", orientation_degrees)
        ),
        "run_id": run.get("run_id"),
        "started_at_utc": run.get("started_at_utc"),
        "ended_at_utc": run.get("ended_at_utc"),
        "duration_s": run.get("duration_s"),
        "outcome": run.get("outcome"),
        "reason": run.get("reason"),
        "message": run.get("message"),
        "failed_phase": run.get("failed_phase"),
        "final_safety_state": run.get("final_safety_state"),
        "home_distance_m": terminal.get("home_distance_m"),
        "heading_error_rad": terminal.get("heading_error_rad"),
        "stage_results": run.get("stage_results"),
        "failure_details": run.get("failure_details"),
        "stage_durations": compute_stage_durations(run.get("events")),
    }


def wait_for_ready(
    client: ApiClient,
    *,
    timeout_s: float = READY_TIMEOUT_S,
    sleep=time.sleep,
    clock=time.monotonic,
) -> dict:
    """Poll until the app can accept an activation; abort on latched states."""
    deadline = clock() + timeout_s
    last_blockers: list[str] = []
    while True:
        status = client.status()
        mission = status.get("mission") or {}
        if mission.get("restart_required"):
            raise HarnessAbort(
                "application latched restart-required "
                f"({mission.get('reason')}); session cannot continue"
            )
        activation = status.get("activation") or {}
        if activation.get("ready") and not status.get("active_run_id"):
            return status
        last_blockers = [
            f"{item.get('name')}: {item.get('detail')}"
            for item in activation.get("blockers", [])
        ]
        if clock() >= deadline:
            raise HarnessAbort(
                f"activation not ready within {timeout_s:.0f}s; "
                f"blockers: {'; '.join(last_blockers) or 'unknown'}"
            )
        sleep(POLL_INTERVAL_S)


def wait_for_terminal(
    client: ApiClient,
    run_id: str,
    *,
    target_fruit: str,
    temp_sampler: "ThreadedTempSampler | None" = None,
    timeout_s: float = RUN_TIMEOUT_S,
    sleep=time.sleep,
    clock=time.monotonic,
) -> tuple[dict, list[dict], str | None]:
    """Poll one run to terminal state, sampling telemetry on every tick."""
    deadline = clock() + timeout_s
    samples: list[dict] = []
    while True:
        samples.append(take_sample(client, target_fruit, temp_sampler, clock=clock))
        run = client.result(run_id).get("run") or {}
        if run.get("outcome"):
            return run, samples, None
        if clock() >= deadline:
            client.stop()
            note = f"harness stop issued after {timeout_s:.0f}s without a terminal state"
            run = client.result(run_id).get("run") or run
            return run, samples, note
        sleep(POLL_INTERVAL_S)


def run_session(
    client: ApiClient,
    *,
    runs: int,
    seed: int,
    output_path: Path,
    cooldown_s: float = COOLDOWN_S,
    temp_source: TempSource | None = None,
    device_probe: DeviceProbe | None = None,
    dongle_match: str | None = None,
    note: str | None = None,
    expected_build_label: str | None = None,
    expected_fruits: list[str] | None = None,
    keep_samples: bool = True,
    sleep=time.sleep,
    log=print,
) -> dict:
    temp_sampler = ThreadedTempSampler(temp_source or TempSource())
    status = wait_for_ready(client)
    build_label = status.get("build_label", "unlabelled")
    if expected_build_label is not None and build_label != expected_build_label:
        raise HarnessAbort(
            f"expected build {expected_build_label!r}, got {build_label!r}; "
            "no run was activated"
        )
    qualified = list(client.fruits().get("qualified_fruits", []))
    if expected_fruits is not None:
        expected = sorted(set(expected_fruits))
        actual = sorted(set(qualified))
        if actual != expected:
            raise HarnessAbort(
                f"expected qualified fruits {expected!r}, got {actual!r}; "
                "no run was activated"
            )
    sequence = draw_fruit_sequence(qualified, runs, seed)
    orientation_sequence = draw_orientation_sequence(runs, seed)
    log(f"build: {build_label}")
    log(f"qualified fruits: {', '.join(qualified)}")
    log(f"seed {seed} -> sequence: {', '.join(sequence)}")
    log(
        "orientations: "
        + ", ".join(f"{angle}\N{DEGREE SIGN}" for angle in orientation_sequence)
    )

    session: dict = {
        "schema_version": SCHEMA_VERSION,
        "session": f"fruit-soak-{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H%M%SZ')}",
        "build_label": build_label,
        "note": note,
        "qualified_fruits": qualified,
        "target_runs": runs,
        "seed": seed,
        "fruit_sequence": sequence,
        "orientation_sequence_degrees": orientation_sequence,
        "temperature_source": (
            temp_sampler.source.url
            or temp_sampler.source.command
            or (f"wendy device top ({temp_sampler.source.agent})"
                if temp_sampler.source.agent else None)
        ),
        "runs": [],
        "aborted": None,
    }

    frames_dir = output_path.with_name(output_path.stem + "-frames")

    def persist() -> None:
        session["scorecard"] = score_session(session)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
        with temporary_path.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(session, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)

    persist()
    temp_sampler.start()
    try:
        for number, (fruit, orientation_degrees) in enumerate(
            zip(sequence, orientation_sequence, strict=True),
            start=1,
        ):
            if number > 1:
                sleep(cooldown_s)
            wait_for_ready(client)
            preflight: dict = {}
            if device_probe is not None:
                preflight["dongle"] = dongle_check(
                    {
                        "usb_devices": device_probe.usb_devices(),
                        "audio_devices": device_probe.audio_devices(),
                    },
                    dongle_match,
                )
                preflight["wifi_before"] = device_probe.wifi_status()
            lighting_frame = capture_lighting_frame(client, frames_dir, number)
            log(
                f"run {number}/{runs}: activating {fruit} "
                f"after {orientation_degrees}\N{DEGREE SIGN} orientation turn"
            )
            try:
                run_id = client.activate(fruit, orientation_degrees)["run"]["run_id"]
            except Exception as exc:
                raise HarnessAbort(
                    f"run {number} activation outcome is ambiguous; "
                    f"no automatic retry will be attempted: {exc}"
                ) from exc
            run, samples, harness_note = wait_for_terminal(
                client, run_id, target_fruit=fruit, temp_sampler=temp_sampler
            )
            record = summarize_run(run, fruit, number, orientation_degrees)
            record["lighting_frame"] = lighting_frame
            record["stage_telemetry"] = aggregate_stage_telemetry(samples)
            record["network"] = summarize_network(samples)
            if keep_samples:
                record["samples"] = samples
            if preflight:
                record["preflight_probes"] = preflight
            if device_probe is not None:
                record["wifi_after"] = device_probe.wifi_status()
            if harness_note:
                record["harness_note"] = harness_note
            session["runs"].append(record)
            persist()
            log(
                f"run {number}/{runs}: {record['outcome']} / {record['reason']}"
                f" | home {record['home_distance_m']}"
                f" | polls {record['network']['poll_count']}"
                f" (errors {record['network']['error_count']})"
            )
    except HarnessAbort as abort:
        session["aborted"] = str(abort)
        persist()
        log(f"session aborted: {abort}")
    except KeyboardInterrupt:
        session["aborted"] = "operator interrupt"
        persist()
        log("operator interrupt; requesting robot stop")
        client.stop()
        raise
    finally:
        temp_sampler.stop()

    completed = sum(1 for r in session["runs"] if r["outcome"] == "COMPLETED")
    log(f"recorded {len(session['runs'])} runs ({completed} COMPLETED) -> {output_path}")
    return session


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="192.168.0.107")
    parser.add_argument("--port", type=int, default=8110)
    parser.add_argument("--sidecar-port", type=int, default=8111)
    parser.add_argument("--agent", default=None, help="wendy agent host:port for device probes")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None, help="default: current epoch seconds")
    parser.add_argument("--cooldown", type=float, default=COOLDOWN_S)
    parser.add_argument("--note", default=None, help="session context, e.g. fruit placements")
    parser.add_argument(
        "--expected-build-label",
        default=None,
        help=(
            "abort before activation unless /api/status reports this exact "
            "build label"
        ),
    )
    parser.add_argument(
        "--expected-fruits",
        nargs="+",
        default=None,
        help="abort before activation unless these are the exact qualified fruits",
    )
    parser.add_argument("--temp-url", default=None, help="HTTP JSON endpoint of temperatures")
    parser.add_argument("--temp-cmd", default=None, help="shell command printing temperature JSON")
    parser.add_argument("--no-temps", action="store_true", help="disable temperature sampling")
    parser.add_argument("--device-probes", action="store_true", help="enable wendy CLI probes")
    parser.add_argument("--dongle-match", default=None, help="substring marking the voice dongle")
    parser.add_argument("--no-samples", action="store_true", help="omit raw sample series")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    seed = args.seed if args.seed is not None else int(time.time())
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    output = args.output or Path("benchmarks/results") / f"fruit-soak-{stamp}.json"
    client = ApiClient(
        f"http://{args.host}:{args.port}", f"http://{args.host}:{args.sidecar_port}"
    )
    agent = args.agent or f"{args.host}:50052"
    temp_agent = None
    if not args.no_temps and not args.temp_url and not args.temp_cmd:
        temp_agent = agent  # default: device thermal zones via wendy top
    try:
        session = run_session(
            client,
            runs=args.runs,
            seed=seed,
            output_path=output,
            cooldown_s=args.cooldown,
            temp_source=TempSource()
            if args.no_temps
            else TempSource(url=args.temp_url, command=args.temp_cmd, agent=temp_agent),
            device_probe=DeviceProbe(agent, args.device_probes),
            dongle_match=args.dongle_match,
            note=args.note,
            expected_build_label=args.expected_build_label,
            expected_fruits=args.expected_fruits,
            keep_samples=not args.no_samples,
        )
    except HarnessAbort as exc:
        print(f"session aborted: {exc}", file=sys.stderr)
        return 1
    except urllib.error.URLError as exc:
        print(f"cannot reach the demo app: {exc}", file=sys.stderr)
        return 2
    return 0 if session["aborted"] is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
