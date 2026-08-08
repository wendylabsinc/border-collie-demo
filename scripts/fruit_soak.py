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
    python3 scripts/fruit_soak.py --host 192.168.0.107 --runs 10 --seed 7 \
        --note "apple at 94in; banana and pear at 84in" \
        --dongle-match "USB Audio" --device-probes
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

try:  # package import (tests) or direct script execution
    from scripts.stage_scorecard import score_session
except ImportError:  # pragma: no cover - script-invocation path
    from stage_scorecard import score_session

SCHEMA_VERSION = 3
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

    def activate(self, fruit: str) -> dict:
        return self._request(f"{self.base_url}/api/run", "POST", {"target_fruit": fruit})

    def result(self, run_id: str) -> dict:
        return self._request(f"{self.base_url}/api/results/{run_id}")

    def camera_frame(self) -> bytes:
        request = urllib.request.Request(f"{self.base_url}/api/camera/frame.jpg")
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            return response.read()

    def stop(self) -> dict:
        return self._request(f"{self.base_url}/api/stop", "POST")


class TempSource:
    """Optional per-sample temperature reader: an HTTP JSON URL or a command."""

    def __init__(self, url: str | None = None, command: str | None = None) -> None:
        self.url = url
        self.command = command

    @property
    def enabled(self) -> bool:
        return bool(self.url or self.command)

    def read(self) -> tuple[dict | None, str | None]:
        try:
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


def dongle_check(audio_devices: dict | list | None, match: str | None) -> dict:
    """Record voice-dongle visibility from the device audio list."""
    if audio_devices is None:
        return {"checked": False, "visible": None, "detail": "device probes disabled"}
    rendered = json.dumps(audio_devices)
    if isinstance(audio_devices, dict) and "error" in audio_devices:
        return {"checked": False, "visible": None, "detail": audio_devices["error"]}
    result: dict = {"checked": True, "audio_devices": audio_devices}
    if match:
        result["match"] = match
        result["visible"] = match.casefold() in rendered.casefold()
    return result


def draw_fruit_sequence(qualified: list[str], runs: int, seed: int) -> list[str]:
    """Seeded uniform draw so a session's fruit order is reproducible."""
    if not qualified:
        raise HarnessAbort("deployed build reports no qualified fruits")
    rng = random.Random(seed)
    return [rng.choice(sorted(qualified)) for _ in range(runs)]


def take_sample(
    client: ApiClient,
    target_fruit: str,
    temp_source: TempSource,
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
    if temp_source.enabled:
        temps, temp_error = temp_source.read()
        if temps is not None:
            sample["temps"] = temps
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


def summarize_run(run: dict, fruit: str, number: int) -> dict:
    terminal = run.get("terminal_measurements") or {}
    return {
        "number": number,
        "target_fruit": fruit,
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
    temp_source: TempSource | None = None,
    timeout_s: float = RUN_TIMEOUT_S,
    sleep=time.sleep,
    clock=time.monotonic,
) -> tuple[dict, list[dict], str | None]:
    """Poll one run to terminal state, sampling telemetry on every tick."""
    temp_source = temp_source or TempSource()
    deadline = clock() + timeout_s
    samples: list[dict] = []
    while True:
        samples.append(take_sample(client, target_fruit, temp_source, clock=clock))
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
    keep_samples: bool = True,
    sleep=time.sleep,
    log=print,
) -> dict:
    temp_source = temp_source or TempSource()
    status = wait_for_ready(client)
    build_label = status.get("build_label", "unlabelled")
    qualified = list(client.fruits().get("qualified_fruits", []))
    sequence = draw_fruit_sequence(qualified, runs, seed)
    log(f"build: {build_label}")
    log(f"qualified fruits: {', '.join(qualified)}")
    log(f"seed {seed} -> sequence: {', '.join(sequence)}")

    session: dict = {
        "schema_version": SCHEMA_VERSION,
        "session": f"fruit-soak-{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H%M%SZ')}",
        "build_label": build_label,
        "note": note,
        "qualified_fruits": qualified,
        "target_runs": runs,
        "seed": seed,
        "fruit_sequence": sequence,
        "temperature_source": temp_source.url or temp_source.command,
        "runs": [],
        "aborted": None,
    }

    frames_dir = output_path.with_name(output_path.stem + "-frames")

    def persist() -> None:
        session["scorecard"] = score_session(session)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(session, indent=2) + "\n")

    persist()
    try:
        for number, fruit in enumerate(sequence, start=1):
            if number > 1:
                sleep(cooldown_s)
            wait_for_ready(client)
            preflight: dict = {}
            if device_probe is not None:
                preflight["dongle"] = dongle_check(
                    device_probe.audio_devices(), dongle_match
                )
                preflight["wifi_before"] = device_probe.wifi_status()
            lighting_frame = capture_lighting_frame(client, frames_dir, number)
            log(f"run {number}/{runs}: activating {fruit}")
            run_id = client.activate(fruit)["run"]["run_id"]
            run, samples, harness_note = wait_for_terminal(
                client, run_id, target_fruit=fruit, temp_source=temp_source
            )
            record = summarize_run(run, fruit, number)
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
    parser.add_argument("--temp-url", default=None, help="HTTP JSON endpoint of temperatures")
    parser.add_argument("--temp-cmd", default=None, help="shell command printing temperature JSON")
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
    try:
        session = run_session(
            client,
            runs=args.runs,
            seed=seed,
            output_path=output,
            cooldown_s=args.cooldown,
            temp_source=TempSource(url=args.temp_url, command=args.temp_cmd),
            device_probe=DeviceProbe(agent, args.device_probes),
            dongle_match=args.dongle_match,
            note=args.note,
            keep_samples=not args.no_samples,
        )
    except urllib.error.URLError as exc:
        print(f"cannot reach the demo app: {exc}", file=sys.stderr)
        return 2
    return 0 if session["aborted"] is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
