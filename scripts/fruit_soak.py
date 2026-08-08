"""Supervised fruit-soak harness: randomized qualified-fruit Demo Runs.

Runs from the operator's machine against a deployed demo and records every
result. It never selects a fruit the deployed build does not qualify: the
sequence is drawn from the live ``/api/fruits`` qualified list, so the same
harness is valid on base (pear/apple) and edge (pear/apple/banana) builds.

An operator must supervise the robot for the whole session. The harness is a
trigger-and-recorder; every safety behavior (preflight, watchdogs, takeover,
fail-closed camera rules) belongs to the deployed application, and a latched
restart-required state aborts the session immediately.

Usage:
    python3 scripts/fruit_soak.py --host 192.168.0.107 --runs 10 --seed 7
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1
POLL_INTERVAL_S = 2.0
READY_TIMEOUT_S = 90.0
RUN_TIMEOUT_S = 180.0
COOLDOWN_S = 10.0


class HarnessAbort(RuntimeError):
    """Session-ending condition that is not an individual run failure."""


class ApiClient:
    """Minimal JSON client for the demo API."""

    def __init__(self, base_url: str, timeout_s: float = 8.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def _request(self, method: str, path: str, payload: dict | None = None) -> dict:
        body = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            return json.loads(response.read())

    def status(self) -> dict:
        return self._request("GET", "/api/status")

    def fruits(self) -> dict:
        return self._request("GET", "/api/fruits")

    def activate(self, fruit: str) -> dict:
        return self._request("POST", "/api/run", {"target_fruit": fruit})

    def result(self, run_id: str) -> dict:
        return self._request("GET", f"/api/results/{run_id}")

    def stop(self) -> dict:
        return self._request("POST", "/api/stop")


def draw_fruit_sequence(qualified: list[str], runs: int, seed: int) -> list[str]:
    """Seeded uniform draw so a session's fruit order is reproducible."""
    if not qualified:
        raise HarnessAbort("deployed build reports no qualified fruits")
    rng = random.Random(seed)
    return [rng.choice(sorted(qualified)) for _ in range(runs)]


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
    timeout_s: float = RUN_TIMEOUT_S,
    sleep=time.sleep,
    clock=time.monotonic,
) -> tuple[dict, str | None]:
    """Poll one run to its terminal state; stop the robot if it overruns."""
    deadline = clock() + timeout_s
    while True:
        run = client.result(run_id).get("run") or {}
        if run.get("outcome"):
            return run, None
        if clock() >= deadline:
            client.stop()
            note = f"harness stop issued after {timeout_s:.0f}s without a terminal state"
            run = client.result(run_id).get("run") or run
            return run, note
        sleep(POLL_INTERVAL_S)


def run_session(
    client: ApiClient,
    *,
    runs: int,
    seed: int,
    output_path: Path,
    cooldown_s: float = COOLDOWN_S,
    sleep=time.sleep,
    log=print,
) -> dict:
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
        "qualified_fruits": qualified,
        "target_runs": runs,
        "seed": seed,
        "fruit_sequence": sequence,
        "runs": [],
        "aborted": None,
    }

    def persist() -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(session, indent=2) + "\n")

    persist()
    try:
        for number, fruit in enumerate(sequence, start=1):
            if number > 1:
                sleep(cooldown_s)
            wait_for_ready(client)
            log(f"run {number}/{runs}: activating {fruit}")
            run_id = client.activate(fruit)["run"]["run_id"]
            run, harness_note = wait_for_terminal(client, run_id)
            record = summarize_run(run, fruit, number)
            if harness_note:
                record["harness_note"] = harness_note
            session["runs"].append(record)
            persist()
            log(
                f"run {number}/{runs}: {record['outcome']} / {record['reason']}"
                f" | home {record['home_distance_m']}"
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
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None, help="default: current epoch seconds")
    parser.add_argument("--cooldown", type=float, default=COOLDOWN_S)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    seed = args.seed if args.seed is not None else int(time.time())
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    output = args.output or Path("benchmarks/results") / f"fruit-soak-{stamp}.json"
    client = ApiClient(f"http://{args.host}:{args.port}")
    try:
        session = run_session(
            client,
            runs=args.runs,
            seed=seed,
            output_path=output,
            cooldown_s=args.cooldown,
        )
    except urllib.error.URLError as exc:
        print(f"cannot reach the demo app: {exc}", file=sys.stderr)
        return 2
    return 0 if session["aborted"] is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
