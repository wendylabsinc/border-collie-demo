"""Replay a ROS2 bag through perception containers and check observations."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Sequence


PERCEPTION_SERVICES = frozenset(
    {"yolo-fruits", "lidar-voxel", "transcription", "health-high-low"}
)


def _ros_domain_id(value: str) -> int:
    domain_id = int(value)
    if not 0 <= domain_id <= 232:
        raise argparse.ArgumentTypeError("must be between 0 and 232")
    return domain_id


def is_json_subset(actual: Any, expected: Any) -> bool:
    """Return whether ``expected`` recursively describes part of ``actual``.

    Expected lists are unordered subsets. This makes it possible to assert one
    detection or proximity sector without copying every item emitted by a bag.
    """
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and is_json_subset(actual[key], value)
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return False
        if not expected:
            return not actual
        return all(
            any(is_json_subset(item, wanted) for item in actual) for wanted in expected
        )
    return actual == expected


def unmatched_expectations(
    observations: Sequence[dict[str, Any]], expectations: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Return expectations that no current observation satisfies."""
    return [
        expected
        for expected in expectations
        if not any(is_json_subset(observation, expected) for observation in observations)
    ]


def load_expectations(
    streams: Sequence[str], expectation_file: Path | None
) -> list[dict[str, Any]]:
    expectations: list[dict[str, Any]] = [{"stream": stream} for stream in streams]
    if expectation_file is None:
        return expectations

    value = json.loads(expectation_file.read_text())
    if isinstance(value, dict):
        value = value.get("observations")
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(
            "expectation file must be a JSON list, or an object with an "
            "'observations' list"
        )
    for item in value:
        if not isinstance(item.get("stream"), str):
            raise ValueError("every observation expectation must contain a stream")
    expectations.extend(value)
    return expectations


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path, help="ROS2 bag directory or MCAP file")
    parser.add_argument(
        "--service",
        action="append",
        required=True,
        choices=sorted(PERCEPTION_SERVICES),
        help="perception service to test; repeat for multiple services",
    )
    parser.add_argument(
        "--expect",
        action="append",
        default=[],
        metavar="STREAM",
        help="observation stream that must be published; repeat as needed",
    )
    parser.add_argument(
        "--expect-file",
        type=Path,
        help="JSON observation subset expectations",
    )
    parser.add_argument("--rate", type=float, default=1.0, help="bag playback rate")
    parser.add_argument(
        "--discovery-seconds",
        type=float,
        default=2.0,
        help="delay after bag publishers start so DDS subscribers can match",
    )
    parser.add_argument(
        "--startup-seconds",
        type=float,
        default=5.0,
        help="time for ROS subscriptions and lazy models to start before playback",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="time to wait for observations after playback",
    )
    parser.add_argument("--ros-domain-id", type=_ros_domain_id)
    parser.add_argument(
        "--keep",
        action="store_true",
        help="leave test containers running for inspection",
    )
    return parser


def _available_port() -> int:
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        return int(server.getsockname()[1])


def _get_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=2.0) as response:
        return json.load(response)


def _wait_for_health(url: str, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            if _get_json(f"{url}/healthz").get("status") == "ok":
                return
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
            last_error = error
        time.sleep(0.25)
    raise RuntimeError(f"world-state did not become healthy: {last_error}")


def _wait_for_expectations(
    url: str, expectations: Sequence[dict[str, Any]], timeout: float
) -> list[dict[str, Any]]:
    deadline = time.monotonic() + timeout
    observations: list[dict[str, Any]] = []
    while time.monotonic() < deadline:
        snapshot = _get_json(f"{url}/v1/state")
        observations = snapshot.get("observations", [])
        if not unmatched_expectations(observations, expectations):
            return observations
        time.sleep(0.25)

    missing = unmatched_expectations(observations, expectations)
    streams = sorted(
        observation.get("stream", "<missing stream>") for observation in observations
    )
    raise AssertionError(
        "bag replay did not produce the expected observations\n"
        f"unmatched: {json.dumps(missing, indent=2, sort_keys=True)}\n"
        f"observed streams: {streams}"
    )


def _compose_command(root: Path, project: str) -> list[str]:
    return [
        "docker",
        "compose",
        "--project-name",
        project,
        "--file",
        str(root / "docker-compose.yml"),
        "--file",
        str(root / "docker-compose.local.yml"),
        "--file",
        str(root / "docker-compose.bag-test.yml"),
    ]


def _run(command: Sequence[str], env: dict[str, str], *, check: bool = True) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=env["ROBOTKIT_PROJECT_ROOT"], env=env, check=check)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    bag = args.bag.expanduser().resolve()
    if not bag.exists():
        raise SystemExit(f"bag does not exist: {bag}")
    if args.rate <= 0:
        raise SystemExit("--rate must be greater than zero")
    if args.startup_seconds < 0 or args.discovery_seconds < 0 or args.timeout <= 0:
        raise SystemExit(
            "startup and discovery seconds cannot be negative; timeout must be positive"
        )

    try:
        expectations = load_expectations(args.expect, args.expect_file)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"invalid expectations: {error}") from error
    if not expectations:
        raise SystemExit("provide at least one --expect or --expect-file")

    root = Path(__file__).resolve().parents[3]
    port = _available_port()
    domain_id = args.ros_domain_id
    if domain_id is None:
        domain_id = 100 + os.getpid() % 100
    project = f"robotkit-bag-{os.getpid()}"
    url = f"http://127.0.0.1:{port}"
    env = dict(os.environ)
    env.update(
        {
            "ROBOTKIT_PROJECT_ROOT": str(root),
            "ROBOTKIT_BAG_MOUNT": str(bag.parent),
            "ROBOTKIT_BAG_URI": f"/bags/{bag.name}",
            "ROBOTKIT_BAG_TEST_PORT": str(port),
            "ROBOTKIT_BAG_TEST_ROS_DOMAIN_ID": str(domain_id),
        }
    )
    compose = _compose_command(root, project)
    services = list(dict.fromkeys(args.service))
    successful = False
    print(
        f"bag test project={project} ROS_DOMAIN_ID={domain_id} world_state={url}",
        flush=True,
    )
    try:
        _run([*compose, "up", "--detach", "--build", "world-state", *services], env)
        _wait_for_health(url)
        time.sleep(args.startup_seconds)
        _run(
            [
                *compose,
                "run",
                "--rm",
                "bag-replay",
                "ros2",
                "bag",
                "play",
                env["ROBOTKIT_BAG_URI"],
                "--rate",
                str(args.rate),
                "--delay",
                str(args.discovery_seconds),
                "--disable-keyboard-controls",
            ],
            env,
        )
        observations = _wait_for_expectations(url, expectations, args.timeout)
        streams = sorted(observation["stream"] for observation in observations)
        print(f"PASS: matched {len(expectations)} expectation(s); streams={streams}")
        successful = True
        return 0
    except (
        AssertionError,
        OSError,
        RuntimeError,
        subprocess.CalledProcessError,
    ) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        _run(
            [*compose, "logs", "--no-color", "--tail", "200", "world-state", *services],
            env,
            check=False,
        )
        return 1
    finally:
        if args.keep:
            outcome = "completed" if successful else "failed"
            print(f"test {outcome}; kept Docker Compose project {project}")
        else:
            _run(
                [*compose, "down", "--volumes", "--remove-orphans"],
                env,
                check=False,
            )


if __name__ == "__main__":
    raise SystemExit(main())
