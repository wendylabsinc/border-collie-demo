"""Compare three retained A/B/C search-policy soak sessions."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median

POLICIES = ("fast-lock", "slow-sweep", "double-back")


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _profile(session: dict[str, object]) -> dict[str, object]:
    runs = session.get("runs")
    if not isinstance(runs, list):
        raise TypeError("each policy session must contain runs")
    completed = [run for run in runs if run.get("outcome") == "COMPLETED"]
    failed = [run for run in runs if run.get("outcome") != "COMPLETED"]
    failures_by_reason = Counter(str(run.get("reason")) for run in failed)
    failures_by_phase = Counter(str(run.get("failed_phase")) for run in failed)
    failures_by_fruit = Counter(str(run.get("target_fruit")) for run in failed)
    acquisition_s: list[float] = []
    home_distances: list[float] = []
    network_poll_errors = 0
    for run in runs:
        durations = run.get("stage_durations")
        if isinstance(durations, dict):
            acquisition = sum(
                value
                for phase in ("turn_to_fruit", "find_fruit")
                if (value := _number(durations.get(phase))) is not None
            )
            acquisition_s.append(acquisition)
        if (home := _number(run.get("home_distance_m"))) is not None:
            home_distances.append(home)
        network = run.get("network")
        if isinstance(network, dict):
            network_poll_errors += int(
                network.get("error_count", network.get("poll_errors", 0)) or 0
            )
    recovered = sum(
        1
        for run in failed
        if isinstance(run.get("recovery"), dict)
        and run["recovery"].get("outcome") == "COMPLETED"
    )
    attempted = len(runs)
    return {
        "attempted": attempted,
        "completed": len(completed),
        "failed": len(failed),
        "completion_rate": round(len(completed) / attempted, 4) if attempted else 0.0,
        "failures_by_reason": dict(sorted(failures_by_reason.items())),
        "failures_by_phase": dict(sorted(failures_by_phase.items())),
        "failures_by_fruit": dict(sorted(failures_by_fruit.items())),
        "median_acquisition_s": (
            round(median(acquisition_s), 3) if acquisition_s else None
        ),
        "mean_home_distance_m": (
            round(mean(home_distances), 4) if home_distances else None
        ),
        "maximum_home_distance_m": (
            round(max(home_distances), 4) if home_distances else None
        ),
        "recovery_success_rate": (
            round(recovered / len(failed), 4) if failed else None
        ),
        "network_poll_errors": network_poll_errors,
    }


def compare_sessions(sessions: list[dict[str, object]]) -> dict[str, object]:
    """Return one comparison only when the A/B/C trial inputs are matched."""
    if len(sessions) != len(POLICIES):
        raise ValueError("comparison requires exactly three policy sessions")
    by_policy: dict[str, dict[str, object]] = {}
    for session in sessions:
        policy = session.get("search_policy")
        if policy not in POLICIES or policy in by_policy:
            raise ValueError("comparison requires one session for each search policy")
        by_policy[str(policy)] = session
    reference = by_policy[POLICIES[0]]
    seed = reference.get("seed")
    fruits = reference.get("fruit_sequence")
    orientations = reference.get("orientation_sequence_degrees")
    if any(
        session.get("seed") != seed
        or session.get("fruit_sequence") != fruits
        or session.get("orientation_sequence_degrees") != orientations
        for session in by_policy.values()
    ):
        raise ValueError(
            "A/B/C sessions must use identical fruit and orientation sequences"
        )
    profiles = {policy: _profile(by_policy[policy]) for policy in POLICIES}
    ranking = sorted(
        (
            {"search_policy": policy, **summary}
            for policy, summary in profiles.items()
        ),
        key=lambda row: (
            -float(row["completion_rate"]),
            int(row["failed"]),
            float(row["median_acquisition_s"] or float("inf")),
        ),
    )
    return {
        "schema_version": 1,
        "trial": {
            "seed": seed,
            "runs_per_policy": len(reference.get("runs") or []),
            "fruit_sequence": fruits,
            "orientation_sequence_degrees": orientations,
        },
        "profiles": profiles,
        "ranking": ranking,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sessions", nargs=3, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    comparison = compare_sessions(
        [json.loads(path.read_text(encoding="utf-8")) for path in args.sessions]
    )
    payload = json.dumps(comparison, indent=2) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
