"""Stage-readiness scorecard over a fruit-soak session.

Evaluates every stage-readiness criterion against a session JSON and records
whether each passed — as data only. Nothing here gates or stops a session:
the soak keeps running whatever the numbers say, and the scorecard is
recomputed and embedded after every run so a session interrupted at run 6
still carries its scorecard for runs 1-6.

Each criterion reports ``passed`` as true, false, or null (not measured —
e.g. temperatures when no temp source was wired, or the dongle when device
probes were off). ``observations`` carries the lighting/recognition signals
that have no pass/fail meaning: per-fruit confidence across runs and how
long acquisition took under the session's lighting and placement.

Standalone use (re-score an existing session file):
    python3 scripts/stage_scorecard.py benchmarks/results/fruit-soak-X.json --write
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HOME_GATE_M = 0.10
LATENCY_P95_LIMIT_MS = 250.0
THERMAL_RISE_LIMIT_C = 10.0
MIN_SUCCESSES_PER_FRUIT = 2

# Meaningful confidence floors per fruit. Banana's app-side threshold is 0.20,
# but every published banana detection already passed the resident specialist
# at 0.55, so 0.55 is the floor that tells us recognition is healthy.
FRUIT_CONFIDENCE_FLOORS = {"apple": 0.70, "banana": 0.55, "pear": 0.65}

APPROACH_STAGES = ("approach_fruit",)
ACQUISITION_STAGES = ("turn_to_fruit", "search_fruit", "search")


def _criterion(passed: bool | None, **details) -> dict:
    return {"passed": passed, **details}


def _score_completion(
    runs: list[dict], target_runs: int, *, excluded: int = 0
) -> dict:
    completed = [r for r in runs if r.get("outcome") == "COMPLETED"]
    takeovers = [r for r in runs if r.get("outcome") == "REMOTE_TAKEOVER"]
    harness_stops = [r for r in runs if r.get("harness_note")]
    return _criterion(
        len(runs) == target_runs
        and len(completed) == target_runs
        and not takeovers
        and not harness_stops
        if runs
        else None,
        completed=len(completed),
        attempted=len(runs),
        target_runs=target_runs,
        excluded=excluded,
        remote_takeovers=len(takeovers),
        harness_stops=len(harness_stops),
    )


def _score_home_gate(runs: list[dict]) -> dict:
    distances = [
        r["home_distance_m"] for r in runs if r.get("home_distance_m") is not None
    ]
    if not distances:
        return _criterion(None, detail="no terminal home measurements yet")
    missing = len(runs) - len(distances)
    return _criterion(
        missing == 0 and all(d <= HOME_GATE_M for d in distances),
        gate_m=HOME_GATE_M,
        max_m=round(max(distances), 4),
        per_run=[round(d, 4) for d in distances],
        missing_measurements=missing,
    )


def _score_fruit_coverage(
    runs: list[dict], qualified_fruits: list[str] | None = None
) -> dict:
    per_fruit: dict[str, dict] = {
        fruit: {"attempts": 0, "successes": 0}
        for fruit in (qualified_fruits or [])
    }
    for run in runs:
        fruit = run.get("target_fruit")
        bucket = per_fruit.setdefault(fruit, {"attempts": 0, "successes": 0})
        bucket["attempts"] += 1
        if run.get("outcome") == "COMPLETED":
            bucket["successes"] += 1
    if not runs:
        return _criterion(
            None,
            detail="no runs yet",
            minimum_successes=MIN_SUCCESSES_PER_FRUIT,
            required_fruits=sorted(per_fruit),
            per_fruit=per_fruit,
        )
    short = {
        fruit: stats
        for fruit, stats in per_fruit.items()
        if stats["successes"] < MIN_SUCCESSES_PER_FRUIT
    }
    return _criterion(
        not short,
        minimum_successes=MIN_SUCCESSES_PER_FRUIT,
        required_fruits=sorted(per_fruit),
        per_fruit=per_fruit,
    )


def _approach_confidence_min(run: dict) -> float | None:
    stages = run.get("stage_telemetry") or {}
    minima = [
        stage["target_confidence"]["min"]
        for name, stage in stages.items()
        if name in APPROACH_STAGES and stage.get("target_confidence")
    ]
    return min(minima) if minima else None


def _score_confidence_floor(runs: list[dict]) -> dict:
    per_run = []
    verdicts = []
    for run in runs:
        fruit = run.get("target_fruit")
        floor = FRUIT_CONFIDENCE_FLOORS.get(fruit)
        observed = _approach_confidence_min(run)
        entry = {"number": run.get("number"), "fruit": fruit, "floor": floor,
                 "min_confidence": observed}
        if floor is not None and observed is not None:
            entry["held"] = observed >= floor
            verdicts.append(entry["held"])
        per_run.append(entry)
    return _criterion(
        all(verdicts) if verdicts else None,
        per_run=per_run,
    )


def _score_network(runs: list[dict]) -> dict:
    cutouts = sum((r.get("network") or {}).get("error_count", 0) for r in runs)
    p95s = [
        (r.get("network") or {}).get("latency_p95_ms")
        for r in runs
        if (r.get("network") or {}).get("latency_p95_ms") is not None
    ]
    if not p95s and not cutouts:
        return _criterion(None, detail="no network samples yet")
    worst_p95 = max(p95s) if p95s else None
    return _criterion(
        cutouts == 0 and (worst_p95 is None or worst_p95 <= LATENCY_P95_LIMIT_MS),
        cutouts=cutouts,
        worst_p95_ms=worst_p95,
        p95_limit_ms=LATENCY_P95_LIMIT_MS,
    )


def _run_max_temps(run: dict) -> dict[str, float]:
    maxima: dict[str, float] = {}
    for stage in (run.get("stage_telemetry") or {}).values():
        for zone, stats in (stage.get("temps_c") or {}).items():
            if stats and stats.get("max") is not None:
                maxima[zone] = max(maxima.get(zone, float("-inf")), stats["max"])
    return maxima


def _score_thermal(runs: list[dict]) -> dict:
    series: dict[str, list[float]] = {}
    for run in runs:
        for zone, peak in _run_max_temps(run).items():
            series.setdefault(zone, []).append(peak)
    measured = {zone: values for zone, values in series.items() if len(values) >= 2}
    if not measured:
        return _criterion(None, detail="temperatures not measured (or <2 runs)")
    rises = {
        zone: round(values[-1] - values[0], 2) for zone, values in measured.items()
    }
    return _criterion(
        all(rise <= THERMAL_RISE_LIMIT_C for rise in rises.values()),
        rise_limit_c=THERMAL_RISE_LIMIT_C,
        first_to_last_rise_c=rises,
        per_run_peaks_c={z: [round(v, 1) for v in vals] for z, vals in measured.items()},
    )


def _score_dongle(runs: list[dict]) -> dict:
    checks = [
        ((r.get("preflight_probes") or {}).get("dongle") or {}) for r in runs
    ]
    verdicts = [c.get("visible") for c in checks if c.get("visible") is not None]
    if not verdicts:
        return _criterion(None, detail="dongle not probed (enable --device-probes and --dongle-match)")
    return _criterion(
        all(verdicts),
        visible_per_run=verdicts,
    )


def _observations(runs: list[dict]) -> dict:
    """Lighting/recognition signals: informational, no pass/fail meaning."""
    confidence_by_fruit: dict[str, list[float]] = {}
    acquisition_s: dict[str, list[float]] = {}
    frames = []
    for run in runs:
        fruit = run.get("target_fruit")
        observed = _approach_confidence_min(run)
        if observed is not None:
            confidence_by_fruit.setdefault(fruit, []).append(observed)
        durations = run.get("stage_durations") or {}
        acquire = sum(durations.get(s, 0.0) for s in ACQUISITION_STAGES)
        if acquire:
            acquisition_s.setdefault(fruit, []).append(round(acquire, 2))
        if run.get("lighting_frame"):
            frames.append(run["lighting_frame"].get("path"))
    return {
        "approach_min_confidence_by_fruit": {
            fruit: {
                "min": round(min(vals), 3),
                "mean": round(sum(vals) / len(vals), 3),
                "runs": len(vals),
            }
            for fruit, vals in confidence_by_fruit.items()
        },
        "acquisition_seconds_by_fruit": acquisition_s,
        "lighting_frames": frames,
    }


def score_session(session: dict) -> dict:
    runs = session.get("runs") or []
    eligible_runs = [
        run for run in runs if run.get("reliability_eligible", True) is not False
    ]
    excluded_runs = [
        run for run in runs if run.get("reliability_eligible", True) is False
    ]
    eligible_target_runs = max(
        0, int(session.get("target_runs", 0)) - len(excluded_runs)
    )
    return {
        "recorded_only": True,
        "note": "criteria are recorded, never enforced; the session continues regardless",
        "operator_exclusions": {
            "count": len(excluded_runs),
            "runs": [
                {
                    "number": run.get("number"),
                    "run_id": run.get("run_id"),
                    "classification": (
                        (run.get("operator_review") or {}).get("classification")
                        or "excluded"
                    ),
                    "reason": (
                        (run.get("operator_review") or {}).get("reason")
                        or "operator excluded this attempt"
                    ),
                }
                for run in excluded_runs
            ],
        },
        "criteria": {
            "completion": _score_completion(
                eligible_runs,
                eligible_target_runs,
                excluded=len(excluded_runs),
            ),
            "home_gate": _score_home_gate(runs),
            "fruit_coverage": _score_fruit_coverage(
                eligible_runs, session.get("qualified_fruits")
            ),
            "approach_confidence_floor": _score_confidence_floor(eligible_runs),
            "network_stability": _score_network(eligible_runs),
            "thermal_trend": _score_thermal(eligible_runs),
            "dongle_visible": _score_dongle(eligible_runs),
        },
        "observations": _observations(eligible_runs),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path, help="fruit-soak session JSON")
    parser.add_argument("--write", action="store_true", help="embed the scorecard in the file")
    args = parser.parse_args(argv)

    session = json.loads(args.session.read_text())
    scorecard = score_session(session)
    print(json.dumps(scorecard, indent=2))
    if args.write:
        session["scorecard"] = scorecard
        args.session.write_text(json.dumps(session, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
