"""Typed, caller-neutral policy for a sequence of independent Demo Runs."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any


HARD_STOP_REASONS = frozenset(
    {
        "CAMERA_FAILURE",
        "MOTION_FAILURE",
        "REMOTE_TAKEOVER",
        "PROCESS_INTERRUPTED",
        "RETURN_HOME_FAILURE",
        "PREFLIGHT_FAILURE",
        "INTERNAL_ERROR",
        "OPERATOR_STOP",
    }
)
HARD_STOP_PHASES = frozenset(
    {"preflight", "capture_home", "return_home", "restore_heading", "remote_takeover"}
)


@dataclass(frozen=True)
class FailureSelector:
    """A stable Run Result reason, failed phase, or their intersection."""

    failed_phase: str | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        phase = self.failed_phase.strip() if self.failed_phase else None
        reason = self.reason.strip().upper() if self.reason else None
        if not phase and not reason:
            raise ValueError("a failure selector requires failed_phase or reason")
        object.__setattr__(self, "failed_phase", phase)
        object.__setattr__(self, "reason", reason)

    def matches(self, run: dict[str, Any]) -> bool:
        return (
            (self.failed_phase is None or run.get("failed_phase") == self.failed_phase)
            and (self.reason is None or run.get("reason") == self.reason)
        )

    def to_dict(self) -> dict[str, str | None]:
        return {"failed_phase": self.failed_phase, "reason": self.reason}


@dataclass(frozen=True)
class CohortPolicy:
    """Operator intent for a bounded cohort of independent Demo Runs."""

    seed: int
    runs: int = 5
    randomized: bool = True
    target_fruit: str | None = None
    tolerated_failures: tuple[FailureSelector, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.runs, bool) or not 1 <= self.runs <= 100:
            raise ValueError("runs must be within 1..100")
        target = self.target_fruit.casefold().strip() if self.target_fruit else None
        if not self.randomized and target is None:
            raise ValueError("target_fruit is required when randomized is false")
        object.__setattr__(self, "target_fruit", target)

    def to_dict(self) -> dict[str, Any]:
        return {
            "runs": self.runs,
            "randomized": self.randomized,
            "target_fruit": self.target_fruit,
            "seed": self.seed,
            "tolerated_failures": [item.to_dict() for item in self.tolerated_failures],
        }


def choose_fruit_sequence(
    policy: CohortPolicy, qualified_fruits: list[str] | tuple[str, ...]
) -> list[str]:
    """Return the complete persisted Target Fruit sequence before activation."""
    fruits = sorted({fruit.casefold().strip() for fruit in qualified_fruits if fruit})
    if not fruits:
        raise ValueError("deployed build reports no qualified fruits")
    if not policy.randomized:
        if policy.target_fruit not in fruits:
            raise ValueError(f"Target Fruit {policy.target_fruit!r} is not qualified")
        return [policy.target_fruit] * policy.runs
    rng = random.Random(policy.seed)
    repetitions, remainder = divmod(policy.runs, len(fruits))
    sequence = fruits * repetitions
    sequence.extend(rng.sample(fruits, remainder))
    rng.shuffle(sequence)
    return sequence


def decide_terminal_run(
    run: dict[str, Any], policy: CohortPolicy
) -> dict[str, Any]:
    """Decide only the policy gate; Home clearance remains a separate gate."""
    reason = str(run.get("reason") or "UNKNOWN")
    phase = str(run.get("failed_phase") or "unknown")
    failure_details = run.get("failure_details")
    safety_class = (
        failure_details.get("safety_class")
        if isinstance(failure_details, dict)
        else None
    )
    hard_stop = (
        run.get("final_safety_state") != "DISARMED_CONFIRMED"
        or reason in HARD_STOP_REASONS
        or phase in HARD_STOP_PHASES
        or run.get("outcome") == "REMOTE_TAKEOVER"
        or safety_class
        in {"camera", "pose", "motion", "takeover", "restart_required"}
    )
    if hard_stop:
        return {
            "action": "STOP_COHORT",
            "code": "HARD_SAFETY_STOP",
            "detail": f"{reason} in {phase} is a non-tolerable safety boundary",
            "matched_selector": None,
        }

    if run.get("outcome") == "COMPLETED" and reason == "SUCCESS":
        return {
            "action": "AWAIT_HOME_CLEARANCE",
            "code": "RUN_COMPLETED",
            "detail": "Demo Run completed; fresh Home clearance is still required",
            "matched_selector": None,
        }

    selector = next(
        (item for item in policy.tolerated_failures if item.matches(run)), None
    )
    if selector is not None:
        return {
            "action": "AWAIT_HOME_CLEARANCE",
            "code": "FAILURE_TOLERATED",
            "detail": f"{reason} in {phase} is tolerated only for the cohort boundary",
            "matched_selector": selector.to_dict(),
        }
    return {
        "action": "STOP_COHORT",
        "code": "FAILURE_POLICY_STOP",
        "detail": f"{reason} in {phase} is configured to stop the cohort",
        "matched_selector": None,
    }


def evaluate_home_clearance(
    status: dict[str, Any], prior_run_id: str
) -> dict[str, Any]:
    """Evaluate the exact-run, exact-zero, fresh-Home cohort boundary."""
    mission = status.get("mission") or {}
    activation = status.get("activation") or {}
    inter_run = activation.get("inter_run") or {}
    motion = (status.get("hardware") or {}).get("motion") or {}
    command = motion.get("last_command") or {}
    exact_zero = all(
        isinstance(command.get(key), (int, float))
        and not isinstance(command.get(key), bool)
        and float(command[key]) == 0.0
        for key in ("forward_mps", "yaw_rps")
    )
    evidence = {
        **inter_run,
        "prior_run_matches": inter_run.get("prior_run_id") == prior_run_id,
        "client_run_cleared": status.get("active_run_id") is None,
        "client_motion_disarmed": motion.get("armed") is False and exact_zero,
        "activation_ready": activation.get("ready") is True,
        "restart_required": mission.get("restart_required") is True,
    }
    evidence["safe_to_continue"] = all(
        (
            evidence["prior_run_matches"],
            inter_run.get("returned_home") is True,
            evidence["client_run_cleared"],
            evidence["client_motion_disarmed"],
            evidence["activation_ready"],
            not evidence["restart_required"],
        )
    )
    return evidence
