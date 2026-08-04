"""Pure draft state model for a bounded return to captured Home."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum


class ReturnMode(str, Enum):
    READY = "ready"
    TURN_TO_HOME = "turn_to_home"
    DRIVE_TO_HOME = "drive_to_home"
    BLOCKED = "blocked"
    RESTORE_HEADING = "restore_heading"
    AWAITING_DISARM = "awaiting_disarm"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass(frozen=True)
class DraftConfig:
    pose_maximum_age_s: float = 0.50
    position_tolerance_m: float = 0.10
    heading_tolerance_deg: float = 5.0
    course_gate_deg: float = 30.0
    minimum_progress_m: float = 0.05
    progress_window_s: float = 3.0
    maximum_home_distance_m: float = 3.0
    maximum_replans: int = 1


@dataclass(frozen=True)
class ReturnState:
    mode: ReturnMode = ReturnMode.READY
    home_distance_m: float = 1.20
    course_error_deg: float = 170.0
    home_heading_error_deg: float = 170.0
    pose_age_s: float = 0.05
    best_distance_m: float = 1.20
    no_progress_s: float = 0.0
    replans_used: int = 0
    motion_armed: bool = False
    safety_state: str = "DISARMED_CONFIRMED"
    reason: str = "waiting to start"


def start(state: ReturnState, config: DraftConfig) -> ReturnState:
    failure = _pose_or_envelope_failure(state, config)
    if failure:
        return _fail(state, failure)
    return _choose_motion_mode(
        replace(
            state,
            best_distance_m=state.home_distance_m,
            no_progress_s=0.0,
            reason="fresh pose accepted",
        ),
        config,
    )


def observe(
    state: ReturnState,
    config: DraftConfig,
    *,
    home_distance_m: float | None = None,
    course_error_deg: float | None = None,
    home_heading_error_deg: float | None = None,
    pose_age_s: float | None = None,
    elapsed_s: float = 0.1,
    obstacle_blocked: bool = False,
) -> ReturnState:
    if state.mode in {ReturnMode.COMPLETE, ReturnMode.FAILED}:
        return state
    updated = replace(
        state,
        home_distance_m=(
            state.home_distance_m
            if home_distance_m is None
            else home_distance_m
        ),
        course_error_deg=(
            state.course_error_deg
            if course_error_deg is None
            else course_error_deg
        ),
        home_heading_error_deg=(
            state.home_heading_error_deg
            if home_heading_error_deg is None
            else home_heading_error_deg
        ),
        pose_age_s=state.pose_age_s if pose_age_s is None else pose_age_s,
    )
    failure = _pose_or_envelope_failure(updated, config)
    if failure:
        return _fail(updated, failure)
    if obstacle_blocked:
        return replace(
            updated,
            mode=ReturnMode.BLOCKED,
            motion_armed=False,
            safety_state="DISARMED_CONFIRMED",
            reason="route blocked; stopped before bounded replan",
        )
    if updated.home_distance_m <= config.position_tolerance_m:
        if abs(updated.home_heading_error_deg) > config.heading_tolerance_deg:
            return replace(
                updated,
                mode=ReturnMode.RESTORE_HEADING,
                motion_armed=True,
                safety_state="STOP_REQUESTED_UNCONFIRMED",
                reason="position reached; restoring captured Home heading",
            )
        return replace(
            updated,
            mode=ReturnMode.AWAITING_DISARM,
            motion_armed=False,
            safety_state="STOP_REQUESTED_UNCONFIRMED",
            reason="position and heading reached; awaiting confirmed disarm",
        )

    if updated.home_distance_m <= (
        state.best_distance_m - config.minimum_progress_m
    ):
        updated = replace(
            updated,
            best_distance_m=updated.home_distance_m,
            no_progress_s=0.0,
        )
    elif state.mode == ReturnMode.DRIVE_TO_HOME:
        updated = replace(
            updated,
            no_progress_s=state.no_progress_s + elapsed_s,
        )
        if updated.no_progress_s >= config.progress_window_s:
            return _fail(updated, "bounded return made no measured progress")
    return _choose_motion_mode(updated, config)


def replan(state: ReturnState, config: DraftConfig, *, available: bool) -> ReturnState:
    if state.mode != ReturnMode.BLOCKED:
        return state
    if not available or state.replans_used >= config.maximum_replans:
        return _fail(state, "blocked route has no bounded safe replan")
    return replace(
        state,
        mode=ReturnMode.TURN_TO_HOME,
        replans_used=state.replans_used + 1,
        reason="one stationary collision-aware replan accepted",
    )


def confirm_disarm(state: ReturnState) -> ReturnState:
    if state.mode != ReturnMode.AWAITING_DISARM:
        return state
    return replace(
        state,
        mode=ReturnMode.COMPLETE,
        motion_armed=False,
        safety_state="DISARMED_CONFIRMED",
        reason="Home position, heading, and disarm confirmed",
    )


def _choose_motion_mode(state: ReturnState, config: DraftConfig) -> ReturnState:
    if abs(state.course_error_deg) > config.course_gate_deg:
        return replace(
            state,
            mode=ReturnMode.TURN_TO_HOME,
            motion_armed=True,
            safety_state="STOP_REQUESTED_UNCONFIRMED",
            reason="turning toward measured Home bearing",
        )
    return replace(
        state,
        mode=ReturnMode.DRIVE_TO_HOME,
        motion_armed=True,
        safety_state="STOP_REQUESTED_UNCONFIRMED",
        reason="forward-only collision-aware translation toward Home",
    )


def _pose_or_envelope_failure(
    state: ReturnState,
    config: DraftConfig,
) -> str | None:
    if state.pose_age_s > config.pose_maximum_age_s:
        return "fresh Home Distance is unavailable"
    if state.home_distance_m < 0.0:
        return "Home Distance is invalid"
    if state.home_distance_m > config.maximum_home_distance_m:
        return "Home lies outside the qualified return envelope"
    return None


def _fail(state: ReturnState, reason: str) -> ReturnState:
    return replace(
        state,
        mode=ReturnMode.FAILED,
        motion_armed=False,
        safety_state="STOP_REQUESTED_UNCONFIRMED",
        reason=reason,
    )
