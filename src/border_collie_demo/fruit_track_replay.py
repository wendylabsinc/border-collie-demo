"""Deterministic replay of flight-recorder perception samples."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from .persistent_fruit_tracker import FruitTrackState, PersistentFruitTracker


@dataclass(frozen=True)
class FruitTrackReplayResult:
    processed_frames: int
    acquisition_epochs: int
    phase_identity_resets: int
    degraded_recoveries: int
    off_axis_identity_preserved: int
    confirmed_losses: int
    maximum_confirmed_loss_s: float
    degraded_arrival_advances: int
    transitions: tuple[Mapping[str, object], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "processed_frames": self.processed_frames,
            "acquisition_epochs": self.acquisition_epochs,
            "phase_identity_resets": self.phase_identity_resets,
            "degraded_recoveries": self.degraded_recoveries,
            "off_axis_identity_preserved": self.off_axis_identity_preserved,
            "confirmed_losses": self.confirmed_losses,
            "maximum_confirmed_loss_s": self.maximum_confirmed_loss_s,
            "degraded_arrival_advances": self.degraded_arrival_advances,
            "transitions": [dict(transition) for transition in self.transitions],
        }


class FruitTrackReplay:
    """Replay saved black-box events through the production tracker seam."""

    def __init__(self, tracker: PersistentFruitTracker) -> None:
        self._tracker = tracker

    def replay(
        self,
        events: Iterable[Mapping[str, object]],
    ) -> FruitTrackReplayResult:
        transitions: list[Mapping[str, object]] = []
        processed = 0
        phase_resets = 0
        degraded_recoveries = 0
        off_axis_preserved = 0
        confirmed_losses = 0
        maximum_loss_s = 0.0
        degraded_arrival_advances = 0
        previous_phase: str | None = None
        previous_state = self._tracker.state
        previous_epoch = 0
        degraded_started_at: float | None = None

        for event in events:
            if event.get("kind") != "perception_sample":
                continue
            payload = event.get("payload")
            if not isinstance(payload, Mapping):
                continue
            now_raw = event.get("recorded_monotonic_s")
            if isinstance(now_raw, bool) or not isinstance(now_raw, (int, float)):
                continue
            phase = str(payload.get("phase") or "unknown")
            report = self._tracker.observe(payload, now_s=float(now_raw))
            processed += 1

            if (
                previous_phase is not None
                and phase != previous_phase
                and previous_epoch > 0
                and report.acquisition_epoch != previous_epoch
            ):
                phase_resets += 1
            if previous_state is FruitTrackState.DEGRADED and report.state in {
                FruitTrackState.LOCKED,
                FruitTrackState.LOCKED_OFF_AXIS,
            }:
                degraded_recoveries += 1
            if (
                report.state is FruitTrackState.LOCKED_OFF_AXIS
                and report.same_identity
                and not report.motion_authorized
            ):
                off_axis_preserved += 1
            if report.state is FruitTrackState.DEGRADED and degraded_started_at is None:
                degraded_started_at = float(now_raw)
            if report.state is FruitTrackState.LOST and previous_state is not FruitTrackState.LOST:
                confirmed_losses += 1
                if degraded_started_at is not None:
                    maximum_loss_s = max(
                        maximum_loss_s,
                        float(now_raw) - degraded_started_at,
                    )
                degraded_started_at = None
            elif report.state is not FruitTrackState.DEGRADED:
                degraded_started_at = None
            if report.state is FruitTrackState.DEGRADED and report.arrival_eligible:
                degraded_arrival_advances += 1

            transitions.append(
                {
                    "phase": phase,
                    "recorded_monotonic_s": float(now_raw),
                    **report.to_evidence(),
                    "resulting_command": payload.get("resulting_command"),
                }
            )
            previous_phase = phase
            previous_state = report.state
            previous_epoch = report.acquisition_epoch

        return FruitTrackReplayResult(
            processed_frames=processed,
            acquisition_epochs=previous_epoch,
            phase_identity_resets=phase_resets,
            degraded_recoveries=degraded_recoveries,
            off_axis_identity_preserved=off_axis_preserved,
            confirmed_losses=confirmed_losses,
            maximum_confirmed_loss_s=maximum_loss_s,
            degraded_arrival_advances=degraded_arrival_advances,
            transitions=tuple(transitions),
        )
