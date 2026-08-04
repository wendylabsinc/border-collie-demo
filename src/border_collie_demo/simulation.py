"""Explicit zero-motion boundaries for an end-to-end base Demo Run."""

from __future__ import annotations

from time import monotonic

from .hardware import HardwareUnavailable


class SimulatedHardware:
    async def start(self) -> None:
        return None

    async def close(self) -> list[str]:
        return []

    async def emergency_stop(self) -> list[str]:
        return []

    async def run_forward_pulse(self, confirmation: str) -> dict[str, object]:
        del confirmation
        raise HardwareUnavailable("simulation never sends motion commands")

    def capture_home(self) -> dict[str, object]:
        captured = monotonic()
        return {
            "x_m": 0.0,
            "y_m": 0.0,
            "yaw_rad": 0.0,
            "captured_monotonic_s": captured,
            "age_s": 0.0,
            "source": "simulation",
        }

    def status(self) -> dict[str, object]:
        return {
            "configured": True,
            "autonomy_enabled": True,
            "simulated": True,
            "lab_motion_enabled": False,
            "clients_initialized": True,
            "connected": True,
            "fault": None,
            "network_interface": None,
            "active_operation": None,
            "can_pulse_forward": False,
            "motion": {"initialized": True, "armed": False},
            "pose": {"healthy": True, "age_s": 0.0, "error": None},
            "last_pulse": None,
        }


def simulated_camera_perception() -> dict[str, object]:
    return {
        "ready": True,
        "detail": "simulation provides deterministic Target Fruit evidence",
        "generation": "simulation",
        "detection": {
            "label": "pear",
            "confidence": 0.81,
            "consecutive_detections": 5,
        },
    }
