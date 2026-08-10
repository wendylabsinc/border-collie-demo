from __future__ import annotations

import math
import os
from dataclasses import dataclass


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name, "1" if default else "0")
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class HardwareConfig:
    enabled: bool = False
    autonomy_enabled: bool = False
    lab_motion_enabled: bool = False
    network_interface: str | None = None
    forward_pulse_mps: float = 0.50
    forward_pulse_duration_s: float = 0.40
    command_heartbeat_s: float = 0.10
    maximum_forward_mps: float = 1.0
    maximum_yaw_rps: float = 0.80
    command_watchdog_s: float = 0.35
    motion_authority_ttl_s: float = 2.0
    rpc_timeout_s: float = 0.75
    client_timeout_s: float = 12.0
    remote_api_settle_s: float = 0.50
    pose_maximum_age_s: float = 0.50
    breadcrumb_spacing_m: float = 0.20
    breadcrumb_reach_m: float = 0.25
    maximum_breadcrumbs: int = 128
    pear_tracking_minimum_confidence: float = 0.55
    pear_tracking_confirmations: int = 3

    def __post_init__(self) -> None:
        positive_values = (
            "forward_pulse_mps",
            "forward_pulse_duration_s",
            "command_heartbeat_s",
            "maximum_forward_mps",
            "maximum_yaw_rps",
            "command_watchdog_s",
            "motion_authority_ttl_s",
            "rpc_timeout_s",
            "client_timeout_s",
            "pose_maximum_age_s",
            "breadcrumb_spacing_m",
            "breadcrumb_reach_m",
        )
        for name in positive_values:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.forward_pulse_mps > self.maximum_forward_mps:
            raise ValueError("forward pulse exceeds the configured motion limit")
        if self.command_heartbeat_s >= self.command_watchdog_s:
            raise ValueError("command heartbeat must be faster than the watchdog")
        if self.motion_authority_ttl_s <= self.command_watchdog_s:
            raise ValueError("motion authority TTL must exceed the command watchdog")
        if self.remote_api_settle_s < 0.0:
            raise ValueError("remote_api_settle_s must be non-negative")
        if not 0.0 <= self.pear_tracking_minimum_confidence <= 1.0:
            raise ValueError(
                "pear_tracking_minimum_confidence must be between zero and one"
            )
        if self.pear_tracking_confirmations < 1:
            raise ValueError("pear_tracking_confirmations must be positive")
        if self.maximum_breadcrumbs < 2:
            raise ValueError("maximum_breadcrumbs must be at least two")

    @classmethod
    def from_env(cls) -> HardwareConfig:
        interface = os.environ.get("GO2_NETWORK_INTERFACE", "").strip() or None
        return cls(
            enabled=env_bool("BORDER_COLLIE_HARDWARE_ENABLED"),
            autonomy_enabled=env_bool("BORDER_COLLIE_AUTONOMY_ENABLED"),
            lab_motion_enabled=env_bool("BORDER_COLLIE_LAB_MOTION_ENABLED"),
            network_interface=interface,
            forward_pulse_mps=float(
                os.environ.get("BORDER_COLLIE_FORWARD_PULSE_MPS", "0.50")
            ),
            forward_pulse_duration_s=float(
                os.environ.get("BORDER_COLLIE_FORWARD_PULSE_DURATION_S", "0.40")
            ),
            command_heartbeat_s=float(
                os.environ.get("BORDER_COLLIE_COMMAND_HEARTBEAT_S", "0.10")
            ),
            maximum_forward_mps=float(
                os.environ.get("BORDER_COLLIE_MAX_FORWARD_MPS", "1.0")
            ),
            maximum_yaw_rps=float(os.environ.get("BORDER_COLLIE_MAX_YAW_RPS", "0.80")),
            command_watchdog_s=float(
                os.environ.get("BORDER_COLLIE_COMMAND_WATCHDOG_S", "0.35")
            ),
            motion_authority_ttl_s=float(
                os.environ.get("BORDER_COLLIE_MOTION_AUTHORITY_TTL_S", "2.0")
            ),
            rpc_timeout_s=float(os.environ.get("BORDER_COLLIE_RPC_TIMEOUT_S", "0.75")),
            client_timeout_s=float(
                os.environ.get("BORDER_COLLIE_CLIENT_TIMEOUT_S", "12.0")
            ),
            remote_api_settle_s=float(
                os.environ.get("BORDER_COLLIE_REMOTE_API_SETTLE_S", "0.50")
            ),
            pose_maximum_age_s=float(
                os.environ.get("BORDER_COLLIE_POSE_MAX_AGE_S", "0.50")
            ),
            breadcrumb_spacing_m=float(
                os.environ.get("BORDER_COLLIE_BREADCRUMB_SPACING_M", "0.20")
            ),
            breadcrumb_reach_m=float(
                os.environ.get("BORDER_COLLIE_BREADCRUMB_REACH_M", "0.25")
            ),
            maximum_breadcrumbs=int(
                os.environ.get("BORDER_COLLIE_MAXIMUM_BREADCRUMBS", "128")
            ),
            pear_tracking_minimum_confidence=float(
                os.environ.get(
                    "BORDER_COLLIE_PEAR_TRACKING_MIN_CONFIDENCE",
                    "0.55",
                )
            ),
            pear_tracking_confirmations=int(
                os.environ.get(
                    "BORDER_COLLIE_PEAR_TRACKING_CONFIRMATIONS",
                    "3",
                )
            ),
        )


@dataclass(frozen=True)
class PerceptionConfig:
    enabled: bool = False
    status_url: str = "http://127.0.0.1:8111/status"
    target_url: str = "http://127.0.0.1:8111/api/target"
    frame_url: str = "http://127.0.0.1:8111/api/camera/frame.jpg"
    timeout_s: float = 0.25

    def __post_init__(self) -> None:
        if not self.status_url.startswith(("http://", "https://")):
            raise ValueError("perception status_url must use http or https")
        if not self.target_url.startswith(("http://", "https://")):
            raise ValueError("perception target_url must use http or https")
        if not self.frame_url.startswith(("http://", "https://")):
            raise ValueError("perception frame_url must use http or https")
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0.0:
            raise ValueError("perception timeout_s must be finite and positive")

    @classmethod
    def from_env(cls) -> PerceptionConfig:
        return cls(
            enabled=env_bool("BORDER_COLLIE_PERCEPTION_ENABLED"),
            status_url=os.environ.get(
                "BORDER_COLLIE_PERCEPTION_STATUS_URL",
                "http://127.0.0.1:8111/status",
            ).strip(),
            target_url=os.environ.get(
                "BORDER_COLLIE_PERCEPTION_TARGET_URL",
                "http://127.0.0.1:8111/api/target",
            ).strip(),
            frame_url=os.environ.get(
                "BORDER_COLLIE_PERCEPTION_FRAME_URL",
                "http://127.0.0.1:8111/api/camera/frame.jpg",
            ).strip(),
            timeout_s=float(
                os.environ.get("BORDER_COLLIE_PERCEPTION_TIMEOUT_S", "0.25")
            ),
        )
