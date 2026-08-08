from __future__ import annotations

from typing import Any


def evaluate_preflight(
    hardware: dict[str, Any],
    camera_perception: dict[str, Any] | None = None,
    media: dict[str, Any] | None = None,
    remote_input: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pose = hardware.get("pose")
    motion = hardware.get("motion")
    camera = camera_perception or {
        "ready": False,
        "detail": "production camera/perception adapter is not connected",
    }
    camera_source_ready = bool(camera.get("camera_healthy", camera.get("ready")))
    camera_source_detail = (
        "camera source is healthy and advancing"
        if camera_source_ready and "camera_healthy" in camera
        else str(camera.get("detail") or "camera readiness unavailable")
    )
    checks = [
        {
            "name": "durable_run_storage",
            "ready": True,
            "detail": "Run Result was persisted before preflight",
        },
        {
            "name": "hardware_connected",
            "ready": bool(hardware.get("connected")),
            "detail": (
                "Go2 hardware is connected"
                if hardware.get("connected")
                else hardware.get("fault") or "Go2 hardware is not connected"
            ),
        },
        {
            "name": "autonomy_enabled",
            "ready": bool(hardware.get("autonomy_enabled", True)),
            "detail": (
                "autonomous demo motion is enabled"
                if hardware.get("autonomy_enabled", True)
                else "autonomous demo motion is disabled"
            ),
        },
        {
            "name": "fresh_pose",
            "ready": bool(pose and pose.get("healthy")),
            "detail": (
                f"pose age is {pose.get('age_s')} seconds"
                if pose and pose.get("healthy")
                else (pose or {}).get("error") or "fresh Go2 pose is unavailable"
            ),
        },
        {
            "name": "motion_disarmed",
            "ready": bool(
                hardware.get("active_operation") is None
                and (motion is None or not motion.get("armed", False))
            ),
            "detail": "application motion owner is disarmed",
        },
        {
            "name": "camera_perception_ready",
            "ready": camera_source_ready,
            "detail": camera_source_detail,
        },
    ]
    if media is not None:
        checks.append(
            {
                "name": "bark_media_ready",
                "ready": bool(media.get("ready")),
                "detail": str(media.get("detail") or "bark readiness unavailable"),
            }
        )
    if remote_input is not None:
        checks.append(
            {
                "name": "remote_takeover_monitor",
                "ready": bool(remote_input.get("ready")),
                "detail": (
                    "physical controller takeover monitor is fresh"
                    if remote_input.get("ready")
                    else str(
                        remote_input.get("error")
                        or "physical controller takeover monitor is unavailable"
                    )
                ),
            }
        )
    return {
        "ready": all(check["ready"] for check in checks),
        "checks": checks,
        "camera_perception": camera,
        "media": media,
        "remote_input": remote_input,
    }


def preflight_check_ready(report: dict[str, Any], name: str) -> bool:
    return any(
        check.get("name") == name and bool(check.get("ready"))
        for check in report.get("checks", [])
    )
