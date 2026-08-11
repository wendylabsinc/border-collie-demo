from border_collie_demo.preflight import evaluate_preflight


def test_camera_arrival_profile_does_not_require_metric_range() -> None:
    hardware = {
        "connected": True,
        "autonomy_enabled": True,
        "active_operation": None,
        "pose": {"healthy": True, "age_s": 0.01},
        "motion": {"armed": False},
        "metric_arrival_required": False,
        "metric_arrival": {"configured": False, "ready": False},
    }

    report = evaluate_preflight(
        hardware,
        {"ready": True, "camera_healthy": True},
        {"ready": True},
    )

    assert report["ready"] is True
    assert all(
        check["name"] != "metric_arrival_calibrated"
        for check in report["checks"]
    )
