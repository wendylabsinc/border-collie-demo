from __future__ import annotations

import pytest

from border_collie_demo.config import PerceptionConfig
from border_collie_demo.perception import PerceptionStatusClient


def valid_payload() -> dict[str, object]:
    return {
        "generation": "generation-1",
        "source": {
            "pts": 12345,
            "time_base": "1/90000",
            "received_monotonic_s": 99.80,
            "consecutive_frames": 10,
            "width": 1280,
            "height": 720,
        },
        "detection": {
            "generation": "generation-1",
            "source_pts": 12345,
            "source_time_base": "1/90000",
            "label": "pear",
            "confidence": 0.81,
            "consecutive_detections": 5,
            "inference_s": 0.08,
            "completed_monotonic_s": 99.85,
            "bbox_xyxy": [480, 360, 800, 700],
            "inference_passes": 2,
            "crop_confirmation": {
                "attempted": True,
                "promoted": True,
                "full_frame_confidence": 0.61,
                "crop_confidence": 0.81,
                "crop_xyxy": [400, 280, 880, 720],
                "agreement_iou": 0.72,
            },
            "color_identity": "red_apple",
            "color_confidence": 0.91,
        },
    }


def client_for(payload: dict[str, object]) -> PerceptionStatusClient:
    return PerceptionStatusClient(
        PerceptionConfig(enabled=True),
        fetcher=lambda _url, _timeout: payload,
        clock=lambda: 100.0,
    )


def test_qualified_camera_and_pear_evidence_is_ready() -> None:
    status = client_for(valid_payload()).status()

    assert status["ready"] is True
    assert status["generation"] == "generation-1"
    assert status["source"]["age_s"] < 0.350
    assert status["detection"]["age_s"] < 0.250


def test_qualified_apple_evidence_uses_its_own_threshold() -> None:
    payload = valid_payload()
    payload["target_fruit"] = "apple"
    payload["supported_fruits"] = ["apple", "banana", "pear"]
    payload["detection"]["label"] = "apple"
    payload["detection"]["confidence"] = 0.40

    status = client_for(payload).status()

    assert status["target_fruit"] == "apple"
    assert status["target_ready"] is True
    assert status["motion_qualified"] is True
    assert status["thresholds"]["target_minimum_confidence"] == 0.40


def test_qualified_mango_requires_raw_bowl_and_derived_identity() -> None:
    payload = valid_payload()
    payload["target_fruit"] = "mango"
    payload["supported_fruits"] = ["apple", "banana", "mango", "pear"]
    payload["detection"].update(
        label="mango",
        confidence=0.70,
        raw_label="bowl",
        raw_confidence=0.128,
        raw_bbox_xyxy=[480, 360, 800, 700],
        derived_identity="mango",
        derived_confidence=1.0,
    )

    ready = client_for(payload).status()

    assert ready["target_ready"] is True
    assert ready["detection"]["raw_label"] == "bowl"
    assert ready["detection"]["derived_identity"] == "mango"

    for field, value in (
        ("raw_label", "sports ball"),
        ("raw_confidence", 0.08),
        ("derived_identity", "unknown"),
        ("derived_confidence", 0.79),
        ("derived_confidence", None),
    ):
        invalid = valid_payload()
        invalid["target_fruit"] = "mango"
        invalid["detection"].update(payload["detection"])
        invalid["detection"][field] = value
        status = client_for(invalid).status()
        assert status["target_ready"] is False
        assert "derived Mango identity" in status["detail"]


def test_perception_client_selects_target_through_the_read_only_sidecar() -> None:
    calls: list[tuple[str, str, float]] = []
    client = PerceptionStatusClient(
        PerceptionConfig(enabled=True),
        target_poster=lambda url, fruit, timeout: (
            calls.append((url, fruit, timeout))
            or {
                "target_fruit": fruit,
                "supported_fruits": ["apple", "banana", "pear"],
            }
        ),
    )

    selected = client.select_target("banana")

    assert selected["target_fruit"] == "banana"
    assert calls == [("http://127.0.0.1:8111/api/target", "banana", 0.25)]


def test_status_preserves_validated_geometry_for_approach_and_arrival() -> None:
    status = client_for(valid_payload()).status()

    assert status["camera_healthy"] is True
    assert status["target_ready"] is True
    assert status["source"]["width"] == 1280
    assert status["source"]["height"] == 720
    assert status["detection"]["bbox_xyxy"] == [480.0, 360.0, 800.0, 700.0]
    assert status["detection"]["center_x_ratio"] == 0.5
    assert status["detection"]["center_y_ratio"] == pytest.approx(0.736111)
    assert status["detection"]["bottom_ratio"] == pytest.approx(0.972222)
    assert status["detection"]["bbox_width_ratio"] == 0.25
    assert status["detection"]["bbox_height_ratio"] == pytest.approx(0.472222)
    assert status["detection"]["bbox_area_ratio"] == pytest.approx(0.1180556)
    assert status["detection"]["inference_passes"] == 2
    assert status["detection"]["crop_confirmation"]["promoted"] is True
    assert status["detection"]["crop_confirmation"]["crop_confidence"] == 0.81
    assert status["detection"]["color_identity"] == "red_apple"
    assert status["detection"]["color_confidence"] == 0.91


def test_detection_must_be_bound_to_the_current_camera_generation() -> None:
    payload = valid_payload()
    payload["detection"]["generation"] = "old-generation"

    status = client_for(payload).status()

    assert status["camera_healthy"] is True
    assert status["target_ready"] is False
    assert "generation does not match" in status["detail"]


def test_disabled_perception_fails_closed_without_fetching() -> None:
    calls = 0

    def fetch(_url: str, _timeout: float) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return valid_payload()

    status = PerceptionStatusClient(
        PerceptionConfig(enabled=False),
        fetcher=fetch,
    ).status()

    assert status == {
        "ready": False,
        "detail": "production camera/perception adapter is disabled",
    }
    assert calls == 0


def test_stale_source_or_detection_fails_closed() -> None:
    source_stale = valid_payload()
    source_stale["source"]["received_monotonic_s"] = 99.0
    assert client_for(source_stale).status()["ready"] is False
    assert "source progress is stale" in client_for(source_stale).status()["detail"]

    detection_stale = valid_payload()
    detection_stale["detection"]["completed_monotonic_s"] = 99.0
    assert client_for(detection_stale).status()["ready"] is False
    assert "pear detection is stale" in client_for(detection_stale).status()["detail"]


def test_insufficient_or_slow_evidence_fails_closed() -> None:
    payload = valid_payload()
    payload["source"]["consecutive_frames"] = 9
    payload["detection"]["confidence"] = 0.64
    payload["detection"]["consecutive_detections"] = 4
    payload["detection"]["inference_s"] = 0.201

    status = client_for(payload).status()

    assert status["ready"] is False
    assert "fewer than 10" in status["detail"]
    assert "below 0.65" in status["detail"]
    assert "fewer than 5" in status["detail"]
    assert "exceeds 0.200" in status["detail"]


def test_malformed_or_unreachable_status_fails_closed() -> None:
    malformed = client_for({}).status()
    assert malformed["ready"] is False
    assert "generation is missing" in malformed["detail"]

    def unavailable(_url: str, _timeout: float) -> dict[str, object]:
        raise TimeoutError("sidecar timed out")

    status = PerceptionStatusClient(
        PerceptionConfig(enabled=True),
        fetcher=unavailable,
    ).status()
    assert status["ready"] is False
    assert status["detail"] == (
        "camera/perception status unavailable: sidecar timed out"
    )


def test_camera_violations_name_the_failing_source_check() -> None:
    """A camera_unhealthy stop must be diagnosable from the record alone.

    Guidance halts a run the moment camera_healthy goes false, but the reason
    is a source-level check the run record previously discarded, leaving the
    failure undiagnosable without a live repro.
    """
    healthy = client_for(valid_payload()).status()
    assert healthy["camera_healthy"] is True
    assert healthy["camera_violations"] == []

    stalled = valid_payload()
    stalled["source"]["received_monotonic_s"] = 99.0  # 1.0 s old, gate is 0.350
    status = client_for(stalled).status()

    assert status["camera_healthy"] is False
    assert status["camera_violations"] == ["source progress is stale"]

    reconnected = valid_payload()
    reconnected["source"]["consecutive_frames"] = 3
    status = client_for(reconnected).status()

    assert status["camera_healthy"] is False
    assert status["camera_violations"] == ["fewer than 10 consecutive source frames"]


def test_camera_violations_exclude_target_only_failures() -> None:
    """A weak detection is not a camera fault and must not be reported as one."""
    weak = valid_payload()
    weak["detection"]["confidence"] = 0.01

    status = client_for(weak).status()

    assert status["ready"] is False
    assert status["camera_healthy"] is True
    assert status["camera_violations"] == []
