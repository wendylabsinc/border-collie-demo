from __future__ import annotations

from copy import deepcopy

import pytest

from border_collie_demo.search_policy import SearchPolicy


def qualified_apple_status(*, consecutive: int = 3) -> dict[str, object]:
    return {
        "camera_healthy": True,
        "target_ready": False,
        "generation": "camera-1",
        "source": {
            "pts": 900,
            "time_base": "1/90000",
            "age_s": 0.02,
        },
        "detection": {
            "label": "apple",
            "generation": "camera-1",
            "source_pts": 900,
            "source_time_base": "1/90000",
            "confidence": 0.806853175163269,
            "consecutive_detections": consecutive,
            "inference_s": 0.08,
            "age_s": 0.03,
            "center_x_ratio": 0.48,
            "center_y_ratio": 0.55,
            "bottom_ratio": 0.67,
            "bbox_area_ratio": 0.04,
        },
    }


def test_cli_search_policy_overrides_environment() -> None:
    policy = SearchPolicy.configured(
        cli_name="double-back",
        environ={"BORDER_COLLIE_SEARCH_POLICY": "slow-sweep"},
    )

    assert policy.name == "double-back"


def test_search_policy_uses_environment_when_cli_is_absent() -> None:
    policy = SearchPolicy.configured(
        environ={"BORDER_COLLIE_SEARCH_POLICY": "fast-lock"},
    )

    assert policy.name == "fast-lock"


def test_search_policy_defaults_to_slow_sweep_for_production_safety() -> None:
    assert SearchPolicy.configured(environ={}).name == "slow-sweep"


def test_profiles_change_only_the_named_experiment_dimension() -> None:
    fast = SearchPolicy.named("fast-lock")
    slow = SearchPolicy.named("slow-sweep")
    revisit = SearchPolicy.named("double-back")

    assert fast.minimum_consecutive_detections == 3
    assert fast.broad_yaw_rps == 1.0
    assert slow.minimum_consecutive_detections == 5
    assert slow.broad_yaw_rps == 0.5
    assert slow.broad_sweep_rad == pytest.approx(2.0 * 3.141592653589793)
    assert slow.broad_timeout_s == 30.0
    assert revisit.minimum_consecutive_detections == 5
    assert revisit.broad_yaw_rps == 1.0
    assert revisit.candidate_mode == "double-back"


def test_fast_lock_recovers_four_frame_apple_but_not_one_frame_noise() -> None:
    policy = SearchPolicy.named("fast-lock")

    assert policy.evaluate(qualified_apple_status(consecutive=4), "apple").qualified
    one_frame = policy.evaluate(qualified_apple_status(consecutive=1), "apple")
    assert one_frame.qualified is False
    assert one_frame.reason == "insufficient_consecutive_detections"


def test_slow_sweep_requires_three_apple_frames_but_five_pear_frames() -> None:
    policy = SearchPolicy.named("slow-sweep")

    apple = policy.evaluate(qualified_apple_status(consecutive=3), "apple")
    pear_status = qualified_apple_status(consecutive=3)
    pear_detection = pear_status["detection"]
    assert isinstance(pear_detection, dict)
    pear_detection["label"] = "pear"
    pear = policy.evaluate(pear_status, "pear")

    assert apple.qualified is True
    assert policy.required_consecutive_detections("apple") == 3
    assert pear.qualified is False
    assert pear.reason == "insufficient_consecutive_detections"
    assert policy.required_consecutive_detections("pear") == 5


@pytest.mark.parametrize(
    ("path", "unsafe_value", "reason"),
    (
        (("camera_healthy",), False, "camera_unhealthy"),
        (("detection", "label"), "pear", "wrong_target_fruit"),
        (("detection", "confidence"), 0.49, "confidence_below_acquisition_floor"),
        (("detection", "age_s"), 0.251, "detection_stale"),
        (("source", "age_s"), 0.351, "source_stale"),
        (("detection", "generation"), "camera-old", "generation_mismatch"),
        (("detection", "source_time_base"), "1/1000", "timebase_mismatch"),
        (("detection", "source_pts"), 901, "source_pts_invalid"),
        (("detection", "inference_s"), 0.201, "inference_too_slow"),
        (("detection", "bbox_area_ratio"), 0.0, "geometry_invalid"),
    ),
)
def test_fast_lock_preserves_every_motion_qualification_gate(
    path: tuple[str, ...],
    unsafe_value: object,
    reason: str,
) -> None:
    observation = deepcopy(qualified_apple_status())
    node = observation
    for key in path[:-1]:
        child = node[key]
        assert isinstance(child, dict)
        node = child
    node[path[-1]] = unsafe_value

    decision = SearchPolicy.named("fast-lock").evaluate(observation, "apple")

    assert decision.qualified is False
    assert decision.reason == reason


@pytest.mark.parametrize("name", ["", "baseline", "FAST", "double_back"])
def test_search_policy_rejects_unknown_or_noncanonical_names(name: str) -> None:
    with pytest.raises(ValueError, match="fast-lock, slow-sweep, double-back"):
        SearchPolicy.configured(
            cli_name=name,
            environ={"BORDER_COLLIE_SEARCH_POLICY": "slow-sweep"},
        )
