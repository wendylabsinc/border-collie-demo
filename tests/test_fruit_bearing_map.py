import math

import pytest

from border_collie_demo.fruit_bearing_map import FruitBearingMap

HOME = {
    "x_m": 1.0,
    "y_m": -2.0,
    "yaw_rad": 0.0,
    "age_s": 0.02,
}
FRAME = {
    "generation": "camera-a",
    "source_pts": 100,
    "source_time_base": "1/90000",
    "odometry_epoch": "odom-a",
}


def observation(label: str, confidence: float, center_x: float) -> dict[str, object]:
    return {
        "label": label,
        "confidence": confidence,
        "center_x_ratio": center_x,
        "source_pts": FRAME["source_pts"],
        "source_time_base": FRAME["source_time_base"],
        "generation": FRAME["generation"],
    }


def test_one_scan_maps_every_centered_supported_fruit_and_routes_shortest_turn() -> None:
    bearing_map = FruitBearingMap(clock=lambda: 10.0)
    pose = {**HOME, "yaw_rad": math.radians(30.0)}

    status = bearing_map.observe(
        home_pose=HOME,
        robot_pose=pose,
        frame_identity=FRAME,
        observations={
            "apple": observation("apple", 0.72, 0.49),
            "banana": observation("banana", 0.61, 0.50),
            "pear": observation("pear", 0.83, 0.51),
        },
    )
    route = bearing_map.route_to(
        "apple",
        {**HOME, "generation": "camera-a", "odometry_epoch": "odom-a"},
    )

    assert set(status["fruits"]) == {"apple", "banana", "pear"}
    assert route.available is True
    assert route.angular_delta_rad == pytest.approx(math.radians(30.0))
    assert route.direction == "left"
    assert route.evidence["sample_count"] == 1


def test_advisory_map_accepts_real_search_sample_inside_fine_focus_corridor() -> None:
    bearing_map = FruitBearingMap(clock=lambda: 10.0)

    status = bearing_map.observe(
        home_pose=HOME,
        robot_pose={**HOME, "yaw_rad": -1.15},
        frame_identity=FRAME,
        observations={"pear": observation("pear", 0.809, 0.39)},
    )

    assert status["valid"] is True
    assert status["fruits"]["pear"]["sample_count"] == 1
    assert status["thresholds"]["center_tolerance_ratio"] == 0.12


def test_off_axis_observation_is_diagnostic_only_and_cannot_create_route() -> None:
    bearing_map = FruitBearingMap(clock=lambda: 10.0)

    status = bearing_map.observe(
        home_pose=HOME,
        robot_pose={**HOME, "yaw_rad": 1.0},
        frame_identity=FRAME,
        observations={"pear": observation("pear", 0.95, 0.80)},
    )
    route = bearing_map.route_to(
        "pear",
        {**HOME, "generation": "camera-a", "odometry_epoch": "odom-a"},
    )

    assert status["valid"] is False
    assert status["invalidation_reason"] == "waiting_for_centered_observation"
    assert route.available is False
    assert route.reason == "map_unanchored"


def test_route_uses_current_home_heading_even_when_home_position_changed() -> None:
    bearing_map = FruitBearingMap(clock=lambda: 10.0)
    bearing_map.observe(
        home_pose=HOME,
        robot_pose={**HOME, "yaw_rad": math.radians(120.0)},
        frame_identity=FRAME,
        observations={"pear": observation("pear", 0.8, 0.5)},
    )

    route = bearing_map.route_to(
        "pear",
        {
            **HOME,
            "x_m": 1.50,
            "y_m": -1.50,
            "yaw_rad": math.radians(-170.0),
            "generation": "camera-a",
            "odometry_epoch": "odom-a",
        },
    )

    assert route.available is True
    assert route.angular_delta_rad == pytest.approx(math.radians(-70.0))
    assert route.direction == "right"
    assert bearing_map.status()["valid"] is True
    assert bearing_map.status()["current_home_position_error_m"] == pytest.approx(
        math.hypot(0.5, 0.5)
    )


def test_home_yaw_difference_is_route_input_not_map_invalidation() -> None:
    bearing_map = FruitBearingMap(clock=lambda: 10.0)
    bearing_map.observe(
        home_pose=HOME,
        robot_pose=HOME,
        frame_identity=FRAME,
        observations={"pear": observation("pear", 0.8, 0.5)},
    )

    route = bearing_map.route_to(
        "pear",
        {
            **HOME,
            "yaw_rad": math.radians(5.01),
            "generation": "camera-a",
            "odometry_epoch": "odom-a",
        },
    )

    assert route.available is True
    assert route.angular_delta_rad == pytest.approx(math.radians(-5.01))
    assert route.direction == "right"
    assert bearing_map.status()["current_home_yaw_error_deg"] == pytest.approx(5.01)


def test_camera_generation_change_invalidates_map() -> None:
    bearing_map = FruitBearingMap(clock=lambda: 10.0)
    bearing_map.observe(
        home_pose=HOME,
        robot_pose=HOME,
        frame_identity=FRAME,
        observations={"apple": observation("apple", 0.8, 0.5)},
    )

    route = bearing_map.route_to(
        "apple",
        {**HOME, "generation": "camera-b", "odometry_epoch": "odom-a"},
    )

    assert route.available is False
    assert route.reason == "camera_generation_changed"


def test_wraparound_uses_shortest_signed_angular_delta() -> None:
    bearing_map = FruitBearingMap(clock=lambda: 10.0)
    bearing_map.observe(
        home_pose={**HOME, "yaw_rad": math.radians(-179.0)},
        robot_pose={**HOME, "yaw_rad": math.radians(179.0)},
        frame_identity=FRAME,
        observations={"banana": observation("banana", 0.8, 0.5)},
    )

    route = bearing_map.route_to(
        "banana",
        {
            **HOME,
            "yaw_rad": math.radians(-179.0),
            "generation": "camera-a",
            "odometry_epoch": "odom-a",
        },
    )

    assert route.angular_delta_rad == pytest.approx(math.radians(-2.0))
    assert route.direction == "right"


def test_new_process_instance_has_no_persisted_map() -> None:
    first = FruitBearingMap(clock=lambda: 10.0)
    first.observe(
        home_pose=HOME,
        robot_pose=HOME,
        frame_identity=FRAME,
        observations={"pear": observation("pear", 0.8, 0.5)},
    )

    restarted = FruitBearingMap(clock=lambda: 11.0)

    assert restarted.status()["valid"] is False
    assert restarted.route_to("pear", HOME).reason == "map_unanchored"


def test_contradictory_centered_bearing_invalidates_instead_of_averaging() -> None:
    bearing_map = FruitBearingMap(clock=lambda: 10.0)
    bearing_map.observe(
        home_pose=HOME,
        robot_pose=HOME,
        frame_identity=FRAME,
        observations={"pear": observation("pear", 0.8, 0.5)},
    )
    second_frame = {**FRAME, "source_pts": 101}
    second_observation = observation("pear", 0.9, 0.5)
    second_observation["source_pts"] = 101

    status = bearing_map.observe(
        home_pose=HOME,
        robot_pose={**HOME, "yaw_rad": math.radians(50.0)},
        frame_identity=second_frame,
        observations={"pear": second_observation},
    )

    assert status["valid"] is False
    assert status["invalidation_reason"] == "contradictory_pear_bearing"
    assert status["fruits"] == {}


def test_duplicate_frame_does_not_inflate_sample_count() -> None:
    bearing_map = FruitBearingMap(clock=lambda: 10.0)
    kwargs = {
        "home_pose": HOME,
        "robot_pose": HOME,
        "frame_identity": FRAME,
        "observations": {"pear": observation("pear", 0.8, 0.5)},
    }

    bearing_map.observe(**kwargs)
    status = bearing_map.observe(**kwargs)

    assert status["last_observation_reason"] == "duplicate_frame"
    assert status["fruits"]["pear"]["sample_count"] == 1


def test_expired_bearing_falls_back_to_bounded_search() -> None:
    now = 10.0
    bearing_map = FruitBearingMap(clock=lambda: now, maximum_bearing_age_s=2.0)
    bearing_map.observe(
        home_pose=HOME,
        robot_pose=HOME,
        frame_identity=FRAME,
        observations={"pear": observation("pear", 0.8, 0.5)},
    )
    now = 12.01

    route = bearing_map.route_to(
        "pear",
        {**HOME, "generation": "camera-a", "odometry_epoch": "odom-a"},
    )

    assert route.available is False
    assert route.reason == "bearing_stale"
