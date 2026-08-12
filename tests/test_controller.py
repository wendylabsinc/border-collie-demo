from robotkit.controller.logic import control
from robotkit.controller.service import control_tick
from tests.conftest import NOW


def _mission_goal(goal_factory, stage, *, mission_type="apple"):
    return goal_factory(
        stage,
        {
            "mission_schema_version": "1",
            "mission_type": mission_type,
            "mission_stage": stage,
            "trigger_event_id": "voice-event-1",
        },
    )


def test_idle_at_home_produces_zero_velocity(snapshot_factory, goal_factory):
    effect = control(
        snapshot_factory(),
        _mission_goal(goal_factory, "idle_at_home", mission_type="idle"),
    )

    assert effect.effect_type == "cmd_vel"
    assert effect.parameters["linear_x_mps"] == 0.0
    assert effect.parameters["angular_z_rps"] == 0.0
    assert effect.parameters["stage_complete"] is True


def test_search_rotates_until_an_apple_is_visible(snapshot_factory, goal_factory):
    effect = control(
        snapshot_factory(), _mission_goal(goal_factory, "search_apple")
    )

    assert effect.effect_type == "cmd_vel"
    assert effect.parameters["linear_x_mps"] == 0.0
    assert 0 < effect.parameters["angular_z_rps"] <= 1.0
    assert effect.parameters["stage_complete"] is False


def test_control_renewal_tick_is_time_bucketed_and_replica_stable():
    assert control_tick(NOW, 0.25) == control_tick(NOW, 0.25)
    assert control_tick(NOW, 0.25) != control_tick(
        NOW.replace(microsecond=NOW.microsecond + 250_000), 0.25
    )


def test_approach_refuses_blind_motion_without_lidar(
    observation_factory, snapshot_factory, goal_factory
):
    fruit = observation_factory(
        stream="vision.fruits",
        payload={
            "detections": [
                {
                    "class_name": "apple",
                    "confidence": 0.9,
                    "bbox_xyxy_normalized": [0.4, 0.2, 0.6, 0.8],
                }
            ]
        },
    )
    effect = control(
        snapshot_factory([fruit]),
        _mission_goal(goal_factory, "approach_apple"),
    )

    assert effect.parameters["linear_x_mps"] == 0.0
    assert effect.parameters["stage_complete"] is False
    assert "refusing blind advance" in effect.parameters["decision_reason"]


def test_approach_advances_with_fused_fresh_range(
    observation_factory, snapshot_factory, goal_factory
):
    fruit = observation_factory(
        stream="vision.fruits",
        payload={
            "detections": [
                {
                    "class_name": "apple",
                    "confidence": 0.9,
                    "bbox_xyxy_normalized": [0.45, 0.2, 0.55, 0.8],
                }
            ]
        },
        revision=1,
    )
    proximity = observation_factory(
        stream="lidar.proximity",
        payload={
            "sectors": [
                {
                    "bearing_min_rad": -0.2,
                    "bearing_max_rad": 0.2,
                    "nearest_distance_m": 1.0,
                }
            ]
        },
        revision=2,
    ).model_copy(update={"frame_id": "base_link"})

    effect = control(
        snapshot_factory([fruit, proximity]),
        _mission_goal(goal_factory, "approach_apple"),
    )

    assert 0 < effect.parameters["linear_x_mps"] <= 0.3
    assert effect.parameters["stage_complete"] is False
    assert effect.require_fresh_streams == ["vision.fruits", "lidar.proximity"]


def test_approach_completes_only_inside_thirty_centimetres(
    observation_factory, snapshot_factory, goal_factory
):
    fruit = observation_factory(
        stream="vision.fruits",
        payload={
            "detections": [
                {
                    "class_name": "apple",
                    "bbox_xyxy_normalized": [0.4, 0.2, 0.6, 0.8],
                }
            ]
        },
        revision=1,
    )
    proximity = observation_factory(
        stream="lidar.proximity",
        payload={
            "sectors": [
                {
                    "bearing_min_rad": -0.2,
                    "bearing_max_rad": 0.2,
                    "nearest_distance_m": 0.30,
                }
            ]
        },
        revision=2,
    ).model_copy(update={"frame_id": "base_link"})

    effect = control(
        snapshot_factory([fruit, proximity]),
        _mission_goal(goal_factory, "approach_apple"),
    )

    assert effect.parameters["linear_x_mps"] == 0.0
    assert effect.parameters["stage_complete"] is True


def test_bark_and_lie_down_have_narrow_safe_parameters(snapshot_factory, goal_factory):
    bark = control(snapshot_factory(), _mission_goal(goal_factory, "bark"))
    lie = control(snapshot_factory(), _mission_goal(goal_factory, "lie_down"))

    assert (bark.effect_type, bark.parameters) == ("unitree_bark", {"sound": "bark"})
    assert (lie.effect_type, lie.parameters) == ("unitree_lie_down", {})


def test_go_home_uses_pose_and_reports_completion(
    observation_factory, snapshot_factory, goal_factory
):
    pose = observation_factory(
        stream="localization.pose",
        payload={"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0},
    )
    effect = control(
        snapshot_factory([pose]),
        _mission_goal(goal_factory, "go_home", mission_type="health_return"),
    )

    assert effect.effect_type == "cmd_vel"
    assert effect.parameters["linear_x_mps"] == 0.0
    assert effect.parameters["stage_complete"] is True
    assert effect.require_fresh_streams == []
