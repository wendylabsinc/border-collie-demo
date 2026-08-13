from __future__ import annotations

import wave
from datetime import timedelta

import pytest

from robotkit.executor.adapters import UnitreeSportAdapter, WavPlaybackAdapter
from robotkit.executor.mission import (
    MissionConfig,
    approach_apple,
    bark,
    go_home,
    approach_fruit,
    search_fruit,
    search_apple,
)
from tests.conftest import NOW


def _vision(observation_factory, *, center_x: float = 0.5, revision: int = 1, **kwargs):
    return observation_factory(
        stream="vision.fruits",
        revision=revision,
        payload={
            "detections": [
                {
                    "class_name": "apple",
                    "confidence": 0.91,
                    "bbox_xyxy_normalized": [
                        center_x - 0.05,
                        0.2,
                        center_x + 0.05,
                        0.7,
                    ],
                }
            ]
        },
        **kwargs,
    )


def _proximity(observation_factory, distance: float | None, *, revision: int = 2, **kwargs):
    observation = observation_factory(
        stream="lidar.proximity",
        revision=revision,
        payload={
            "representation": "angular_proximity_sectors_v1",
            "sectors": [
                {
                    "bearing_min_rad": -0.5,
                    "bearing_center_rad": 0.0,
                    "bearing_max_rad": 0.5,
                    "nearest_distance_m": distance,
                }
            ],
        },
        **kwargs,
    )
    return observation.model_copy(update={"frame_id": "base_link"})


def test_search_rotates_until_a_fresh_apple_is_seen(observation_factory, snapshot_factory):
    decision = search_apple(snapshot_factory(), at=NOW)
    assert decision.parameters == {"linear_x_mps": 0.0, "angular_z_rps": 0.45}
    assert not decision.completed

    seen = search_apple(snapshot_factory([_vision(observation_factory)]), at=NOW)
    assert seen.completed
    assert seen.parameters["angular_z_rps"] == 0.0


def test_search_does_not_accept_a_stale_detection(observation_factory, snapshot_factory):
    stale = _vision(
        observation_factory,
        observed_at=NOW - timedelta(seconds=3),
        ttl_seconds=1,
    )
    decision = search_apple(snapshot_factory([stale]), at=NOW)
    assert not decision.completed
    assert decision.parameters["angular_z_rps"] > 0


@pytest.mark.parametrize("target", ["pear", "grapes", "banana", "orange"])
def test_search_and_approach_support_other_fruits(
    target, observation_factory, snapshot_factory
):
    fruit = _vision(observation_factory).model_copy(
        update={
            "payload": {
                "detections": [
                    {
                        "class_name": target,
                        "confidence": 0.91,
                        "bbox_xyxy_normalized": [0.45, 0.2, 0.55, 0.7],
                    }
                ]
            }
        }
    )
    assert search_fruit(snapshot_factory([fruit]), target, at=NOW).completed

    proximity = _proximity(observation_factory, 1.0)
    decision = approach_fruit(
        snapshot_factory([fruit, proximity]), target, at=NOW
    )
    assert decision.parameters["target"] == target
    assert decision.parameters["linear_x_mps"] > 0


def test_approach_fuses_bbox_bearing_and_lidar_range(
    observation_factory, snapshot_factory
):
    observations = [
        _vision(observation_factory, center_x=0.4),
        _proximity(observation_factory, 1.2),
    ]
    decision = approach_apple(snapshot_factory(observations), at=NOW)
    assert decision.effect_type == "cmd_vel"
    assert 0 < decision.parameters["linear_x_mps"] <= 0.30
    assert 0 < decision.parameters["angular_z_rps"] <= 0.80
    assert decision.require_fresh_streams == ("vision.fruits", "lidar.proximity")


@pytest.mark.parametrize("missing", ["vision", "lidar", "range"])
def test_approach_never_advances_blindly(
    missing, observation_factory, snapshot_factory
):
    observations = []
    if missing != "vision":
        observations.append(_vision(observation_factory))
    if missing != "lidar":
        observations.append(
            _proximity(observation_factory, None if missing == "range" else 1.0)
        )
    decision = approach_apple(snapshot_factory(observations), at=NOW)
    assert decision.parameters["linear_x_mps"] == 0.0
    assert decision.parameters["angular_z_rps"] == 0.0
    assert not decision.completed


def test_approach_stops_at_thirty_centimeters(observation_factory, snapshot_factory):
    decision = approach_apple(
        snapshot_factory(
            [_vision(observation_factory), _proximity(observation_factory, 0.30)]
        ),
        at=NOW,
    )
    assert decision.completed
    assert decision.parameters["linear_x_mps"] == 0.0


def test_approach_rejects_wrong_lidar_frame(observation_factory, snapshot_factory):
    wrong_frame = _proximity(observation_factory, 1.0).model_copy(
        update={"frame_id": "utlidar_lidar"}
    )
    decision = approach_apple(
        snapshot_factory([_vision(observation_factory), wrong_frame]), at=NOW
    )
    assert decision.parameters["linear_x_mps"] == 0.0
    assert "frame" in decision.reason


def test_go_home_uses_pose_and_turns_before_translating_when_home_is_behind(
    observation_factory, snapshot_factory
):
    pose = observation_factory(
        stream="localization.pose",
        payload={"x_m": 1.0, "y_m": 0.0, "yaw_rad": 0.0},
    )
    decision = go_home(snapshot_factory([pose]), at=NOW)
    assert decision.parameters["linear_x_mps"] == 0.0
    assert abs(decision.parameters["angular_z_rps"]) <= 0.8
    assert decision.require_fresh_streams == ("localization.pose",)


def test_go_home_drives_bounded_and_completes_at_home(
    observation_factory, snapshot_factory
):
    config = MissionConfig(home_x_m=1.0, home_y_m=0.0, home_yaw_rad=0.0)
    away = observation_factory(
        stream="localization.pose",
        payload={"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0},
    )
    decision = go_home(snapshot_factory([away]), at=NOW, config=config)
    assert 0 < decision.parameters["linear_x_mps"] <= 0.30
    assert decision.parameters["angular_z_rps"] == 0.0

    home = observation_factory(
        stream="localization.pose",
        payload={"x_m": 1.05, "y_m": 0.0, "yaw_rad": 0.05},
    )
    completed = go_home(snapshot_factory([home]), at=NOW, config=config)
    assert completed.completed


def test_go_home_refuses_stale_or_invalid_localization(
    observation_factory, snapshot_factory
):
    stale = observation_factory(
        stream="localization.pose",
        observed_at=NOW - timedelta(seconds=3),
        ttl_seconds=1,
        payload={"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0},
    )
    decision = go_home(snapshot_factory([stale]), at=NOW)
    assert decision.parameters == {"linear_x_mps": 0.0, "angular_z_rps": 0.0}


def test_bark_is_a_semantic_audio_effect():
    decision = bark()
    assert decision.effect_type == "unitree_bark"
    assert decision.parameters == {"sound": "bark"}


def test_wav_adapter_validates_and_plays_configured_asset(tmp_path, effect_factory):
    path = tmp_path / "bark.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(8_000)
        wav.writeframes(b"\0\0" * 800)

    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))

    adapter = WavPlaybackAdapter(
        str(path), runner=run, which=lambda name: f"/usr/bin/{name}" if name == "aplay" else None
    )
    effect = effect_factory().model_copy(
        update={"effect_type": "unitree_bark", "parameters": {"sound": "bark"}}
    )
    result = adapter.apply(effect)
    assert calls[0][0] == ["/usr/bin/aplay", str(path)]
    assert calls[0][1]["check"] is True
    assert result["duration_seconds"] == 0.1


def test_wav_adapter_never_uses_an_effect_supplied_path(tmp_path, effect_factory):
    configured = tmp_path / "configured.wav"
    with wave.open(str(configured), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(8_000)
        wav.writeframes(b"\0\0" * 80)
    commands = []
    adapter = WavPlaybackAdapter(
        str(configured),
        runner=lambda command, **kwargs: commands.append(command),
        which=lambda _: "/bin/player",
    )
    effect = effect_factory().model_copy(
        update={
            "effect_type": "unitree_bark",
            "parameters": {"sound": "bark", "path": str(tmp_path / "evil.wav")},
        }
    )
    adapter.apply(effect)
    assert commands == [["/bin/player", str(configured)]]


def test_unitree_sport_adapter_maps_velocity_and_lie_down(effect_factory):
    class FakeSportClient:
        def __init__(self):
            self.calls = []

        def Move(self, x, y, yaw):
            self.calls.append(("Move", x, y, yaw))
            return 0

        def StandDown(self):
            self.calls.append(("StandDown",))
            return 0

        def StopMove(self):
            self.calls.append(("StopMove",))
            return 0

    client = FakeSportClient()
    adapter = UnitreeSportAdapter(client=client)
    velocity = effect_factory().model_copy(
        update={
            "effect_type": "cmd_vel",
            "parameters": {"linear_x_mps": 0.2, "angular_z_rps": -0.3},
        }
    )
    lie = effect_factory().model_copy(
        update={"effect_type": "unitree_lie_down", "parameters": {}}
    )

    velocity_result = adapter.apply(velocity)
    lie_result = adapter.apply(lie)
    adapter.close()

    assert client.calls == [
        ("Move", 0.2, 0.0, -0.3),
        ("StandDown",),
        ("StopMove",),
    ]
    assert velocity_result["api"] == "Move"
    assert lie_result["api"] == "StandDown"


def test_unitree_sport_adapter_deadman_stops_unrenewed_motion(effect_factory):
    class FakeSportClient:
        def __init__(self):
            self.calls = []

        def Move(self, x, y, yaw):
            self.calls.append(("Move", x, y, yaw))
            return 0

        def StopMove(self):
            self.calls.append(("StopMove",))
            return 0

    class FakeTimer:
        def __init__(self, interval, callback):
            self.interval = interval
            self.callback = callback
            self.daemon = False
            self.cancelled = False

        def start(self):
            pass

        def cancel(self):
            self.cancelled = True

    timers = []

    def timer_factory(interval, callback):
        timer = FakeTimer(interval, callback)
        timers.append(timer)
        return timer

    client = FakeSportClient()
    adapter = UnitreeSportAdapter(
        client=client,
        watchdog_seconds=0.8,
        timer_factory=timer_factory,
    )
    velocity = effect_factory().model_copy(
        update={
            "effect_type": "cmd_vel",
            "parameters": {"linear_x_mps": 0.2, "angular_z_rps": 0.0},
        }
    )

    adapter.apply(velocity)
    assert timers[0].interval == 0.8
    timers[0].callback()

    assert client.calls == [("Move", 0.2, 0.0, 0.0), ("StopMove",)]


def test_unitree_sport_adapter_ignores_cancelled_watchdog_race(effect_factory):
    class FakeSportClient:
        def __init__(self):
            self.calls = []

        def Move(self, x, y, yaw):
            self.calls.append(("Move", x, y, yaw))
            return 0

        def StopMove(self):
            self.calls.append(("StopMove",))
            return 0

    class FakeTimer:
        def __init__(self, interval, callback):
            self.callback = callback
            self.daemon = False

        def start(self):
            pass

        def cancel(self):
            pass

    timers = []

    def timer_factory(interval, callback):
        timer = FakeTimer(interval, callback)
        timers.append(timer)
        return timer

    client = FakeSportClient()
    adapter = UnitreeSportAdapter(client=client, timer_factory=timer_factory)
    velocity = effect_factory().model_copy(
        update={
            "effect_type": "cmd_vel",
            "parameters": {"linear_x_mps": 0.0, "angular_z_rps": 0.45},
        }
    )

    adapter.apply(velocity)
    adapter.apply(velocity)
    timers[0].callback()

    assert client.calls == [
        ("Move", 0.0, 0.0, 0.45),
        ("Move", 0.0, 0.0, 0.45),
    ]
