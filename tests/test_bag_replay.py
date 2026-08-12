import json

import pytest

from robotkit.testing.bag_replay import (
    is_json_subset,
    load_expectations,
    unmatched_expectations,
)


def test_nested_observation_expectations_are_unordered_subsets():
    observation = {
        "stream": "vision.fruits",
        "payload": {
            "detections": [
                {"class_name": "banana", "confidence": 0.8},
                {"class_name": "apple", "confidence": 0.9},
            ]
        },
        "revision": 4,
    }

    assert is_json_subset(
        observation,
        {
            "stream": "vision.fruits",
            "payload": {"detections": [{"class_name": "apple"}]},
        },
    )
    assert not is_json_subset(observation, {"payload": {"detections": []}})


def test_unmatched_expectations_report_only_missing_observations():
    observations = [
        {"stream": "health.temperature", "payload": {"band": "normal"}}
    ]
    expectations = [
        {"stream": "health.temperature", "payload": {"band": "normal"}},
        {"stream": "health.battery"},
    ]

    assert unmatched_expectations(observations, expectations) == [
        {"stream": "health.battery"}
    ]


def test_expectation_file_accepts_wrapped_observation_list(tmp_path):
    path = tmp_path / "expected.json"
    path.write_text(
        json.dumps(
            {"observations": [{"stream": "lidar.proximity", "frame_id": "base_link"}]}
        )
    )

    assert load_expectations(["lidar.room_map"], path) == [
        {"stream": "lidar.room_map"},
        {"stream": "lidar.proximity", "frame_id": "base_link"},
    ]


def test_expectation_file_requires_a_stream(tmp_path):
    path = tmp_path / "expected.json"
    path.write_text('[{"payload": {"band": "normal"}}]')

    with pytest.raises(ValueError, match="must contain a stream"):
        load_expectations([], path)
