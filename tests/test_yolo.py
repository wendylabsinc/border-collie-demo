from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from robotkit.contracts import PublishResult
from robotkit.perception.yolo.adapters import DetectionFrame, UltralyticsDetector
from robotkit.perception.yolo.core import Detection, interpret_detections
from robotkit.perception.yolo.producer import YoloProducer


def test_interpretation_filters_coco_fruits_and_normalizes_boxes():
    interpreted = interpret_detections(
        [
            Detection(47, "apple", 0.8, (-10, 20, 110, 80)),
            Detection(46, "banana", 0.95, (20, 10, 60, 50)),
            Detection(49, "orange", 0.1, (1, 1, 2, 2)),
            Detection(0, "person", 0.99, (0, 0, 100, 100)),
            Detection(47, "apple", 0.9, (10, 10, 10, 20)),
        ],
        image_width=100,
        image_height=100,
        min_confidence=0.25,
    )

    assert interpreted.stream == "vision.fruits"
    assert interpreted.observation_type == "vision.coco_fruits.v1"
    assert interpreted.confidence == 0.95
    assert interpreted.payload["count"] == 2
    assert [item["class_name"] for item in interpreted.payload["detections"]] == [
        "banana",
        "apple",
    ]
    assert interpreted.payload["detections"][1]["bbox_xyxy_px"] == [
        0.0,
        20.0,
        100.0,
        80.0,
    ]
    assert interpreted.payload["detections"][1]["bbox_xyxy_normalized"] == [
        0.0,
        0.2,
        1.0,
        0.8,
    ]


def test_interpretation_publishes_explicit_empty_frame():
    interpreted = interpret_detections([], image_width=640, image_height=480)

    assert interpreted.confidence is None
    assert interpreted.payload["count"] == 0
    assert interpreted.payload["detections"] == []


@pytest.mark.parametrize("width,height", [(0, 10), (10, -1)])
def test_interpretation_rejects_invalid_dimensions(width, height):
    with pytest.raises(ValueError, match="dimensions"):
        interpret_detections([], image_width=width, image_height=height)


def test_ultralytics_adapter_parses_injected_model_without_runtime_dependency():
    boxes = SimpleNamespace(
        cls=[46.0, 47.0],
        conf=[0.92, 0.81],
        xyxy=[[1.0, 2.0, 30.0, 40.0], [50.0, 60.0, 80.0, 90.0]],
    )
    result = SimpleNamespace(
        orig_shape=(480, 640), boxes=boxes, names={46: "banana", 47: "apple"}
    )

    class FakeModel:
        def __init__(self):
            self.calls = []

        def predict(self, **kwargs):
            self.calls.append(kwargs)
            return [result]

    model = FakeModel()
    frame = UltralyticsDetector(model=model, confidence=0.4, device="cpu").detect("frame.jpg")

    assert (frame.width, frame.height) == (640, 480)
    assert frame.detections[0] == Detection(
        46, "banana", 0.92, (1.0, 2.0, 30.0, 40.0)
    )
    assert model.calls == [
        {"source": "frame.jpg", "conf": 0.4, "verbose": False, "device": "cpu"}
    ]


def test_producer_builds_versioned_idempotent_observation():
    class FakeDetector:
        def detect(self, image):
            assert image == b"pixels"
            return DetectionFrame(
                (Detection(49, "orange", 0.88, (10, 10, 30, 30)),), 100, 50
            )

    class FakePublisher:
        def __init__(self):
            self.observations = []

        def publish_observation(self, observation):
            self.observations.append(observation)
            return PublishResult(revision=len(self.observations), duplicate=False)

    publisher = FakePublisher()
    producer = YoloProducer(
        FakeDetector(),
        publisher,
        instance_id="green-1",
        deployment_generation=7,
    )
    at = datetime(2026, 8, 12, tzinfo=timezone.utc)

    result = producer.process_image(
        b"pixels", source_key="camera:42", observed_at=at, frame_id="front_camera"
    )
    producer.process_image(
        b"pixels", source_key="camera:42", observed_at=at, frame_id="front_camera"
    )

    observation = publisher.observations[0]
    assert result.revision == 1
    assert observation.schema_version == "1"
    assert observation.producer_id == "yolo-coco-fruits"
    assert observation.deployment_generation == 7
    assert observation.stream == "vision.fruits"
    assert observation.observation_type == "vision.coco_fruits.v1"
    assert observation.frame_id == "front_camera"
    assert observation.payload["detections"][0]["class_name"] == "orange"
    assert observation.idempotency_key == publisher.observations[1].idempotency_key
