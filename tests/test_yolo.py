import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from robotkit.contracts import PublishResult
from robotkit.perception.yolo.adapters import DetectionFrame, UltralyticsDetector
from robotkit.perception.yolo.core import Detection, interpret_detections
from robotkit.perception.yolo.producer import YoloProducer
from robotkit.perception.yolo import run


def test_interpretation_filters_prompted_fruits_and_normalizes_boxes():
    interpreted = interpret_detections(
        [
            Detection(47, "apple", 0.8, (-10, 20, 110, 80)),
            Detection(46, "banana", 0.95, (20, 10, 60, 50)),
            Detection(49, "orange", 0.1, (1, 1, 2, 2)),
            Detection(50, "pear", 0.75, (10, 10, 30, 40)),
            Detection(51, "grapes", 0.7, (30, 10, 50, 40)),
            Detection(0, "person", 0.99, (0, 0, 100, 100)),
            Detection(47, "apple", 0.9, (10, 10, 10, 20)),
        ],
        image_width=100,
        image_height=100,
        min_confidence=0.25,
    )

    assert interpreted.stream == "vision.fruits"
    assert interpreted.observation_type == "vision.prompted_fruits.v1"
    assert interpreted.confidence == 0.95
    assert interpreted.payload["count"] == 4
    assert [item["class_name"] for item in interpreted.payload["detections"]] == [
        "banana",
        "apple",
        "pear",
        "grapes",
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


def test_ultralytics_adapter_configures_yoloe_prompts_once():
    result = SimpleNamespace(orig_shape=(10, 20), boxes=None, names={})

    class FakeYoloE:
        def __init__(self):
            self.prompt_calls = []

        def get_text_pe(self, names):
            self.prompt_calls.append(("embed", names))
            return "embeddings"

        def set_classes(self, names, embeddings):
            self.prompt_calls.append(("classes", names, embeddings))

        def predict(self, **kwargs):
            return [result]

    model = FakeYoloE()
    detector = UltralyticsDetector(
        model=model,
        classes=["apple", "pear"],
    )
    detector.detect("first.jpg")
    detector.detect("second.jpg")

    assert model.prompt_calls == [
        ("embed", ["apple", "pear"]),
        ("classes", ["apple", "pear"], "embeddings"),
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
    assert observation.producer_id == "yoloe-fruits"
    assert observation.deployment_generation == 7
    assert observation.stream == "vision.fruits"
    assert observation.observation_type == "vision.prompted_fruits.v1"
    assert observation.frame_id == "front_camera"
    assert observation.payload["detections"][0]["class_name"] == "orange"
    assert observation.idempotency_key == publisher.observations[2].idempotency_key
    diagnostic = publisher.observations[1]
    assert diagnostic.stream == "diagnostics.yolo"
    assert diagnostic.payload["status"] == "ok"
    assert diagnostic.payload["detection_count"] == 1
    assert diagnostic.idempotency_key == publisher.observations[3].idempotency_key


def test_producer_publishes_durable_failure_diagnostic_before_reraising():
    class BrokenDetector:
        def detect(self, image):
            raise RuntimeError("model unavailable")

    class FakePublisher:
        def __init__(self):
            self.observations = []

        def publish_observation(self, observation):
            self.observations.append(observation)
            return PublishResult(revision=len(self.observations), duplicate=False)

    publisher = FakePublisher()
    producer = YoloProducer(
        BrokenDetector(), publisher, instance_id="green-1", model_name="yolo11n.pt"
    )

    with pytest.raises(RuntimeError, match="model unavailable"):
        producer.process_image(b"pixels", source_key="camera:broken")

    assert len(publisher.observations) == 1
    diagnostic = publisher.observations[0]
    assert diagnostic.stream == "diagnostics.yolo"
    assert diagnostic.payload["status"] == "failed"
    assert diagnostic.payload["model"] == "yolo11n.pt"
    assert "model unavailable" in diagnostic.payload["error"]


def test_go2_webrtc_retries_transient_connection_failure(monkeypatch):
    attempts = []
    diagnostics = []
    sleeps = []

    async def fail_session(producer):
        attempts.append(producer)
        if len(attempts) == 3:
            raise asyncio.CancelledError
        raise RuntimeError("ICE still checking")

    async def record_sleep(seconds):
        sleeps.append(seconds)

    class FakeProducer:
        def publish_failure(self, error, *, source_key):
            diagnostics.append((str(error), source_key))

    producer = FakeProducer()
    monkeypatch.setattr(run, "_run_go2_webrtc_async", fail_session)
    monkeypatch.setattr(run.asyncio, "sleep", record_sleep)
    monkeypatch.setenv("GO2_RETRY_SECONDS", "0.25")

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(run._run_go2_webrtc_forever(producer))

    assert attempts == [producer, producer, producer]
    assert sleeps == [0.25, 0.25]
    assert [message for message, _ in diagnostics] == [
        "ICE still checking",
        "ICE still checking",
    ]
    assert all(key.startswith("go2-webrtc:") for _, key in diagnostics)


def test_http_camera_reuses_existing_feed(monkeypatch):
    requests = []
    processed = []

    class FakeResponse:
        content = b"jpeg-frame"

        def raise_for_status(self):
            return None

    class FakeClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.closed = False

        def get(self, url, *, headers):
            requests.append((url, headers))
            return FakeResponse()

        def close(self):
            self.closed = True

    class FakeProducer:
        def process_image(self, image, **kwargs):
            processed.append((image, kwargs))

        def publish_failure(self, error, *, source_key):
            raise AssertionError(f"unexpected failure: {error} ({source_key})")

    clients = []

    def make_client(**kwargs):
        client = FakeClient(**kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(run.httpx, "Client", make_client)
    monkeypatch.setattr(run, "decode_image_bytes", lambda payload: ("image", payload))
    monkeypatch.setattr(run, "run_loop", lambda step, interval: step())
    monkeypatch.setenv("YOLO_CAMERA_URL", "http://127.0.0.1:8111/frame.jpg")

    run.run_http(FakeProducer())

    assert requests == [
        (
            "http://127.0.0.1:8111/frame.jpg",
            {"Accept": "image/jpeg,image/*", "Cache-Control": "no-cache"},
        )
    ]
    assert processed[0][0] == ("image", b"jpeg-frame")
    assert processed[0][1]["source_key"].startswith(
        "http:http://127.0.0.1:8111/frame.jpg:"
    )
    assert processed[0][1]["frame_id"] == "camera_link"
    assert clients[0].closed is True


def test_go2_dds_camera_uses_video_service(monkeypatch):
    processed = []
    failures = []

    class FakeClient:
        def GetImageSample(self):
            return 0, [0xFF, 0xD8, 0xFF, 0xE0]

    class FakeProducer:
        def process_image(self, image, **kwargs):
            processed.append((image, kwargs))

        def publish_failure(self, error, *, source_key):
            failures.append((error, source_key))

    monkeypatch.setattr(run, "_make_go2_video_client", FakeClient)
    monkeypatch.setattr(run, "decode_image_bytes", lambda payload: ("image", payload))
    monkeypatch.setattr(run, "run_loop", lambda step, interval: step())

    run.run_go2_dds(FakeProducer())

    assert failures == []
    assert processed[0][0] == ("image", b"\xff\xd8\xff\xe0")
    assert processed[0][1]["source_key"].startswith("go2-dds:")
    assert processed[0][1]["frame_id"] == "camera_link"


def test_go2_dds_camera_reports_video_service_error(monkeypatch):
    failures = []

    class FakeClient:
        def GetImageSample(self):
            return 3104, []

    class FakeProducer:
        def process_image(self, image, **kwargs):
            raise AssertionError("failed frame must not be processed")

        def publish_failure(self, error, *, source_key):
            failures.append((str(error), source_key))

    monkeypatch.setattr(run, "_make_go2_video_client", FakeClient)
    monkeypatch.setattr(run, "run_loop", lambda step, interval: step())

    with pytest.raises(RuntimeError, match="3104"):
        run.run_go2_dds(FakeProducer())

    assert failures[0][0] == "Go2 video service returned error code 3104"
    assert failures[0][1].startswith("go2-dds:")
