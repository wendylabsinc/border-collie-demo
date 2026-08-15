import asyncio
import io
import json
import logging
import time
import zipfile
from types import SimpleNamespace

from fastapi.testclient import TestClient

from media import perception_sidecar
from media.coco_tester import CocoTester
from media.model_router import FruitModelRouter
from media.perception_sidecar import EvidenceFrameBuffer, PerceptionEvidence, create_app


class FakeTensor:
    def __init__(self, value):
        self.value = value

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.value


class FakeBoxes:
    def __init__(self, confidences, boxes):
        self.conf = FakeTensor(confidences)
        self.xyxy = [FakeTensor(box) for box in boxes]

    def __len__(self):
        return len(self.conf.value)


class ArrayFrame:
    def __init__(self, image):
        self.image = image

    def to_ndarray(self, *, format: str):
        assert format == "bgr24"
        return self.image


class FakeImage:
    def __init__(self, height: int, width: int) -> None:
        self.shape = (height, width, 3)

    def __getitem__(self, slices):
        y_slice, x_slice = slices[:2]
        return FakeImage(y_slice.stop - y_slice.start, x_slice.stop - x_slice.start)


class FakeClasses:
    def __init__(self, values):
        self.values = values

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.values


def test_coco_tester_ranks_all_frame_confidence_without_touching_target_evidence() -> None:
    class Boxes:
        def __init__(self) -> None:
            self.conf = FakeTensor([0.80, 0.60])
            self.cls = FakeClasses([47, 49])
            self.xyxy = [
                FakeTensor([10, 20, 100, 200]),
                FakeTensor([30, 40, 120, 220]),
            ]

        def __len__(self):
            return 2

    class Model:
        def __init__(self) -> None:
            self.names = {47: "apple", 49: "orange"}

        def predict(self, **options):
            assert options["conf"] == 0.05
            assert "classes" not in options
            return [SimpleNamespace(boxes=Boxes())]

    tester = CocoTester(model_loader=lambda _path: Model(), clock=lambda: 10.0)

    enabled = tester.configure(enabled=True, minimum_confidence=0.05, reset=True)
    first = tester.observe(object(), pts=100, now_s=10.0)
    second = tester.observe(object(), pts=101, now_s=10.5)

    assert enabled["class_count"] == 2
    assert first is True and second is True
    status = tester.status()
    assert status["enabled"] is True
    assert status["frames_processed"] == 2
    assert status["classes"][0] == {
        "label": "apple",
        "class_id": 47,
        "latest_confidence": 0.8,
        "mean_detected_confidence": 0.8,
        "all_frame_score": 0.8,
        "maximum_confidence": 0.8,
        "detection_rate": 1.0,
        "frames_detected": 2,
    }
    assert status["classes"][1]["label"] == "orange"
    assert status["classes"][1]["all_frame_score"] == 0.6


def test_coco_tester_is_disabled_by_default_and_rate_limits_extra_inference() -> None:
    calls = 0

    class Model:
        def __init__(self) -> None:
            self.names = {47: "apple"}

        def predict(self, **_options):
            nonlocal calls
            calls += 1
            return [SimpleNamespace(boxes=FakeBoxes([], []))]

    tester = CocoTester(model_loader=lambda _path: Model(), interval_s=0.5)

    assert tester.observe(object(), pts=1, now_s=1.0) is False
    tester.configure(enabled=True, minimum_confidence=0.05, reset=True)
    assert tester.observe(object(), pts=2, now_s=2.0) is True
    assert tester.observe(object(), pts=3, now_s=2.2) is False
    assert tester.observe(object(), pts=4, now_s=2.5) is True
    tester.configure(enabled=False)
    assert tester.observe(object(), pts=5, now_s=3.0) is False
    assert calls == 2


def test_preview_encoding_cannot_starve_the_sidecar_status_event_loop() -> None:
    class Frame:
        def to_ndarray(self, *, format: str):
            assert format == "bgr24"
            return object()

    class Model:
        def predict(self, **_options):
            time.sleep(0.02)
            return [SimpleNamespace(boxes=[])]

    async def scenario() -> None:
        runtime = perception_sidecar.PerceptionRuntime()
        runtime._model = Model()
        runtime._fruit_class_ids = {"pear": 0}

        def slow_preview(*_args, **_options) -> None:
            time.sleep(0.15)

        runtime._publish_preview = slow_preview
        runtime._frames.put_nowait((Frame(), time.monotonic(), 1, "1/90000"))

        loop = asyncio.get_running_loop()
        status_tick = asyncio.Event()
        started = loop.time()
        loop.call_later(0.04, status_tick.set)
        runtime._detector_task = asyncio.create_task(runtime._detect())
        try:
            await asyncio.wait_for(status_tick.wait(), timeout=0.50)
            status_elapsed_s = loop.time() - started
        finally:
            await runtime.close()

        assert status_elapsed_s < 0.09

    asyncio.run(scenario())


def test_small_uncertain_pear_is_confirmed_with_one_bounded_crop_pass() -> None:
    full_frame = FakeImage(720, 1280)

    class Model:
        def __init__(self) -> None:
            self.sources = []

        def predict(self, **options):
            source = options["source"]
            self.sources.append(source)
            if len(self.sources) == 1:
                return [
                    SimpleNamespace(boxes=FakeBoxes([0.60], [[610, 480, 640, 520]]))
                ]
            return [SimpleNamespace(boxes=FakeBoxes([0.84], [[113, 108, 143, 148]]))]

    runtime = perception_sidecar.PerceptionRuntime()
    model = Model()
    runtime._model = model
    runtime._fruit_class_ids = {"pear": 0}
    previews = []
    runtime._publish_preview = lambda *_args, **options: previews.append(options)
    received = time.monotonic()
    runtime.evidence.note_source(
        pts=100,
        time_base="1/90000",
        received_monotonic_s=received,
        width=1280,
        height=720,
    )

    runtime._process_frame(
        ArrayFrame(full_frame),
        received,
        100,
        "1/90000",
    )

    detection = runtime.evidence.status()["detection"]
    assert len(model.sources) == 2
    assert model.sources[1].shape == (256, 256, 3)
    assert detection["confidence"] == 0.84
    assert detection["bbox_xyxy"] == [610, 480, 640, 520]
    assert detection["inference_passes"] == 2
    assert detection["crop_confirmation"] == {
        "attempted": True,
        "promoted": True,
        "full_frame_confidence": 0.60,
        "crop_confidence": 0.84,
        "crop_xyxy": [497, 372, 753, 628],
        "agreement_iou": 1.0,
    }
    assert "CROP CONFIRMED" in previews[-1]["message"]
    runtime._inference_executor.shutdown(wait=True, cancel_futures=True)


def test_large_confident_pear_does_not_spend_a_second_inference_pass() -> None:
    full_frame = FakeImage(720, 1280)

    class Model:
        def __init__(self) -> None:
            self.calls = 0

        def predict(self, **_options):
            self.calls += 1
            return [SimpleNamespace(boxes=FakeBoxes([0.82], [[400, 300, 800, 700]]))]

    runtime = perception_sidecar.PerceptionRuntime()
    model = Model()
    runtime._model = model
    runtime._fruit_class_ids = {"pear": 0}
    runtime._publish_preview = lambda *_args, **_options: None

    runtime._process_frame(
        ArrayFrame(full_frame),
        time.monotonic(),
        100,
        "1/90000",
    )

    detection = runtime.evidence.status()["detection"]
    assert model.calls == 1
    assert detection["confidence"] == 0.82
    assert detection["inference_passes"] == 1
    assert detection["crop_confirmation"]["attempted"] is False
    assert runtime.status()["crop_confirm"] == {
        "enabled": True,
        "minimum_candidate_confidence": 0.35,
        "uncertain_below_confidence": 0.65,
        "small_bbox_area_ratio": 0.005,
        "minimum_crop_side_px": 256,
        "context_scale": 6.0,
        "minimum_confirmation_confidence": 0.55,
        "minimum_agreement_iou": 0.10,
    }
    runtime._inference_executor.shutdown(wait=True, cancel_futures=True)


def test_crop_result_must_overlap_and_improve_the_full_frame_candidate() -> None:
    full_frame = FakeImage(720, 1280)

    class Model:
        def __init__(self) -> None:
            self.calls = 0

        def predict(self, **_options):
            self.calls += 1
            if self.calls == 1:
                return [
                    SimpleNamespace(boxes=FakeBoxes([0.60], [[610, 480, 640, 520]]))
                ]
            return [SimpleNamespace(boxes=FakeBoxes([0.95], [[0, 0, 20, 20]]))]

    runtime = perception_sidecar.PerceptionRuntime()
    model = Model()
    runtime._model = model
    runtime._fruit_class_ids = {"pear": 0}
    runtime._publish_preview = lambda *_args, **_options: None

    runtime._process_frame(
        ArrayFrame(full_frame),
        time.monotonic(),
        100,
        "1/90000",
    )

    detection = runtime.evidence.status()["detection"]
    assert model.calls == 2
    assert detection["confidence"] == 0.60
    assert detection["bbox_xyxy"] == [610, 480, 640, 520]
    assert detection["crop_confirmation"]["promoted"] is False
    assert detection["crop_confirmation"]["crop_confidence"] == 0.95
    assert detection["crop_confirmation"]["agreement_iou"] == 0.0
    runtime._inference_executor.shutdown(wait=True, cancel_futures=True)


def test_tiny_weak_proposal_does_not_trigger_crop_confirmation() -> None:
    full_frame = FakeImage(720, 1280)

    class Model:
        def __init__(self) -> None:
            self.calls = 0

        def predict(self, **_options):
            self.calls += 1
            return [SimpleNamespace(boxes=FakeBoxes([0.20], [[610, 480, 640, 520]]))]

    runtime = perception_sidecar.PerceptionRuntime()
    model = Model()
    runtime._model = model
    runtime._fruit_class_ids = {"pear": 0}
    runtime._publish_preview = lambda *_args, **_options: None

    runtime._process_frame(
        ArrayFrame(full_frame),
        time.monotonic(),
        100,
        "1/90000",
    )

    detection = runtime.evidence.status()["detection"]
    assert model.calls == 1
    assert detection["crop_confirmation"]["attempted"] is False
    runtime._inference_executor.shutdown(wait=True, cancel_futures=True)


def test_camera_only_fruit_without_full_frame_proposal_gets_bounded_search_crop() -> (
    None
):
    full_frame = FakeImage(720, 1280)

    class Model:
        def __init__(self) -> None:
            self.sources = []

        def predict(self, **options):
            self.sources.append(options["source"])
            if len(self.sources) == 1:
                return [SimpleNamespace(boxes=FakeBoxes([], []))]
            return [SimpleNamespace(boxes=FakeBoxes([0.58], [[180, 280, 200, 300]]))]

    runtime = perception_sidecar.PerceptionRuntime()
    model = Model()
    runtime._model = model
    runtime._fruit_class_ids = {"apple": 0}
    runtime.select_target("apple")
    previews = []
    runtime._publish_preview = lambda *_args, **options: previews.append(options)

    runtime._process_frame(
        ArrayFrame(full_frame),
        time.monotonic(),
        100,
        "1/90000",
    )

    detection = runtime.evidence.status()["detection"]
    assert len(model.sources) == 2
    assert model.sources[1].shape == (512, 512, 3)
    assert detection["confidence"] == 0.58
    assert detection["bbox_xyxy"] == [564, 488, 584, 508]
    assert detection["inference_passes"] == 2
    assert detection["search_crop"] == {
        "attempted": True,
        "crop_xyxy": [384, 208, 896, 720],
    }
    assert "SEARCH CROP" in previews[-1]["message"]
    runtime._inference_executor.shutdown(wait=True, cancel_futures=True)


def test_banana_specialist_confirmation_reaches_existing_evidence_pipeline() -> None:
    full_frame = FakeImage(720, 1280)

    class Model:
        def __init__(self, confidence, bbox) -> None:
            self.confidence = confidence
            self.bbox = bbox
            self.calls = 0

        def predict(self, **_options):
            self.calls += 1
            return [
                SimpleNamespace(
                    boxes=FakeBoxes([self.confidence], [self.bbox])
                )
            ]

    runtime = perception_sidecar.PerceptionRuntime()
    general = Model(0.42, [550, 575, 675, 620])
    specialist = Model(0.79, [555, 578, 680, 623])
    runtime._model = general
    runtime._fruit_class_ids = {"apple": 1, "banana": 2, "pear": 3}
    runtime._model_router = FruitModelRouter(
        general_model=general,
        general_class_ids=runtime._fruit_class_ids,
        banana_specialist_model=specialist,
        banana_specialist_class_id=0,
    )
    runtime.select_target("banana")
    runtime._publish_preview = lambda *_args, **_options: None

    runtime._process_frame(
        ArrayFrame(full_frame),
        time.monotonic(),
        100,
        "1/90000",
    )

    detection = runtime.evidence.status()["detection"]
    assert general.calls == specialist.calls == 1
    assert detection["label"] == "banana"
    assert detection["confidence"] == 0.79
    assert detection["inference_passes"] == 2
    assert detection["model_route"]["full_frame"]["mode"] == "banana_specialist"
    assert detection["model_route"]["full_frame"]["confirmed"] is True
    assert detection["crop_confirmation"]["attempted"] is False
    runtime._inference_executor.shutdown(wait=True, cancel_futures=True)


def test_rejected_banana_specialist_proposal_continues_search_without_more_passes() -> (
    None
):
    full_frame = FakeImage(720, 1280)

    class Model:
        def __init__(self, bbox) -> None:
            self.bbox = bbox
            self.calls = 0

        def predict(self, **_options):
            self.calls += 1
            return [SimpleNamespace(boxes=FakeBoxes([0.90], [self.bbox]))]

    runtime = perception_sidecar.PerceptionRuntime()
    general = Model([100, 200, 180, 300])
    specialist = Model([800, 100, 900, 200])
    runtime._model = general
    runtime._fruit_class_ids = {"apple": 1, "banana": 2, "pear": 3}
    runtime._model_router = FruitModelRouter(
        general_model=general,
        general_class_ids=runtime._fruit_class_ids,
        banana_specialist_model=specialist,
        banana_specialist_class_id=0,
    )
    runtime.select_target("banana")
    previews = []
    runtime._publish_preview = lambda *_args, **options: previews.append(options)

    runtime._process_frame(
        ArrayFrame(full_frame),
        time.monotonic(),
        100,
        "1/90000",
    )

    assert general.calls == specialist.calls == 1
    assert runtime.evidence.status()["detection"] == {}
    assert previews[-1]["message"] == "SEARCHING FOR BANANA"
    runtime._inference_executor.shutdown(wait=True, cancel_futures=True)


def test_media_logging_suppresses_repetitive_h264_decoder_warnings() -> None:
    logger = logging.getLogger("aiortc.codecs.h264")
    original_level = logger.level
    original_handlers = list(logger.handlers)
    original_propagate = logger.propagate
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    logger.handlers = [handler]
    logger.propagate = False
    logger.setLevel(logging.WARNING)

    try:
        perception_sidecar.configure_media_logging()
        for _ in range(50_000):
            logger.warning(
                "H264Decoder() failed to decode, skipping package: invalid data"
            )
    finally:
        logger.handlers = original_handlers
        logger.propagate = original_propagate
        logger.setLevel(original_level)

    assert len(stream.getvalue().encode()) < 4 * 1024 * 1024


def test_sidecar_status_binds_pear_geometry_to_advancing_source_markers() -> None:
    evidence = PerceptionEvidence(generation="camera-1")

    for pts in range(100, 110):
        evidence.note_source(
            pts=pts,
            time_base="1/90000",
            received_monotonic_s=10.0 + pts / 1000,
            width=1280,
            height=720,
        )
    for pts in range(105, 110):
        evidence.note_detection(
            pts=pts,
            label="pear",
            confidence=0.81,
            bbox_xyxy=(480, 360, 800, 700),
            inference_s=0.08,
            completed_monotonic_s=10.0 + pts / 1000 + 0.08,
        )

    status = evidence.status()

    assert status["generation"] == "camera-1"
    assert status["source"]["pts"] == 109
    assert status["source"]["consecutive_frames"] == 10
    assert status["detection"]["consecutive_detections"] == 5
    assert status["detection"]["bbox_xyxy"] == [480, 360, 800, 700]
    assert status["error"] is None


def test_switching_supported_fruit_clears_old_detection_stability() -> None:
    evidence = PerceptionEvidence(generation="camera-1")
    evidence.note_detection(
        pts=100,
        label="pear",
        confidence=0.81,
        bbox_xyxy=(480, 360, 800, 700),
        inference_s=0.08,
        completed_monotonic_s=10.08,
    )

    evidence.select_target("apple")
    evidence.note_detection(
        pts=101,
        label="pear",
        confidence=0.99,
        bbox_xyxy=(480, 360, 800, 700),
        inference_s=0.08,
        completed_monotonic_s=10.18,
    )
    evidence.note_detection(
        pts=102,
        label="apple",
        confidence=0.80,
        bbox_xyxy=(480, 360, 800, 700),
        inference_s=0.08,
        completed_monotonic_s=10.28,
    )
    evidence.note_miss("pear")

    status = evidence.status()
    assert status["target_fruit"] == "apple"
    assert status["detection"]["label"] == "apple"
    assert status["detection"]["consecutive_detections"] == 1


def test_apple_sidecar_stability_is_raw_and_ignores_motion_policy_floor(
    monkeypatch,
) -> None:
    monkeypatch.setenv("BORDER_COLLIE_APPLE_ACQUISITION_CONFIDENCE", "0.70")
    evidence = PerceptionEvidence(generation="camera-1", target_fruit="apple")

    evidence.note_detection(
        pts=100,
        label="apple",
        confidence=0.40,
        bbox_xyxy=(480, 360, 800, 700),
        inference_s=0.08,
        completed_monotonic_s=10.08,
    )

    assert evidence.status()["detection"]["consecutive_detections"] == 1


def test_in_flight_old_target_result_cannot_kill_the_preview_worker() -> None:
    full_frame = FakeImage(720, 1280)

    class Model:
        def __init__(self, runtime: perception_sidecar.PerceptionRuntime) -> None:
            self.runtime = runtime
            self.calls = 0

        def predict(self, **_options):
            self.calls += 1
            if self.calls == 1:
                self.runtime.select_target("apple")
            return [SimpleNamespace(boxes=FakeBoxes([0.82], [[400, 300, 800, 700]]))]

    runtime = perception_sidecar.PerceptionRuntime()
    runtime._fruit_class_ids = {"apple": 1, "pear": 0}
    runtime._model = Model(runtime)
    previews = []
    runtime._publish_preview = lambda *_args, **options: previews.append(options)

    runtime._process_frame(
        ArrayFrame(full_frame),
        time.monotonic(),
        100,
        "1/90000",
    )
    runtime._process_frame(
        ArrayFrame(full_frame),
        time.monotonic(),
        101,
        "1/90000",
    )

    detection = runtime.evidence.status()["detection"]
    assert detection["label"] == "apple"
    assert len(previews) == 1
    assert "APPLE" in previews[0]["message"]
    runtime._inference_executor.shutdown(wait=True, cancel_futures=True)


def test_media_http_boundary_exposes_readiness_and_bark() -> None:
    class Runtime:
        def __init__(self) -> None:
            self.events = []

        async def start(self) -> None:
            self.events.append("start")

        async def close(self) -> None:
            self.events.append("close")

        def status(self):
            return {"generation": "camera-1", "bark_ready": True}

        async def bark(self):
            self.events.append("bark")
            return {"ok": True, "uuid": "bark-1"}

    runtime = Runtime()
    with TestClient(create_app(runtime)) as client:
        assert client.get("/status").json()["bark_ready"] is True
        assert client.post("/api/bark").json() == {"ok": True, "uuid": "bark-1"}

    assert runtime.events == ["start", "bark", "close"]


def test_media_http_boundary_selects_a_supported_camera_only_target() -> None:
    class Runtime:
        def __init__(self) -> None:
            self.target_fruit = "pear"

        async def start(self) -> None:
            pass

        async def close(self) -> None:
            pass

        def select_target(self, target_fruit: str) -> dict[str, object]:
            self.target_fruit = target_fruit
            return {
                "target_fruit": target_fruit,
                "supported_fruits": ["apple", "banana", "pear"],
            }

    runtime = Runtime()
    with TestClient(create_app(runtime)) as client:
        response = client.post(
            "/api/target",
            json={"target_fruit": "banana"},
        )

    assert response.status_code == 200
    assert response.json()["target_fruit"] == "banana"
    assert runtime.target_fruit == "banana"


def test_media_http_boundary_configures_only_the_read_only_coco_tester() -> None:
    class Runtime:
        def __init__(self) -> None:
            self.configuration = None

        async def start(self) -> None:
            pass

        async def close(self) -> None:
            pass

        def status(self):
            return {
                "coco_test": {
                    "enabled": False,
                    "strictly_read_only": True,
                    "class_count": 80,
                }
            }

        async def configure_coco_test(self, **configuration):
            self.configuration = configuration
            return {
                **configuration,
                "strictly_read_only": True,
                "class_count": 80,
            }

    runtime = Runtime()
    with TestClient(create_app(runtime)) as client:
        status = client.get("/api/coco-test")
        changed = client.post(
            "/api/coco-test",
            json={"enabled": True, "minimum_confidence": 0.05, "reset": True},
        )

    assert status.json()["strictly_read_only"] is True
    assert changed.json()["class_count"] == 80
    assert runtime.configuration == {
        "enabled": True,
        "minimum_confidence": 0.05,
        "reset": True,
    }


def test_media_http_boundary_exposes_the_latest_annotated_camera_frame() -> None:
    jpeg = b"\xff\xd8annotated-pear-frame\xff\xd9"

    class Runtime:
        async def start(self) -> None:
            pass

        async def close(self) -> None:
            pass

        def camera_frame(self) -> bytes:
            return jpeg

    with TestClient(create_app(Runtime())) as client:
        response = client.get("/api/camera/frame.jpg")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.headers["cache-control"] == "no-store"
    assert response.content == jpeg


def test_media_http_boundary_exposes_fieldmark_raw_frame_and_evidence_archive() -> None:
    raw_jpeg = b"\xff\xd8raw-fieldmark-frame\xff\xd9"
    archive = b"PK\x03\x04fieldmark-evidence"

    class Runtime:
        async def start(self) -> None:
            pass

        async def close(self) -> None:
            pass

        def raw_camera_frame(self) -> bytes:
            return raw_jpeg

        def evidence_archive(self) -> bytes:
            return archive

    with TestClient(create_app(Runtime())) as client:
        raw = client.get("/api/camera/raw.jpg")
        evidence = client.get("/api/evidence/clip.zip")

    assert raw.status_code == 200
    assert raw.headers["content-type"] == "image/jpeg"
    assert raw.headers["cache-control"] == "no-store"
    assert raw.content == raw_jpeg
    assert evidence.status_code == 200
    assert evidence.headers["content-type"] == "application/zip"
    assert evidence.headers["content-disposition"] == (
        'attachment; filename="border-collie-evidence.zip"'
    )
    assert evidence.content == archive


def test_evidence_archive_is_a_bounded_fieldmark_ready_raw_frame_sequence() -> None:
    buffer = EvidenceFrameBuffer(generation="camera-1", maximum_frames=2)
    for sequence, confidence, bbox in (
        (1, 0.01, (10, 20, 20, 40)),
        (2, 0.02, (30, 40, 50, 80)),
        (3, 0.03, (60, 80, 100, 160)),
    ):
        buffer.record(
            raw_jpeg=f"raw-{sequence}".encode(),
            annotated_jpeg=f"annotated-{sequence}".encode(),
            pts=sequence * 100,
            time_base="1/90000",
            received_monotonic_s=10.0 + sequence,
            width=200,
            height=200,
            detection={
                "label": "pear",
                "confidence": confidence,
                "bbox_xyxy": list(bbox),
            },
        )

    assert buffer.raw_camera_frame() == b"raw-3"

    with zipfile.ZipFile(io.BytesIO(buffer.archive())) as archive:
        assert archive.namelist() == [
            "manifest.json",
            "frames/000001.jpg",
            "frames/000002.jpg",
            "terminal/annotated.jpg",
        ]
        manifest = json.loads(archive.read("manifest.json"))
        assert archive.read("frames/000001.jpg") == b"raw-2"
        assert archive.read("frames/000002.jpg") == b"raw-3"
        assert archive.read("terminal/annotated.jpg") == b"annotated-3"

    assert manifest["schema_version"] == 1
    assert manifest["generation"] == "camera-1"
    assert manifest["format"] == "fieldmark-image-sequence"
    assert manifest["summary"] == {
        "sample_count": 2,
        "pear_candidate_samples": 2,
        "maximum_confidence": 0.03,
        "maximum_bbox_area_ratio": 0.08,
        "closest_detection": {
            "source_pts": 300,
            "confidence": 0.03,
            "bbox_xyxy": [60, 80, 100, 160],
            "bbox_area_ratio": 0.08,
        },
    }


def test_source_gap_resets_camera_and_detection_stability() -> None:
    evidence = PerceptionEvidence(generation="camera-1")
    for pts in range(5):
        evidence.note_source(
            pts=pts,
            time_base="1/90000",
            received_monotonic_s=1.0 + pts * 0.05,
            width=1280,
            height=720,
        )
    evidence.note_detection(
        pts=4,
        label="pear",
        confidence=0.81,
        bbox_xyxy=(1, 1, 2, 2),
        inference_s=0.08,
        completed_monotonic_s=1.28,
    )

    evidence.note_source(
        pts=5,
        time_base="1/90000",
        received_monotonic_s=2.0,
        width=1280,
        height=720,
    )

    status = evidence.status()
    assert status["source"]["consecutive_frames"] == 1
    assert status["detection"] == {}


def test_sidecar_publishes_sub_floor_detections_by_contract() -> None:
    """Weak detections are published raw; gating belongs to consumers.

    Deliberate pin for the contract documented in
    docs/camera-perception-contract.md: the close-range continuation path
    consumes detections down to per-fruit floors as low as 0.10, and
    telemetry diagnosability depends on sub-floor samples being visible. A
    helpful-looking publish floor added here would silently break both.
    """
    evidence = PerceptionEvidence(generation="camera-1")
    evidence.note_source(
        pts=100,
        time_base="1/90000",
        received_monotonic_s=10.0,
        width=1280,
        height=720,
    )
    evidence.note_detection(
        pts=100,
        label="pear",
        confidence=0.01,
        bbox_xyxy=(480, 360, 800, 700),
        inference_s=0.08,
        completed_monotonic_s=10.1,
    )

    detection = evidence.status()["detection"]

    assert detection["label"] == "pear"
    assert detection["confidence"] == 0.01
