import asyncio
import io
import json
import logging
import time
import zipfile
from types import SimpleNamespace
from typing import ClassVar

from fastapi.testclient import TestClient

from media import perception_sidecar
from media.model_router import FruitModelRouter
from media.perception_pipeline import FrameRoute
from media.perception_sidecar import EvidenceFrameBuffer, PerceptionEvidence, create_app
from media.service_supervision import ServiceSupervisionConfig, ServiceSupervisor


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


def test_runtime_supervisor_cleans_failed_sessions_and_recovers_in_process() -> None:
    class Video:
        def __init__(self) -> None:
            self.callback = None
            self.channels = []

        def add_track_callback(self, callback) -> None:
            self.callback = callback

        def switchVideoChannel(self, enabled: bool) -> None:
            self.channels.append(enabled)

    class Connection:
        instances: ClassVar[list[object]] = []

        def __init__(self, method, *, ip: str) -> None:
            self.method = method
            self.ip = ip
            self.video = Video()
            self.disconnected = False
            self.index = len(self.instances)
            self.instances.append(self)

        async def connect(self) -> None:
            if self.index < 2:
                raise TimeoutError("DataChannelTimeoutError")

        async def disconnect(self) -> None:
            self.disconnected = True

    async def scenario() -> None:
        runtime = perception_sidecar.PerceptionRuntime()
        runtime._connection_class = Connection
        runtime._connection_method = "local-sta"
        runtime._audiohub_class = lambda connection: ("audiohub", connection)
        runtime._connect_timeout_s = 0.1
        runtime._cleanup_timeout_s = 0.1
        runtime._monitor_interval_s = 0.001
        runtime._supervision = ServiceSupervisor(
            ServiceSupervisionConfig(
                stable_frame_count=2,
                frame_stall_timeout_s=0.25,
                restart_budget=3,
                initial_backoff_s=0.001,
                maximum_backoff_s=0.002,
            )
        )
        runtime._supervisor_task = asyncio.create_task(runtime._supervise_sessions())
        try:
            deadline = asyncio.get_running_loop().time() + 1.0
            while len(Connection.instances) < 3:
                if asyncio.get_running_loop().time() >= deadline:
                    raise AssertionError("third supervised session did not start")
                await asyncio.sleep(0.001)
            generation = runtime.evidence.generation
            runtime._supervision.note_frame(generation, 1)
            runtime._supervision.note_frame(generation, 2)
            assert runtime.status()["supervision"]["state"] == "ready"
            assert runtime.status()["bark_ready"] is True
        finally:
            await runtime.close()

        assert len(Connection.instances) == 3
        assert all(connection.disconnected for connection in Connection.instances)
        assert Connection.instances[2].video.channels == [True, False]

    asyncio.run(scenario())


def test_runtime_first_frame_grace_starts_after_connection_is_open() -> None:
    async def scenario() -> None:
        runtime = perception_sidecar.PerceptionRuntime()
        runtime._monitor_interval_s = 0.001
        runtime._supervision = ServiceSupervisor(
            ServiceSupervisionConfig(
                stable_frame_count=2,
                frame_stall_timeout_s=0.01,
                first_frame_timeout_s=0.05,
                restart_budget=1,
                initial_backoff_s=0.001,
                maximum_backoff_s=0.001,
            )
        )
        publisher_tasks: list[asyncio.Task[None]] = []

        async def delayed_open(generation: str) -> None:
            # The real connection handshake took almost the entire 0.75 second
            # stall window before video activation. First-frame health must not
            # include that handshake time.
            await asyncio.sleep(0.03)

            async def publish_frames() -> None:
                await asyncio.sleep(0.005)
                runtime._supervision.note_frame(generation, 1)
                await asyncio.sleep(0.005)
                runtime._supervision.note_frame(generation, 2)

            publisher_tasks.append(asyncio.create_task(publish_frames()))

        runtime._open_session = delayed_open
        runtime._cleanup_session = lambda: asyncio.sleep(0)
        runtime._supervisor_task = asyncio.create_task(runtime._supervise_sessions())
        try:
            deadline = asyncio.get_running_loop().time() + 0.25
            while not runtime._supervision.ready:
                if asyncio.get_running_loop().time() >= deadline:
                    raise AssertionError(
                        f"supervisor did not become ready: {runtime._supervision.status()}"
                    )
                await asyncio.sleep(0.001)
            status = runtime._supervision.status()
            assert status["attempts"] == 1
            assert status["total_restarts"] == 0
        finally:
            await runtime.close()
            await asyncio.gather(*publisher_tasks, return_exceptions=True)

    asyncio.run(scenario())


def test_media_status_publishes_its_release_cohort(monkeypatch) -> None:
    monkeypatch.setenv("BORDER_COLLIE_RELEASE_ID", "release-9")
    monkeypatch.setenv("BORDER_COLLIE_CONFIG_SCHEMA", "4")

    status = perception_sidecar.release_cohort_status()

    assert status == {
        "release_id": "release-9",
        "config_schema": 4,
        "service": "media",
    }


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
        runtime._frames.put_nowait(
            (runtime.evidence.generation, Frame(), time.monotonic(), 1, "1/90000")
        )

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
            source = options["source"]
            self.sources.append(source)
            if source.shape == (720, 1280, 3):
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
    assert model.sources[0].shape == (720, 1280, 3)
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


def test_throughput_routes_search_crop_on_the_next_advancing_frame() -> None:
    full_frame = FakeImage(720, 1280)

    class Model:
        def __init__(self) -> None:
            self.sources = []

        def predict(self, **options):
            source = options["source"]
            self.sources.append(source)
            if source.shape == (720, 1280, 3):
                return [SimpleNamespace(boxes=FakeBoxes([], []))]
            return [SimpleNamespace(boxes=FakeBoxes([0.58], [[180, 280, 200, 300]]))]

    runtime = perception_sidecar.PerceptionRuntime()
    model = Model()
    runtime._model = model
    runtime._fruit_class_ids = {"apple": 0}
    runtime.select_target("apple")
    runtime._publish_preview = lambda *_args, **_options: None

    first_received = time.monotonic()
    runtime.evidence.note_source(
        pts=100,
        time_base="1/90000",
        received_monotonic_s=first_received,
        width=1280,
        height=720,
    )
    runtime._process_frame(
        ArrayFrame(full_frame),
        first_received,
        100,
        "1/90000",
        route=FrameRoute.FULL_FRAME,
    )

    assert len(model.sources) == 1
    assert runtime.evidence.status()["detection"] == {}

    second_received = first_received + 0.07
    runtime.evidence.note_source(
        pts=101,
        time_base="1/90000",
        received_monotonic_s=second_received,
        width=1280,
        height=720,
    )
    runtime._process_frame(
        ArrayFrame(full_frame),
        second_received,
        101,
        "1/90000",
        route=FrameRoute.LOWER_CENTER_SEARCH_CROP,
    )

    detection = runtime.evidence.status()["detection"]
    assert len(model.sources) == 3
    assert model.sources[1].shape == (720, 1280, 3)
    assert model.sources[2].shape == (512, 512, 3)
    assert detection["source_pts"] == 101
    assert detection["confidence"] == 0.58
    assert detection["bbox_xyxy"] == [564, 488, 584, 508]
    assert detection["inference_passes"] == 2
    assert detection["frame_route"] == "lower_center_search_crop"
    assert detection["observations"]["full_frame"] is None
    assert detection["observations"]["crop"]["route"] == "search_crop"
    runtime._inference_executor.shutdown(wait=True, cancel_futures=True)


def test_search_crop_route_keeps_full_frame_identity_authoritative() -> None:
    full_frame = FakeImage(720, 1280)

    class Model:
        def __init__(self) -> None:
            self.sources = []

        def predict(self, **options):
            source = options["source"]
            self.sources.append(source)
            if source.shape == (720, 1280, 3):
                return [
                    SimpleNamespace(
                        boxes=FakeBoxes([0.61], [[500, 430, 620, 610]])
                    )
                ]
            return [SimpleNamespace(boxes=FakeBoxes([], []))]

    runtime = perception_sidecar.PerceptionRuntime()
    model = Model()
    runtime._model = model
    runtime._fruit_class_ids = {"apple": 0}
    runtime.select_target("apple")
    runtime._publish_preview = lambda *_args, **_options: None
    received = time.monotonic()
    runtime.evidence.note_source(
        pts=101,
        time_base="1/90000",
        received_monotonic_s=received,
        width=1280,
        height=720,
    )

    runtime._process_frame(
        ArrayFrame(full_frame),
        received,
        101,
        "1/90000",
        route=FrameRoute.LOWER_CENTER_SEARCH_CROP,
    )

    detection = runtime.evidence.status()["detection"]
    assert [source.shape for source in model.sources] == [
        (720, 1280, 3),
        (512, 512, 3),
    ]
    assert detection["observations"]["full_frame"] == {
        "label": "apple",
        "confidence": 0.61,
        "bbox_xyxy": [500, 430, 620, 610],
        "route": "full_frame",
    }
    assert detection["observations"]["crop"] is None
    assert detection["inference_passes"] == 2
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


def test_apple_detection_stability_starts_at_fifty_percent_confidence() -> None:
    evidence = PerceptionEvidence(generation="camera-1")
    evidence.select_target("apple")

    detection = evidence.note_detection(
        pts=100,
        label="apple",
        confidence=0.50,
        bbox_xyxy=(480, 360, 800, 700),
        inference_s=0.08,
        completed_monotonic_s=10.08,
    )

    assert detection is not None
    assert detection["consecutive_detections"] == 1


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
