import asyncio
import io
import json
import logging
import time
import zipfile
from types import SimpleNamespace

from fastapi.testclient import TestClient

from media import perception_sidecar
from media.perception_sidecar import EvidenceFrameBuffer, PerceptionEvidence, create_app


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
        runtime._pear_class_id = 0

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
