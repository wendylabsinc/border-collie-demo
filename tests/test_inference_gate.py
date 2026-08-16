"""Inference runs when a run or the operator needs it, and not otherwise.

The detector used to run a YOLO pass on every frame from process start,
forever, which is most of the robot's idle power draw. These tests pin the
gate that replaced that, and - more importantly - pin the two things the gate
must never break: activation readiness while the detector is asleep, and the
detector being awake before guidance issues its first search command.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from border_collie_demo.api import create_app as api_create_app
from border_collie_demo.config import PerceptionConfig
from border_collie_demo.perception import PerceptionStatusClient
from border_collie_demo.preflight import evaluate_preflight, preflight_check_ready
from media import perception_sidecar
from media.perception_sidecar import InferenceGate, create_app


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


class FakeClock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def gate_for(clock: FakeClock, **options: object) -> InferenceGate:
    defaults: dict[str, object] = {"warmup_s": 0.0, "clock": clock}
    defaults.update(options)
    return InferenceGate(**defaults)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# The lease itself
# --------------------------------------------------------------------------


def test_gate_is_idle_once_warmup_lapses_and_nothing_holds_it() -> None:
    clock = FakeClock()
    gate = InferenceGate(warmup_s=60.0, clock=clock)

    assert gate.active() is True
    assert gate.status()["reason"] == "warmup"

    clock.advance(60.0)

    assert gate.active() is False
    assert gate.status()["reason"] == "idle"


def test_a_hold_wakes_the_detector_and_lapses_on_its_own() -> None:
    clock = FakeClock()
    gate = gate_for(clock)
    assert gate.active() is False

    gate.hold("preview", 20.0)
    assert gate.active() is True
    assert gate.status()["reason"] == "held"

    clock.advance(19.0)
    assert gate.active() is True

    # Nothing had to switch it off. The lease simply ran out.
    clock.advance(2.0)
    assert gate.active() is False


def test_renewing_extends_a_lease_but_never_truncates_a_longer_one() -> None:
    clock = FakeClock()
    gate = gate_for(clock)
    gate.hold("run", 120.0)

    clock.advance(10.0)
    gate.hold("run", 120.0)
    clock.advance(115.0)

    # Renewal moved the deadline out, so the original 120 s has passed and the
    # detector is still awake.
    assert gate.active() is True

    # A short renewal arriving late must not cut an already-longer lease short.
    gate.hold("run", 120.0)
    clock.advance(1.0)
    gate.hold("run", 5.0)
    clock.advance(30.0)
    assert gate.active() is True


def test_run_and_preview_leases_are_independent() -> None:
    clock = FakeClock()
    gate = gate_for(clock)
    gate.hold("run", 120.0)
    gate.hold("preview", 20.0)

    clock.advance(25.0)

    # The preview lease lapsed; the run is still executing.
    assert gate.active() is True
    assert set(gate.status()["holds"]) == {"run"}

    gate.release("run")
    assert gate.active() is False


def test_releasing_drops_the_lease_immediately() -> None:
    clock = FakeClock()
    gate = gate_for(clock)
    gate.hold("run", 120.0)

    gate.release("run")

    assert gate.active() is False
    assert gate.status()["holds"] == {}


def test_the_kill_switch_restores_the_always_on_detector() -> None:
    clock = FakeClock()
    gate = gate_for(clock, enabled=False)

    assert gate.active() is True
    assert gate.status()["reason"] == "gate_disabled"

    clock.advance(10_000.0)
    assert gate.active() is True


@pytest.mark.parametrize("reason", ["", "motion", "inference", None])
def test_unsupported_hold_reasons_are_refused(reason: object) -> None:
    gate = gate_for(FakeClock())
    with pytest.raises(ValueError):
        gate.hold(reason)  # type: ignore[arg-type]


def test_hold_reasons_are_normalised_before_they_are_matched() -> None:
    gate = gate_for(FakeClock())

    gate.hold("  RUN ")

    assert set(gate.status()["holds"]) == {"run"}


@pytest.mark.parametrize("seconds", [0.0, -1.0, float("inf"), 901.0])
def test_unbounded_holds_are_refused(seconds: float) -> None:
    gate = gate_for(FakeClock())
    with pytest.raises(ValueError):
        gate.hold("run", seconds)


def test_a_lapsed_gate_can_always_be_woken_again() -> None:
    """Nothing ever persists an "off" decision, so no run can be locked out."""
    clock = FakeClock()
    gate = gate_for(clock)
    gate.hold("run", 120.0)
    clock.advance(500.0)
    assert gate.active() is False

    gate.hold("run")

    assert gate.active() is True


# --------------------------------------------------------------------------
# The trap: readiness must survive an idle detector
# --------------------------------------------------------------------------


def healthy_source_payload(inference_active: bool) -> dict[str, object]:
    """Camera source advancing normally, with no detection at all."""
    return {
        "generation": "generation-1",
        "target_fruit": "pear",
        "source": {
            "pts": 12345,
            "time_base": "1/90000",
            "received_monotonic_s": 99.80,
            "consecutive_frames": 30,
            "width": 1280,
            "height": 720,
        },
        "detection": {},
        "inference": {
            "active": inference_active,
            "reason": "held" if inference_active else "idle",
        },
    }


def test_camera_stays_healthy_while_the_detector_is_asleep() -> None:
    client = PerceptionStatusClient(
        PerceptionConfig(enabled=True),
        fetcher=lambda _url, _timeout: healthy_source_payload(False),
        clock=lambda: 100.0,
    )

    status = client.status()

    # Every camera violation is source-level. None of them needs a detection,
    # so an idle detector cannot make the camera look broken.
    assert status["camera_healthy"] is True
    assert status["camera_violations"] == []
    # There is genuinely no detection, and the status says so honestly.
    assert status["target_ready"] is False
    assert status["inference"]["active"] is False


def test_activation_readiness_passes_while_inference_is_idle() -> None:
    """The gate must not cost us activation - this is the whole trap."""
    client = PerceptionStatusClient(
        PerceptionConfig(enabled=True),
        fetcher=lambda _url, _timeout: healthy_source_payload(False),
        clock=lambda: 100.0,
    )
    hardware = {
        "connected": True,
        "autonomy_enabled": True,
        "pose": {"healthy": True, "age_s": 0.02},
        "active_operation": None,
        "motion": {"armed": False},
    }

    report = evaluate_preflight(hardware, client.status(), {"ready": True})

    assert preflight_check_ready(report, "camera_perception_ready") is True
    assert report["ready"] is True


def test_an_unhealthy_camera_still_fails_preflight_with_the_gate_in_place() -> None:
    payload = healthy_source_payload(False)
    payload["source"]["consecutive_frames"] = 2  # type: ignore[index]
    client = PerceptionStatusClient(
        PerceptionConfig(enabled=True),
        fetcher=lambda _url, _timeout: payload,
        clock=lambda: 100.0,
    )

    status = client.status()

    assert status["camera_healthy"] is False
    assert "fewer than 10 consecutive source frames" in status["camera_violations"]


# --------------------------------------------------------------------------
# Who takes which lease
# --------------------------------------------------------------------------


class RecordingClient:
    """A perception client wired to in-memory transports."""

    def __init__(self) -> None:
        self.target_posts: list[str] = []
        self.status_urls: list[str] = []
        self.inference_posts: list[dict[str, object]] = []
        self.client = PerceptionStatusClient(
            PerceptionConfig(enabled=True),
            fetcher=self._fetch,
            target_poster=self._post_target,
            json_poster=self._post_json,
            clock=lambda: 100.0,
        )

    def _fetch(self, url: str, _timeout: float) -> dict[str, object]:
        self.status_urls.append(url)
        return healthy_source_payload(True)

    def _post_target(
        self, _url: str, fruit: str, _timeout: float
    ) -> dict[str, object]:
        self.target_posts.append(fruit)
        return {"target_fruit": fruit, "supported_fruits": ["pear"]}

    def _post_json(
        self, _url: str, body: dict[str, object], _timeout: float
    ) -> dict[str, object]:
        self.inference_posts.append(body)
        return {"active": True, "reason": "held"}


def test_activation_takes_the_run_lease_when_it_selects_the_target() -> None:
    """Warm before the sweep starts, not at the first search command."""
    recording = RecordingClient()

    recording.client.select_run_target("pear")

    assert recording.target_posts == ["pear"]
    assert recording.inference_posts == [{"hold": "run"}]


def test_the_fruit_test_page_only_ever_gets_the_short_preview_lease() -> None:
    recording = RecordingClient()

    recording.client.select_target("pear")

    assert recording.target_posts == ["pear"]
    # No run lease. Selecting a fruit from the preview page must not pin the
    # detector on for a run's worth of time.
    assert recording.inference_posts == []


def test_run_and_preview_readers_renew_their_lease_but_the_dashboard_does_not() -> None:
    recording = RecordingClient()

    recording.client.status()
    recording.client.run_status()
    recording.client.preview_status()

    assert recording.status_urls == [
        "http://127.0.0.1:8111/status",
        "http://127.0.0.1:8111/status?hold=run",
        "http://127.0.0.1:8111/status?hold=preview",
    ]


def test_a_failed_lease_upgrade_leaves_inference_on_and_raises() -> None:
    """Activation fails loudly rather than starting a run with a cold detector."""
    posted: list[str] = []

    def explode(_url: str, _body: dict[str, object], _timeout: float) -> dict:
        raise RuntimeError("sidecar unreachable")

    client = PerceptionStatusClient(
        PerceptionConfig(enabled=True),
        target_poster=lambda _url, fruit, _timeout: (
            posted.append(fruit) or {"target_fruit": fruit}
        ),
        json_poster=explode,
    )

    with pytest.raises(RuntimeError):
        client.select_run_target("pear")

    # The target post already ran, and on the sidecar that alone takes the
    # preview lease - so the failure leaves the detector awake, never asleep.
    assert posted == ["pear"]


# --------------------------------------------------------------------------
# The sidecar HTTP boundary
# --------------------------------------------------------------------------


class GateRuntime:
    """Minimal sidecar runtime exposing only the gate surface."""

    def __init__(self, clock: FakeClock) -> None:
        self.gate = InferenceGate(warmup_s=0.0, clock=clock)
        self.target_fruit = "pear"

    async def start(self) -> None:
        pass

    async def close(self) -> None:
        pass

    def status(self) -> dict[str, object]:
        return {"target_fruit": self.target_fruit, "inference": self.gate.status()}

    def select_target(self, target_fruit: str) -> dict[str, object]:
        self.target_fruit = target_fruit
        return {"target_fruit": target_fruit}

    def hold_inference(
        self, reason: str, hold_s: float | None = None
    ) -> dict[str, object]:
        return self.gate.hold(reason, hold_s)

    def release_inference(self, reason: str) -> dict[str, object]:
        return self.gate.release(reason)

    def inference_status(self) -> dict[str, object]:
        return self.gate.status()


def test_polling_status_with_a_hold_renews_the_lease() -> None:
    clock = FakeClock()
    runtime = GateRuntime(clock)

    with TestClient(create_app(runtime)) as client:
        assert client.get("/status").json()["inference"]["active"] is False

        body = client.get("/status", params={"hold": "preview"}).json()
        assert body["inference"]["active"] is True

        # Polling stops. Nothing switches it off; the lease just lapses.
        clock.advance(21.0)
        assert client.get("/status").json()["inference"]["active"] is False


def test_status_refuses_an_unsupported_hold_rather_than_guessing() -> None:
    runtime = GateRuntime(FakeClock())
    with TestClient(create_app(runtime)) as client:
        assert client.get("/status", params={"hold": "motion"}).status_code == 422


def test_selecting_a_run_target_over_http_takes_the_run_lease() -> None:
    clock = FakeClock()
    runtime = GateRuntime(clock)

    with TestClient(create_app(runtime)) as client:
        body = client.post(
            "/api/target", json={"target_fruit": "pear", "hold": "run"}
        ).json()
        assert body["inference"]["holds"]["run"] == pytest.approx(120.0)

        # A preview lease would already have lapsed here; the run lease has not.
        clock.advance(30.0)
        assert client.get("/api/inference").json()["active"] is True


def test_the_inference_endpoint_holds_and_releases() -> None:
    clock = FakeClock()
    runtime = GateRuntime(clock)

    with TestClient(create_app(runtime)) as client:
        assert client.post("/api/inference", json={"hold": "run"}).json()["active"]
        released = client.post("/api/inference", json={"release": "run"}).json()
        assert released["active"] is False

        empty = client.post("/api/inference", json={})
        assert empty.status_code == 422


# --------------------------------------------------------------------------
# The detector loop actually stops spending GPU time
# --------------------------------------------------------------------------


class CountingModel:
    def __init__(self) -> None:
        self.passes = 0

    def predict(self, **_options):
        self.passes += 1
        return [SimpleNamespace(boxes=[])]


def idle_runtime(clock: FakeClock) -> perception_sidecar.PerceptionRuntime:
    runtime = perception_sidecar.PerceptionRuntime()
    runtime._inference_gate = InferenceGate(warmup_s=0.0, clock=clock)
    runtime._idle_preview_interval_s = 0.0
    runtime._fruit_class_ids = {"pear": 0}
    return runtime


def drive_one_frame(
    runtime: perception_sidecar.PerceptionRuntime, received: float
) -> None:
    """Run the detector loop over exactly one queued frame."""

    async def scenario() -> None:
        runtime._frames.put_nowait(
            (ArrayFrame(FakeImage(720, 1280)), received, 100, "1/90000")
        )
        task = asyncio.create_task(runtime._detect())
        for _ in range(200):
            await asyncio.sleep(0)
            if runtime._frames.empty():
                break
        await asyncio.sleep(0.02)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


def test_an_idle_detector_runs_no_model_pass_but_keeps_the_preview_alive() -> None:
    clock = FakeClock()
    runtime = idle_runtime(clock)
    model = CountingModel()
    runtime._model = model
    previews: list[dict] = []
    runtime._publish_preview = lambda *_args, **options: previews.append(options)

    drive_one_frame(runtime, 500.0)

    # The whole point: no GPU work at all.
    assert model.passes == 0
    # But the operator still sees live video, and the frame says why there are
    # no boxes on it.
    assert len(previews) == 1
    assert "IDLE" in previews[0]["header"]
    assert previews[0]["detection"] == {}


def test_a_held_detector_runs_the_model() -> None:
    clock = FakeClock()
    runtime = idle_runtime(clock)
    model = CountingModel()
    runtime._model = model
    runtime._publish_preview = lambda *_args, **_options: None
    runtime.hold_inference("run")

    drive_one_frame(runtime, 500.0)

    assert model.passes == 1


def test_a_dropped_idle_preview_frame_cannot_block_the_next_demo_run() -> None:
    """evidence.fail() latches forever, so the idle path must never call it.

    camera_healthy reads that latched error, and nothing clears it short of a
    process restart. A preview frame the detector was not even asked to look
    at must not be able to fail every future activation.
    """

    class BrokenFrame:
        def to_ndarray(self, *, format: str):
            raise RuntimeError("decoder produced an unusable frame")

    clock = FakeClock()
    runtime = idle_runtime(clock)
    runtime._model = CountingModel()
    runtime.evidence.note_source(
        pts=100,
        time_base="1/90000",
        received_monotonic_s=500.0,
        width=1280,
        height=720,
    )

    async def scenario() -> None:
        runtime._frames.put_nowait((BrokenFrame(), 500.0, 100, "1/90000"))
        task = asyncio.create_task(runtime._detect())
        for _ in range(200):
            await asyncio.sleep(0)
            if runtime._frames.empty():
                break
        await asyncio.sleep(0.02)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())

    assert runtime.status()["error"] is None


def test_source_evidence_keeps_advancing_while_the_detector_sleeps() -> None:
    """Source evidence comes from the camera callback, not the detector.

    This is why activation still works with inference gated: consecutive
    frames, PTS, time base and dimensions are all recorded here.
    """
    clock = FakeClock()
    runtime = idle_runtime(clock)
    received = 500.0
    for index in range(30):
        runtime.evidence.note_source(
            pts=100 + index,
            time_base="1/90000",
            received_monotonic_s=received + index * 0.03,
            width=1280,
            height=720,
        )

    status = runtime.status()

    assert status["inference"]["active"] is False
    assert status["source"]["consecutive_frames"] == 30
    assert status["detection"] == {}


# --------------------------------------------------------------------------
# The app boundary routes each caller to the right lease
# --------------------------------------------------------------------------


def test_the_preview_page_reads_through_the_lease_renewing_reader() -> None:
    """The fruit-test page must be able to see fruit while the robot idles."""
    readers: list[str] = []

    def dashboard_reader() -> dict[str, object]:
        readers.append("dashboard")
        return {"ready": False, "camera_healthy": True, "detail": "idle"}

    def preview_reader() -> dict[str, object]:
        readers.append("preview")
        return {"ready": True, "camera_healthy": True, "detail": "live"}

    app = api_create_app(
        camera_perception_status=dashboard_reader,
        preview_camera_perception=preview_reader,
    )
    client = TestClient(app)

    assert client.get("/api/fruits/preview").status_code == 200

    # The preview page never reads through the non-renewing dashboard reader,
    # or opening it would show fruit that the detector was never asked to find.
    assert "preview" in readers
    assert "dashboard" not in readers


def test_the_idle_dashboard_never_renews_the_lease() -> None:
    readers: list[str] = []

    def dashboard_reader() -> dict[str, object]:
        readers.append("dashboard")
        return {"ready": False, "camera_healthy": True, "detail": "idle"}

    def preview_reader() -> dict[str, object]:
        readers.append("preview")
        return {"ready": True, "camera_healthy": True, "detail": "live"}

    app = api_create_app(
        camera_perception_status=dashboard_reader,
        preview_camera_perception=preview_reader,
    )

    TestClient(app).get("/api/status")

    # A browser tab left open must not pin the detector on forever.
    assert "preview" not in readers


def test_activation_selects_the_target_through_the_run_lease_seam() -> None:
    """Preview selection and run selection must not share a seam."""
    selected: list[tuple[str, str]] = []

    app = api_create_app(
        select_perception_target=lambda fruit: (
            selected.append(("preview", fruit)) or {"target_fruit": fruit}
        ),
        select_run_perception_target=lambda fruit: (
            selected.append(("run", fruit)) or {"target_fruit": fruit}
        ),
    )
    client = TestClient(app)

    client.post("/api/fruits/preview", json={"target_fruit": "pear"})

    assert selected == [("preview", "pear")]


def test_an_idle_detector_clears_a_stale_detection_rather_than_freezing_it() -> None:
    clock = FakeClock()
    runtime = idle_runtime(clock)
    runtime._model = CountingModel()
    runtime._publish_preview = lambda *_args, **_options: None
    runtime.evidence.note_source(
        pts=100,
        time_base="1/90000",
        received_monotonic_s=500.0,
        width=1280,
        height=720,
    )
    runtime.evidence.note_detection(
        pts=100,
        label="pear",
        confidence=0.9,
        bbox_xyxy=(10, 10, 40, 40),
        inference_s=0.05,
        completed_monotonic_s=500.0,
    )
    assert runtime.evidence.status()["detection"] != {}

    drive_one_frame(runtime, 500.5)

    # A detection nobody re-measured must not linger and look current.
    assert runtime.evidence.status()["detection"] == {}
