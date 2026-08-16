"""The physical Go2 Start button requests one Apple + Mango + Pear cohort."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

from border_collie_demo.api import (
    CONTROLLER_START_COHORT_FRUITS,
    CONTROLLER_START_HOME_ALIGN_YAW_RPS,
    CONTROLLER_START_SEARCH_YAW_RPS,
    CONTROLLER_START_TOLERATED_FAILURES,
    create_app,
    three_fruit_cohort_request,
)
from border_collie_demo.black_box import RunBlackBox
from border_collie_demo.cohorts import CohortConflict
from border_collie_demo.controller_start import (
    ControllerSample,
    ControllerStartAdapter,
    ControllerStartRefused,
    Go2ControllerStartSource,
)
from border_collie_demo.mission import MissionMachine
from border_collie_demo.models import RemoteInput
from border_collie_demo.orchestrator import SimulatedStageExecutor


TOPIC = "rt/lf/lowstate"
START_MASK = 1 << 2
REPOSITORY_ROOT = Path(__file__).parents[1]


def sample(sequence: int, buttons: int, received_s: float) -> ControllerSample:
    remote = bytearray(40)
    remote[2:4] = buttons.to_bytes(2, "little")
    return ControllerSample(
        source=TOPIC,
        source_sequence=sequence,
        received_monotonic_s=received_s,
        wireless_remote=bytes(remote),
    )


def cohort_record(cohort_id: str) -> dict[str, object]:
    return {
        "cohort_id": cohort_id,
        "status": "RUNNING",
        "fruit_sequence": ["mango", "pear", "apple"],
        "policy": {"runs": 3, "seed": 7},
    }


class ReadyHardware:
    """A Go2 boundary that is connected, posed, disarmed and safe to activate."""

    def __init__(self) -> None:
        self.started = False
        self.capture_home_calls = 0

    async def start(self) -> None:
        self.started = True

    async def close(self) -> list[str]:
        self.started = False
        return []

    async def emergency_stop(self) -> list[str]:
        return []

    def capture_home(self) -> dict[str, object]:
        self.capture_home_calls += 1
        return {
            "x_m": 1.0,
            "y_m": 2.0,
            "yaw_rad": 0.25,
            "captured_monotonic_s": 3.0,
            "age_s": 0.01,
            "source": "test",
        }

    def status(self) -> dict[str, object]:
        return {
            "configured": True,
            "connected": self.started,
            "fault": None,
            "active_operation": None,
            "pose": {
                "healthy": True,
                "age_s": 0.01,
                "error": None,
                "pose": {"x_m": 1.0, "y_m": 2.0},
            },
            "motion": {
                "armed": False,
                "last_command": {"forward_mps": 0.0, "yaw_rps": 0.0},
            },
        }


def ready_camera() -> dict[str, object]:
    return {"ready": True, "detail": "camera ready"}


class ScriptedSource:
    """Replay a fixed controller edge sequence when the app starts."""

    def __init__(self, edges: list[tuple[int, int]]) -> None:
        self._edges = edges
        self.decisions: list[object] = []
        self.started = False
        self.closed = False

    async def start(self, observer) -> None:
        self.started = True
        for index, (sequence, buttons) in enumerate(self._edges):
            self.decisions.append(
                await observer(sample(sequence, buttons, 900.0 + index * 0.05))
            )

    async def close(self) -> None:
        self.closed = True

    def status(self) -> dict[str, object]:
        return {"connected": self.started and not self.closed, "topic": TOPIC}


class InertSource(ScriptedSource):
    def __init__(self) -> None:
        super().__init__([])


class DrivableSource:
    """Let a test submit controller edges into the running application loop."""

    def __init__(self) -> None:
        self._observer = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self.started = False
        self.closed = False

    async def start(self, observer) -> None:
        self._observer = observer
        self._loop = asyncio.get_running_loop()
        self.started = True

    async def close(self) -> None:
        self.closed = True

    def status(self) -> dict[str, object]:
        return {"connected": self.started and not self.closed, "topic": TOPIC}

    def submit(self, sequence: int, buttons: int, received_s: float):
        assert self._observer is not None and self._loop is not None
        return asyncio.run_coroutine_threadsafe(
            self._observer(sample(sequence, buttons, received_s)), self._loop
        ).result(timeout=5.0)

    def call_on_loop(self, callback) -> None:
        assert self._loop is not None
        self._loop.call_soon_threadsafe(callback)


class GatedStageExecutor(SimulatedStageExecutor):
    """Hold the first Demo Run inside a stage so a cohort stays RUNNING."""

    def __init__(self) -> None:
        super().__init__()
        self.release = asyncio.Event()
        self.gate_reached = False

    async def execute(self, phase, context):
        if phase.value == "find_fruit" and not self.release.is_set():
            self.gate_reached = True
            await self.release.wait()
        return await super().execute(phase, context)


def wait_for_cohort(client: TestClient, timeout_s: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        cohort = client.get("/api/cohorts/active").json()["cohort"]
        if cohort and cohort["status"] != "RUNNING":
            return cohort
        time.sleep(0.01)
    raise AssertionError("cohort did not become terminal")


def test_one_start_edge_requests_exactly_one_three_fruit_cohort(tmp_path) -> None:
    async def scenario() -> None:
        cohort_id = str(uuid4())
        starts = 0
        black_box = RunBlackBox(tmp_path)

        async def start_cohort() -> dict[str, object]:
            nonlocal starts
            starts += 1
            return cohort_record(cohort_id)

        adapter = ControllerStartAdapter(
            start_cohort=start_cohort,
            black_box=black_box,
            active_run_id=lambda: None,
        )

        assert (await adapter.observe(sample(10, 0, 100.0))).disposition == "armed"
        accepted = await adapter.observe(sample(11, START_MASK, 100.05))
        held = await adapter.observe(sample(12, START_MASK, 100.10))
        duplicate = await adapter.observe(sample(12, START_MASK, 100.10))

        assert accepted.disposition == "accepted"
        assert accepted.cohort_id == cohort_id
        assert accepted.activation_id == f"go2-controller-start:{TOPIC}:11"
        assert held.disposition == "held"
        assert duplicate.disposition == "duplicate"
        assert starts == 1

        status = adapter.status()
        assert status["ready"] is True
        assert status["accepted_edges"] == 1
        assert status["last_cohort_id"] == cohort_id
        assert status["recent_edges"] == [
            {
                "activation_id": f"go2-controller-start:{TOPIC}:11",
                "button": "start",
                "button_mask": START_MASK,
                "button_word": START_MASK,
                "cohort_id": cohort_id,
                "disposition": "accepted",
                "fruit_sequence": ["mango", "pear", "apple"],
                "received_monotonic_s": 100.05,
                "runs": 3,
                "seed": 7,
                "source": TOPIC,
                "source_sequence": 11,
                "start_pressed": True,
            }
        ]
        black_box.close()

    asyncio.run(scenario())


def test_start_held_across_boot_is_inert_until_release_and_new_edge(tmp_path) -> None:
    async def scenario() -> None:
        starts = 0

        async def start_cohort() -> dict[str, object]:
            nonlocal starts
            starts += 1
            return cohort_record(str(uuid4()))

        black_box = RunBlackBox(tmp_path)
        adapter = ControllerStartAdapter(
            start_cohort=start_cohort,
            black_box=black_box,
            active_run_id=lambda: None,
        )

        first = await adapter.observe(sample(20, START_MASK, 200.0))
        held = await adapter.observe(sample(21, START_MASK, 200.1))
        released = await adapter.observe(sample(22, 0, 200.2))
        pressed = await adapter.observe(sample(23, START_MASK, 200.3))

        assert first.disposition == "startup_held"
        assert held.disposition == "startup_held"
        assert released.disposition == "armed"
        assert pressed.disposition == "accepted"
        assert starts == 1
        black_box.close()

    asyncio.run(scenario())


def test_edges_during_an_active_cohort_are_refused_and_never_stack(tmp_path) -> None:
    async def scenario() -> None:
        active_run_id = str(uuid4())
        black_box = RunBlackBox(tmp_path)
        black_box.record(active_run_id, "run_started", phase="find_fruit", payload={})
        attempts = 0

        async def start_cohort() -> dict[str, object]:
            nonlocal attempts
            attempts += 1
            raise ControllerStartRefused(
                "active_cohort", "the active cohort owns Demo Run activation"
            )

        adapter = ControllerStartAdapter(
            start_cohort=start_cohort,
            black_box=black_box,
            active_run_id=lambda: active_run_id,
        )

        await adapter.observe(sample(30, 0, 300.0))
        first = await adapter.observe(sample(31, START_MASK, 300.1))
        await adapter.observe(sample(32, 0, 300.2))
        second = await adapter.observe(sample(33, START_MASK, 300.3))

        assert first.disposition == "active_cohort"
        assert second.disposition == "active_cohort"
        assert first.detail == "the active cohort owns Demo Run activation"
        assert attempts == 2
        assert adapter.status()["accepted_edges"] == 0

        events = [
            event
            for event in black_box.read(active_run_id)
            if event["kind"] == "controller_start"
        ]
        assert [event["payload"]["disposition"] for event in events] == [
            "active_cohort",
            "active_cohort",
        ]
        assert events[0]["payload"]["error"] == (
            "the active cohort owns Demo Run activation"
        )
        black_box.close()

    asyncio.run(scenario())


def test_latched_remote_takeover_refuses_the_edge_without_touching_cohorts(
    tmp_path,
) -> None:
    async def scenario() -> None:
        black_box = RunBlackBox(tmp_path)

        async def start_cohort() -> dict[str, object]:
            raise ControllerStartRefused(
                "takeover_latched",
                "physical remote takeover is latched; restart required",
            )

        adapter = ControllerStartAdapter(
            start_cohort=start_cohort,
            black_box=black_box,
            active_run_id=lambda: None,
        )

        await adapter.observe(sample(40, 0, 400.0))
        refused = await adapter.observe(sample(41, START_MASK, 400.1))

        assert refused.disposition == "takeover_latched"
        assert adapter.status()["accepted_edges"] == 0
        assert adapter.status()["last_error"] == (
            "physical remote takeover is latched; restart required"
        )
        black_box.close()

    asyncio.run(scenario())


def test_unexpected_cohort_failure_is_refused_rather_than_raised(tmp_path) -> None:
    async def scenario() -> None:
        black_box = RunBlackBox(tmp_path)

        async def start_cohort() -> dict[str, object]:
            raise CohortConflict("activation readiness has not passed")

        adapter = ControllerStartAdapter(
            start_cohort=start_cohort,
            black_box=black_box,
            active_run_id=lambda: None,
        )

        await adapter.observe(sample(50, 0, 500.0))
        refused = await adapter.observe(sample(51, START_MASK, 500.1))

        assert refused.disposition == "rejected"
        assert refused.detail == "activation readiness has not passed"
        black_box.close()

    asyncio.run(scenario())


def test_controller_source_ignores_malformed_samples_without_crashing() -> None:
    async def scenario() -> None:
        observed: list[ControllerSample] = []

        class Subscriber:
            def __init__(self, topic, message_type) -> None:
                self.handler = None

            def Init(self, handler, depth) -> None:
                self.handler = handler

            def Close(self) -> None:
                pass

        created: list[Subscriber] = []

        def factory(topic, message_type):
            subscriber = Subscriber(topic, message_type)
            created.append(subscriber)
            return subscriber

        source = Go2ControllerStartSource(
            subscriber_factory=factory,
            message_type=object(),
            monotonic_clock=lambda: 12.5,
        )

        async def observer(sample_in: ControllerSample) -> None:
            observed.append(sample_in)

        await source.start(observer)

        class Message:
            def __init__(self, tick, remote) -> None:
                self.tick = tick
                self.wireless_remote = remote

        created[0].handler(Message(1, b"\x00"))
        created[0].handler(Message(2, bytes(40)))
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        assert [item.source_sequence for item in observed] == [2]
        assert source.status()["received_samples"] == 1
        await source.close()

    asyncio.run(scenario())


def test_controller_start_runs_one_apple_one_mango_one_pear(tmp_path) -> None:
    source = ScriptedSource([(90, 0), (91, START_MASK)])
    app = create_app(
        runs_root=tmp_path / "runs",
        cohorts_root=tmp_path / "cohorts",
        hardware=ReadyHardware(),
        camera_perception_status=ready_camera,
        stage_executor=SimulatedStageExecutor(),
        controller_start_source=source,
    )

    with TestClient(app) as client:
        assert [decision.disposition for decision in source.decisions] == [
            "armed",
            "accepted",
        ]
        cohort = wait_for_cohort(client)

        assert cohort["status"] == "COMPLETED"
        assert cohort["policy"]["runs"] == 3
        assert cohort["selected_fruits"] == ["apple", "mango", "pear"]
        assert sorted(cohort["fruit_sequence"]) == ["apple", "mango", "pear"]
        assert [record["target_fruit"] for record in cohort["runs"]] == cohort[
            "fruit_sequence"
        ]
        assert len(cohort["runs"]) == 3
        assert all(record["outcome"] == "COMPLETED" for record in cohort["runs"])
        assert all(
            record["final_safety_state"] == "DISARMED_CONFIRMED"
            for record in cohort["runs"]
        )
        assert "banana" not in cohort["fruit_sequence"]

        status = client.get("/api/status").json()["controller_start"]
        assert status["enabled"] is True
        assert status["adapter"]["accepted_edges"] == 1
        assert status["adapter"]["last_cohort_id"] == cohort["cohort_id"]
        assert status["cohort_request"]["fruit_subset"] == ["apple", "mango", "pear"]
        assert status["cohort_request"]["runs"] == 3

    assert source.closed is True


def test_app_boot_alone_arms_the_source_but_never_activates_motion(tmp_path) -> None:
    hardware = ReadyHardware()
    source = InertSource()
    app = create_app(
        runs_root=tmp_path / "runs",
        cohorts_root=tmp_path / "cohorts",
        hardware=hardware,
        camera_perception_status=ready_camera,
        stage_executor=SimulatedStageExecutor(),
        controller_start_source=source,
    )

    with TestClient(app) as client:
        status = client.get("/api/status").json()

        assert source.started is True
        assert status["controller_start"]["enabled"] is True
        assert status["controller_start"]["source"]["connected"] is True
        assert status["controller_start"]["adapter"]["ready"] is False
        assert status["active_run_id"] is None
        assert status["cohort"] is None
        assert hardware.capture_home_calls == 0
        assert hardware.status()["motion"]["armed"] is False

    assert source.closed is True


def test_status_reports_controller_start_disabled_when_no_source_is_wired(
    tmp_path,
) -> None:
    app = create_app(
        runs_root=tmp_path / "runs",
        cohorts_root=tmp_path / "cohorts",
        hardware=ReadyHardware(),
        camera_perception_status=ready_camera,
        stage_executor=SimulatedStageExecutor(),
    )

    with TestClient(app) as client:
        assert client.get("/api/status").json()["controller_start"] == {
            "enabled": False,
            "detail": "physical Start activation is disabled",
        }


def test_start_edge_is_refused_while_a_cohort_owns_activation(tmp_path) -> None:
    source = DrivableSource()
    executor = GatedStageExecutor()
    app = create_app(
        runs_root=tmp_path / "runs",
        cohorts_root=tmp_path / "cohorts",
        hardware=ReadyHardware(),
        camera_perception_status=ready_camera,
        stage_executor=executor,
        controller_start_source=source,
    )

    with TestClient(app) as client:
        assert source.submit(70, 0, 700.0).disposition == "armed"
        assert source.submit(71, START_MASK, 700.1).disposition == "accepted"

        deadline = time.monotonic() + 5.0
        while not executor.gate_reached and time.monotonic() < deadline:
            time.sleep(0.01)
        assert executor.gate_reached is True

        # The same guard POST /api/cohorts answers 409 with, so one operator
        # press can never stack a second cohort on the running one.
        assert (
            client.post(
                "/api/cohorts", json={"runs": 3, "randomized": True, "seed": 3}
            ).status_code
            == 409
        )
        source.submit(72, 0, 700.2)
        refused = source.submit(73, START_MASK, 700.3)
        assert refused.disposition == "active_cohort"
        assert refused.detail == "the active cohort owns Demo Run activation"

        source.call_on_loop(executor.release.set)
        cohort = wait_for_cohort(client)
        assert cohort["status"] == "COMPLETED"
        assert len(cohort["runs"]) == 3

        adapter_status = client.get("/api/status").json()["controller_start"]
        assert adapter_status["adapter"]["accepted_edges"] == 1
        assert adapter_status["adapter"]["last_disposition"] == "active_cohort"


def test_start_edge_is_refused_while_a_single_demo_run_is_active(tmp_path) -> None:
    source = DrivableSource()
    executor = GatedStageExecutor()
    app = create_app(
        runs_root=tmp_path / "runs",
        cohorts_root=tmp_path / "cohorts",
        hardware=ReadyHardware(),
        camera_perception_status=ready_camera,
        stage_executor=executor,
        controller_start_source=source,
    )

    with TestClient(app) as client:
        assert source.submit(60, 0, 600.0).disposition == "armed"
        started = client.post("/api/run", json={"target_fruit": "pear"})
        assert started.status_code == 201
        assert client.get("/api/status").json()["active_run_id"] is not None

        refused = source.submit(61, START_MASK, 600.1)
        assert refused.disposition == "active_run"
        assert refused.detail == "a Demo Run is already active"
        assert client.get("/api/cohorts/active").json()["cohort"] is None

        source.call_on_loop(executor.release.set)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if client.get("/api/status").json()["active_run_id"] is None:
                break
            time.sleep(0.01)
        assert client.get("/api/status").json()["active_run_id"] is None


def test_start_edge_while_remote_takeover_is_latched_is_refused(tmp_path) -> None:
    machine = MissionMachine()
    machine.remote_takeover(
        RemoteInput(
            source="go2_controller",
            control="wireless_remote",
            received_monotonic_s=1.0,
        )
    )
    source = ScriptedSource([(80, 0), (81, START_MASK)])
    app = create_app(
        mission=machine,
        runs_root=tmp_path / "runs",
        cohorts_root=tmp_path / "cohorts",
        hardware=ReadyHardware(),
        camera_perception_status=ready_camera,
        stage_executor=SimulatedStageExecutor(),
        controller_start_source=source,
    )

    with TestClient(app) as client:
        assert [decision.disposition for decision in source.decisions] == [
            "armed",
            "takeover_latched",
        ]
        assert client.get("/api/cohorts/active").json()["cohort"] is None
        adapter_status = client.get("/api/status").json()["controller_start"]["adapter"]
        assert adapter_status["accepted_edges"] == 0
        assert adapter_status["last_disposition"] == "takeover_latched"


def test_controller_cohort_request_matches_the_three_fruit_ui_button() -> None:
    request = three_fruit_cohort_request(seed=11)

    assert request.runs == 3
    assert request.randomized is True
    assert request.target_fruit is None
    assert request.fruit_subset == ["apple", "mango", "pear"]
    assert request.seed == 11
    assert request.tuning == {
        "search": {"yaw_rps": CONTROLLER_START_SEARCH_YAW_RPS},
        "home": {"align_yaw_rps": CONTROLLER_START_HOME_ALIGN_YAW_RPS},
    }
    assert [item.model_dump() for item in request.tolerated_failures] == [
        {"failed_phase": None, "reason": None} | dict(item)
        for item in CONTROLLER_START_TOLERATED_FAILURES
    ]
    assert "banana" not in CONTROLLER_START_COHORT_FRUITS
    # Deployed stage defaults; changing them requires re-qualification on Woof.
    assert CONTROLLER_START_SEARCH_YAW_RPS == 0.80
    assert CONTROLLER_START_HOME_ALIGN_YAW_RPS == 0.80


def test_controller_cohort_shape_does_not_drift_from_the_ui_button() -> None:
    markup = (REPOSITORY_ROOT / "web" / "index.html").read_text(encoding="utf-8")
    handler = markup.split("startThreeFruit.addEventListener", 1)[1].split(
        "stopCohort.addEventListener", 1
    )[0]

    assert "runs: 3" in handler
    assert "fruit_subset: ['apple', 'mango', 'pear']" in handler
    for selector in CONTROLLER_START_TOLERATED_FAILURES:
        for value in selector.values():
            assert f"'{value}'" in handler
