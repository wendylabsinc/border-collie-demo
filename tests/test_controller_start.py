"""Physical Go2 controller input: three presses start, any touch stops."""

from __future__ import annotations

import asyncio
import struct
import time
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from border_collie_demo.api import (
    CONTROLLER_ACTIVATION_SOURCE,
    CONTROLLER_START_COHORT_FRUITS,
    CONTROLLER_START_HOME_ALIGN_YAW_RPS,
    CONTROLLER_START_SEARCH_YAW_RPS,
    CONTROLLER_START_TOLERATED_FAILURES,
    controller_run_tuning,
    create_app,
    three_fruit_cohort_request,
)
from border_collie_demo.black_box import RunBlackBox
from border_collie_demo.cohorts import CohortConflict
from border_collie_demo.controller_start import (
    BUTTON_MASKS,
    BUTTON_NAMES,
    FRUIT_RUN_BUTTON_MASKS,
    FRUIT_RUN_BUTTONS,
    START_PRESS_COUNT,
    START_SEQUENCE_WINDOW_S,
    STICK_DEADZONE,
    TAKEOVER_RELEASE_HOLD_S,
    ControllerSample,
    ControllerStartAdapter,
    ControllerStartRefused,
    Go2ControllerStartSource,
    decode_frame,
)
from border_collie_demo.fruits import QUALIFIED_FRUITS
from border_collie_demo.mission import MissionMachine
from border_collie_demo.models import RemoteInput
from border_collie_demo.orchestrator import SimulatedStageExecutor


TOPIC = "rt/lf/lowstate"
START_MASK = 1 << 2
SELECT_MASK = 1 << 3
A_MASK = 1 << 8
# Face buttons that each request one single-fruit Demo Run. The bit indices come
# from Unitree's xKeySwitchUnion order (R1, L1, start, select, R2, L2, F1, F2,
# A, B, X, Y, up, right, down, left), which is the same order decode_frame
# reads. Written out literally here so the production masks are checked against
# a second, independent statement of the layout rather than against themselves.
B_MASK = 1 << 9
X_MASK = 1 << 10
Y_MASK = 1 << 11
FRUIT_BUTTON_CASES = (
    ("X", X_MASK, "pear"),
    ("Y", Y_MASK, "mango"),
    ("B", B_MASK, "apple"),
)
REPOSITORY_ROOT = Path(__file__).parents[1]

# Byte offsets of the four watched stick axes inside wireless_remote. The
# analog L2 float sits at offset 16, which is why ly is at 20 and not 16.
AXIS_OFFSETS = {"lx": 4, "rx": 8, "ry": 12, "ly": 20}


def frame(buttons: int = 0, axes: dict[str, float] | None = None) -> bytes:
    remote = bytearray(40)
    remote[2:4] = buttons.to_bytes(2, "little")
    for name, offset in AXIS_OFFSETS.items():
        struct.pack_into("<f", remote, offset, float((axes or {}).get(name, 0.0)))
    return bytes(remote)


def sample(
    sequence: int,
    buttons: int = 0,
    received_s: float = 0.0,
    axes: dict[str, float] | None = None,
) -> ControllerSample:
    return ControllerSample(
        source=TOPIC,
        source_sequence=sequence,
        received_monotonic_s=received_s,
        wireless_remote=frame(buttons, axes),
    )


def cohort_record(cohort_id: str) -> dict[str, object]:
    return {
        "cohort_id": cohort_id,
        "status": "RUNNING",
        "fruit_sequence": ["mango", "pear", "apple"],
        "policy": {"runs": 3, "seed": 7},
    }


def button_edges(
    first_sequence: int, mask: int = START_MASK, presses: int = START_PRESS_COUNT
):
    """A neutral arming sample followed by ``presses`` release/press pairs."""
    edges = [(first_sequence, 0)]
    for index in range(presses):
        edges.append((first_sequence + 1 + index * 2, mask))
        edges.append((first_sequence + 2 + index * 2, 0))
    return edges


def start_edges(first_sequence: int, presses: int = START_PRESS_COUNT):
    return button_edges(first_sequence, START_MASK, presses)


async def arm(adapter: ControllerStartAdapter, sequence: int, at_s: float):
    return await adapter.observe(sample(sequence, 0, at_s))


async def press_button(
    adapter: ControllerStartAdapter, mask: int, sequence: int, at_s: float
):
    """One complete press of ``mask``: the button goes down, then comes up."""
    decision = await adapter.observe(sample(sequence, mask, at_s))
    await adapter.observe(sample(sequence + 1, 0, at_s + 0.02))
    return decision


async def press_start(adapter: ControllerStartAdapter, sequence: int, at_s: float):
    """One complete Start press: the button goes down, then comes back up."""
    return await press_button(adapter, START_MASK, sequence, at_s)


async def press_button_times(
    adapter: ControllerStartAdapter,
    mask: int,
    first_sequence: int,
    at_s: float,
    presses: int = START_PRESS_COUNT,
    spacing_s: float = 0.1,
):
    """Press ``mask`` ``presses`` times, returning the last decision."""
    decision = None
    for index in range(presses):
        decision = await press_button(
            adapter,
            mask,
            first_sequence + index * 2,
            at_s + index * spacing_s,
        )
    return decision


class ReadyHardware:
    """A Go2 boundary that is connected, posed, disarmed and safe to activate."""

    def __init__(self) -> None:
        self.started = False
        self.capture_home_calls = 0
        self.emergency_stops = 0

    async def start(self) -> None:
        self.started = True

    async def close(self) -> list[str]:
        self.started = False
        return []

    async def emergency_stop(self) -> list[str]:
        self.emergency_stops += 1
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

    def dispositions(self) -> list[str]:
        return [decision.disposition for decision in self.decisions]


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

    def submit(
        self,
        sequence: int,
        buttons: int,
        received_s: float,
        axes: dict[str, float] | None = None,
    ):
        assert self._observer is not None and self._loop is not None
        return asyncio.run_coroutine_threadsafe(
            self._observer(sample(sequence, buttons, received_s, axes)), self._loop
        ).result(timeout=10.0)

    def press(self, mask: int, sequence: int, received_s: float):
        decision = self.submit(sequence, mask, received_s)
        self.submit(sequence + 1, 0, received_s + 0.02)
        return decision

    def press_start(self, sequence: int, received_s: float):
        return self.press(START_MASK, sequence, received_s)

    def launch_button(self, mask: int, first_sequence: int, at_s: float):
        """Arm, then press ``mask`` the full required number of times."""
        self.submit(first_sequence, 0, at_s)
        decision = None
        for index in range(START_PRESS_COUNT):
            decision = self.press(
                mask, first_sequence + 1 + index * 2, at_s + 0.1 * index
            )
        return decision

    def launch(self, first_sequence: int, at_s: float):
        """Arm, then press Start the full required number of times."""
        return self.launch_button(START_MASK, first_sequence, at_s)

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


def wait_for_cohort(client: TestClient, timeout_s: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        cohort = client.get("/api/cohorts/active").json()["cohort"]
        if cohort and cohort["status"] != "RUNNING":
            return cohort
        time.sleep(0.01)
    raise AssertionError("cohort did not become terminal")


def wait_until(predicate, timeout_s: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition was never reached")


# ---------------------------------------------------------------------------
# Frame decoding
# ---------------------------------------------------------------------------


def test_frame_decodes_buttons_and_axes_at_the_pinned_sdk_offsets() -> None:
    decoded = decode_frame(
        frame(START_MASK | A_MASK, {"lx": 0.0, "rx": -0.9, "ry": 0.0, "ly": 0.5})
    )

    assert decoded.buttons == ("start", "A")
    assert dict(decoded.axes) == pytest.approx(
        {
            "left_stick_x": 0.0,
            "right_stick_x": -0.9,
            "right_stick_y": 0.0,
            "left_stick_y": 0.5,
        }
    )
    assert sorted(decoded.displaced_axes) == ["left_stick_y", "right_stick_x"]
    assert decoded.neutral is False
    assert decode_frame(frame()).neutral is True


def test_axis_layout_skips_the_analog_l2_slot() -> None:
    """ly lives at offset 20; offset 16 is the analog L2 the SDK marks unused."""
    remote = bytearray(40)
    struct.pack_into("<f", remote, 16, 1.0)

    assert decode_frame(bytes(remote)).neutral is True

    struct.pack_into("<f", remote, 20, 1.0)
    assert decode_frame(bytes(remote)).displaced_axes == ("left_stick_y",)


def test_a_frame_too_short_to_prove_the_sticks_are_centered_is_rejected() -> None:
    with pytest.raises(ValueError, match="24 bytes"):
        ControllerSample(TOPIC, 1, 1.0, bytes(20))


def test_a_corrupt_axis_float_is_evidence_not_an_operator_input() -> None:
    remote = bytearray(40)
    struct.pack_into("<f", remote, 4, float("nan"))

    decoded = decode_frame(bytes(remote))

    assert decoded.displaced_axes == ()
    assert decoded.neutral is True


# ---------------------------------------------------------------------------
# Task C: three presses to start
# ---------------------------------------------------------------------------


def test_three_start_presses_request_exactly_one_three_fruit_cohort(tmp_path) -> None:
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

        assert (await arm(adapter, 10, 100.0)).disposition == "armed"
        first = await press_start(adapter, 11, 100.1)
        assert adapter.status()["start_press_progress"] == "1 of 3"
        second = await press_start(adapter, 13, 100.2)
        assert adapter.status()["start_press_progress"] == "2 of 3"
        third = await press_start(adapter, 15, 100.3)

        assert first.disposition == "start_press_recorded"
        assert second.disposition == "start_press_recorded"
        assert third.disposition == "accepted"
        assert third.cohort_id == cohort_id
        assert third.activation_id == f"go2-controller-start:{TOPIC}:15"
        assert starts == 1

        status = adapter.status()
        assert status["accepted_edges"] == 1
        assert status["last_cohort_id"] == cohort_id
        # The counter resets after a launch, so pressing again does not
        # immediately relaunch.
        assert status["start_presses_recorded"] == 0
        assert status["start_press_progress"] == "0 of 3"
        black_box.close()

    asyncio.run(scenario())


def test_two_presses_then_a_lapsed_window_launch_nothing(tmp_path) -> None:
    async def scenario() -> None:
        starts = 0
        black_box = RunBlackBox(tmp_path)

        async def start_cohort() -> dict[str, object]:
            nonlocal starts
            starts += 1
            return cohort_record(str(uuid4()))

        adapter = ControllerStartAdapter(
            start_cohort=start_cohort,
            black_box=black_box,
            active_run_id=lambda: None,
        )

        await arm(adapter, 20, 200.0)
        await press_start(adapter, 21, 200.1)
        await press_start(adapter, 23, 200.2)
        assert adapter.status()["start_presses_recorded"] == 2

        # A neutral sample past the window expires the partial sequence.
        lapsed = await adapter.observe(
            sample(25, 0, 200.2 + START_SEQUENCE_WINDOW_S + 0.1)
        )

        assert lapsed.disposition == "start_sequence_expired"
        assert adapter.status()["start_presses_recorded"] == 0
        assert starts == 0

        # The next press is press one of a fresh sequence, not press three.
        third = await press_start(adapter, 26, 400.0)
        assert third.disposition == "start_press_recorded"
        assert starts == 0
        black_box.close()

    asyncio.run(scenario())


def test_holding_start_down_is_one_press_not_three(tmp_path) -> None:
    async def scenario() -> None:
        starts = 0
        black_box = RunBlackBox(tmp_path)

        async def start_cohort() -> dict[str, object]:
            nonlocal starts
            starts += 1
            return cohort_record(str(uuid4()))

        adapter = ControllerStartAdapter(
            start_cohort=start_cohort,
            black_box=black_box,
            active_run_id=lambda: None,
        )

        await arm(adapter, 30, 300.0)
        first = await adapter.observe(sample(31, START_MASK, 300.1))
        held = [
            await adapter.observe(sample(32 + index, START_MASK, 300.2 + index * 0.05))
            for index in range(8)
        ]

        assert first.disposition == "start_press_recorded"
        assert {decision.disposition for decision in held} == {"held"}
        assert adapter.status()["start_presses_recorded"] == 1
        assert starts == 0
        black_box.close()

    asyncio.run(scenario())


def test_start_held_across_boot_is_inert_until_release_and_three_presses(
    tmp_path,
) -> None:
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

        first = await adapter.observe(sample(40, START_MASK, 200.0))
        held = await adapter.observe(sample(41, START_MASK, 200.1))
        released = await adapter.observe(sample(42, 0, 200.2))

        assert first.disposition == "startup_held"
        assert held.disposition == "startup_held"
        assert released.disposition == "armed"
        assert starts == 0

        await press_start(adapter, 43, 200.3)
        await press_start(adapter, 45, 200.4)
        accepted = await press_start(adapter, 47, 200.5)

        assert accepted.disposition == "accepted"
        assert starts == 1
        black_box.close()

    asyncio.run(scenario())


def test_a_stick_displaced_across_boot_blocks_arming_until_it_is_centered(
    tmp_path,
) -> None:
    async def scenario() -> None:
        black_box = RunBlackBox(tmp_path)
        adapter = ControllerStartAdapter(
            start_cohort=lambda: None,
            black_box=black_box,
            active_run_id=lambda: None,
        )

        leaning = await adapter.observe(sample(50, 0, 500.0, {"ly": 0.8}))
        still_leaning = await adapter.observe(sample(51, 0, 500.1, {"ly": 0.8}))
        centered = await adapter.observe(sample(52, 0, 500.2))

        assert leaning.disposition == "startup_held"
        assert still_leaning.disposition == "startup_held"
        assert centered.disposition == "armed"
        assert adapter.status()["ready"] is True
        black_box.close()

    asyncio.run(scenario())


def test_any_other_control_clears_a_partial_start_sequence(tmp_path) -> None:
    async def scenario() -> None:
        black_box = RunBlackBox(tmp_path)
        adapter = ControllerStartAdapter(
            start_cohort=lambda: None,
            black_box=black_box,
            active_run_id=lambda: None,
        )

        await arm(adapter, 60, 600.0)
        await press_start(adapter, 61, 600.1)
        await press_start(adapter, 63, 600.2)
        assert adapter.status()["start_presses_recorded"] == 2

        interrupted = await adapter.observe(sample(65, SELECT_MASK, 600.3))

        assert interrupted.disposition == "other_input"
        assert adapter.status()["start_presses_recorded"] == 0
        black_box.close()

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# Task A: any input stops an active run
# ---------------------------------------------------------------------------


def make_stopping_adapter(black_box: RunBlackBox, *, run_id: str | None = "run-1"):
    """An adapter whose robot is busy and whose stop path is recorded."""
    stops: list[RemoteInput] = []
    latched = {"value": False}

    async def stop_active(remote_input: RemoteInput) -> dict[str, object]:
        stops.append(remote_input)
        latched["value"] = True
        return {"stopped": True}

    adapter = ControllerStartAdapter(
        start_cohort=lambda: None,
        black_box=black_box,
        active_run_id=lambda: run_id,
        stop_active=stop_active,
        is_busy=lambda: not latched["value"],
        takeover_latched=lambda: latched["value"],
        release_takeover=lambda: latched.update(value=False),
    )
    return adapter, stops, latched


@pytest.mark.parametrize(
    ("label", "buttons", "axes"),
    [
        ("start", START_MASK, None),
        ("select", SELECT_MASK, None),
        ("A", A_MASK, None),
        ("left stick", 0, {"ly": 0.9}),
        ("right stick", 0, {"rx": -0.7}),
    ],
)
def test_any_controller_input_stops_an_active_run(
    tmp_path, label, buttons, axes
) -> None:
    async def scenario() -> None:
        black_box = RunBlackBox(tmp_path)
        adapter, stops, latched = make_stopping_adapter(black_box)

        await arm(adapter, 70, 700.0)
        stopped = await adapter.observe(sample(71, buttons, 700.1, axes))

        assert stopped.disposition == "takeover_stop", label
        assert len(stops) == 1
        assert stops[0].source == TOPIC
        assert latched["value"] is True
        assert adapter.status()["takeover"]["stops"] == 1
        black_box.close()

    asyncio.run(scenario())


def test_a_stick_inside_the_deadzone_never_stops_a_run(tmp_path) -> None:
    async def scenario() -> None:
        black_box = RunBlackBox(tmp_path)
        adapter, stops, _ = make_stopping_adapter(black_box)

        await arm(adapter, 80, 800.0)
        drifting = [
            await adapter.observe(
                sample(81 + index, 0, 800.1 + index * 0.05, {"lx": value, "ry": -value})
            )
            # Well inside the band, including the 0.12 figure measured as the
            # resting drift of real hardware. The exact boundary is not
            # asserted: these axes arrive as float32, so 0.15 round-trips to
            # 0.15000000596 and lands either side of an equality test.
            for index, value in enumerate((0.0, 0.01, 0.05, 0.12, 0.14))
        ]

        assert {decision.disposition for decision in drifting} == {"idle"}
        assert stops == []

        # A deliberate push is far past the band and stops on the first sample.
        pushed = await adapter.observe(sample(90, 0, 801.0, {"lx": 0.6}))
        assert pushed.disposition == "takeover_stop"
        assert len(stops) == 1
        black_box.close()

    asyncio.run(scenario())


def test_the_deadzone_sits_above_measured_resting_drift() -> None:
    # Unitree's own SDK dead zone is 0.01 and a hardware-tuned neutral band
    # for these controllers is 0.12; 0.15 clears both with margin while a
    # deliberate push (0.5..1.0) crosses it on the first sample.
    assert STICK_DEADZONE == 0.15
    assert STICK_DEADZONE > 0.12


def test_start_while_a_run_is_active_stops_and_never_counts(tmp_path) -> None:
    async def scenario() -> None:
        black_box = RunBlackBox(tmp_path)
        started = 0

        async def start_cohort() -> dict[str, object]:
            nonlocal started
            started += 1
            return cohort_record(str(uuid4()))

        busy = {"value": True}
        latched = {"value": False}
        stops: list[RemoteInput] = []

        async def stop_active(remote_input: RemoteInput) -> dict[str, object]:
            stops.append(remote_input)
            busy["value"] = False
            latched["value"] = True
            return {}

        adapter = ControllerStartAdapter(
            start_cohort=start_cohort,
            black_box=black_box,
            active_run_id=lambda: "run-1",
            stop_active=stop_active,
            is_busy=lambda: busy["value"],
            takeover_latched=lambda: latched["value"],
            release_takeover=lambda: latched.update(value=False),
        )

        await arm(adapter, 100, 1000.0)
        stopped = await press_start(adapter, 101, 1000.1)

        assert stopped.disposition == "takeover_stop"
        assert stops[0].control == "start"
        assert started == 0
        # The press that stopped the run must not also count toward a relaunch.
        assert adapter.status()["start_presses_recorded"] == 0
        black_box.close()

    asyncio.run(scenario())


def test_the_press_counter_is_cleared_by_a_stop(tmp_path) -> None:
    async def scenario() -> None:
        black_box = RunBlackBox(tmp_path)
        started = 0

        async def start_cohort() -> dict[str, object]:
            nonlocal started
            started += 1
            return cohort_record(str(uuid4()))

        busy = {"value": False}
        latched = {"value": False}

        async def stop_active(remote_input: RemoteInput) -> dict[str, object]:
            busy["value"] = False
            latched["value"] = True
            return {}

        adapter = ControllerStartAdapter(
            start_cohort=start_cohort,
            black_box=black_box,
            active_run_id=lambda: None,
            stop_active=stop_active,
            is_busy=lambda: busy["value"],
            takeover_latched=lambda: latched["value"],
            release_takeover=lambda: latched.update(value=False),
        )

        await arm(adapter, 110, 1100.0)
        await press_start(adapter, 111, 1100.1)
        await press_start(adapter, 113, 1100.2)
        assert adapter.status()["start_presses_recorded"] == 2

        # The UI starts a run, then the operator touches the controller.
        busy["value"] = True
        stopped = await adapter.observe(sample(115, A_MASK, 1100.3))
        assert stopped.disposition == "takeover_stop"
        assert adapter.status()["start_presses_recorded"] == 0

        # Release, wait out the hold, and the counter is still zero: the two
        # earlier presses cannot combine with one new press to relaunch.
        await adapter.observe(sample(116, 0, 1100.4))
        released = await adapter.observe(
            sample(117, 0, 1100.4 + TAKEOVER_RELEASE_HOLD_S + 0.1)
        )
        assert released.disposition == "takeover_released"
        assert adapter.status()["start_presses_recorded"] == 0

        again = await press_start(adapter, 118, 1200.0)
        assert again.disposition == "start_press_recorded"
        assert started == 0
        black_box.close()

    asyncio.run(scenario())


def test_a_latched_takeover_blocks_everything_until_a_settled_release(
    tmp_path,
) -> None:
    async def scenario() -> None:
        black_box = RunBlackBox(tmp_path)
        adapter, _, latched = make_stopping_adapter(black_box)

        await arm(adapter, 120, 1200.0)
        await adapter.observe(sample(121, A_MASK, 1200.1))
        assert latched["value"] is True

        # Still holding: nothing moves.
        assert (
            await adapter.observe(sample(122, A_MASK, 1200.2))
        ).disposition == "takeover_latched"
        # Released, but the hold has not elapsed yet.
        assert (
            await adapter.observe(sample(123, 0, 1200.3))
        ).disposition == "takeover_latched"
        assert (
            await adapter.observe(sample(124, 0, 1200.4))
        ).disposition == "takeover_latched"
        # Touching a stick again restarts the hold from zero, so a controller
        # still being handled can never age its way out of the latch.
        assert (
            await adapter.observe(sample(125, 0, 1200.5, {"lx": 0.9}))
        ).disposition == "takeover_latched"
        assert (
            await adapter.observe(sample(126, 0, 1200.6))
        ).disposition == "takeover_latched"
        assert (
            await adapter.observe(
                sample(127, 0, 1200.6 + TAKEOVER_RELEASE_HOLD_S - 0.1)
            )
        ).disposition == "takeover_latched"

        # Only a genuine continuous neutral hold hands control back.
        released = await adapter.observe(
            sample(128, 0, 1200.6 + TAKEOVER_RELEASE_HOLD_S + 0.1)
        )

        assert released.disposition == "takeover_released"
        assert latched["value"] is False
        black_box.close()

    asyncio.run(scenario())


def test_a_takeover_stop_is_attributed_to_the_run_it_interrupted(tmp_path) -> None:
    async def scenario() -> None:
        run_id = str(uuid4())
        black_box = RunBlackBox(tmp_path)
        black_box.record(run_id, "run_started", phase="find_fruit", payload={})
        adapter, _, _ = make_stopping_adapter(black_box, run_id=run_id)

        await arm(adapter, 130, 1300.0)
        await adapter.observe(sample(131, 0, 1300.1, {"ry": -0.9}))

        events = [
            event
            for event in black_box.read(run_id)
            if event["kind"] == "controller_input"
        ]
        assert len(events) == 1
        assert events[0]["payload"]["control"] == "right_stick_y"
        assert events[0]["payload"]["disposition"] == "takeover_stop"
        assert events[0]["payload"]["axes_displaced"] == ["right_stick_y"]
        black_box.close()

    asyncio.run(scenario())


def test_a_control_already_held_when_the_run_began_does_not_stop_it(tmp_path) -> None:
    """The launching press is still down for a few frames; it must not stop."""

    async def scenario() -> None:
        black_box = RunBlackBox(tmp_path)
        stops: list[RemoteInput] = []
        busy = {"value": False}

        async def start_cohort() -> dict[str, object]:
            busy["value"] = True
            return cohort_record(str(uuid4()))

        async def stop_active(remote_input: RemoteInput) -> dict[str, object]:
            stops.append(remote_input)
            return {}

        adapter = ControllerStartAdapter(
            start_cohort=start_cohort,
            black_box=black_box,
            active_run_id=lambda: None,
            stop_active=stop_active,
            is_busy=lambda: busy["value"],
        )

        await arm(adapter, 140, 1400.0)
        await press_start(adapter, 141, 1400.1)
        await press_start(adapter, 143, 1400.2)
        accepted = await adapter.observe(sample(145, START_MASK, 1400.3))
        assert accepted.disposition == "accepted"
        assert busy["value"] is True

        # Start is still physically down for the next several frames.
        still_down = [
            await adapter.observe(sample(146 + index, START_MASK, 1400.4 + index * 0.05))
            for index in range(5)
        ]
        assert {decision.disposition for decision in still_down} == {"held"}
        assert stops == []

        # Releasing is not an input either; pressing again is, and stops.
        assert (await adapter.observe(sample(160, 0, 1401.0))).disposition == "idle"
        stopped = await adapter.observe(sample(161, START_MASK, 1401.1))
        assert stopped.disposition == "takeover_stop"
        assert len(stops) == 1
        black_box.close()

    asyncio.run(scenario())


def test_a_missing_stop_path_is_reported_rather_than_silently_ignored(
    tmp_path,
) -> None:
    async def scenario() -> None:
        black_box = RunBlackBox(tmp_path)
        adapter = ControllerStartAdapter(
            start_cohort=lambda: None,
            black_box=black_box,
            active_run_id=lambda: "run-1",
        )

        await arm(adapter, 170, 1700.0)
        decision = await adapter.observe(sample(171, A_MASK, 1700.1))

        assert decision.disposition == "takeover_unavailable"
        assert adapter.status()["takeover"]["stop_path_connected"] is False
        black_box.close()

    asyncio.run(scenario())


def test_a_failing_stop_path_stays_visible_and_never_crashes(tmp_path) -> None:
    async def scenario() -> None:
        black_box = RunBlackBox(tmp_path)

        async def stop_active(remote_input: RemoteInput) -> dict[str, object]:
            raise RuntimeError("sport client unreachable")

        adapter = ControllerStartAdapter(
            start_cohort=lambda: None,
            black_box=black_box,
            active_run_id=lambda: "run-1",
            stop_active=stop_active,
            is_busy=lambda: True,
        )

        await arm(adapter, 180, 1800.0)
        decision = await adapter.observe(sample(181, A_MASK, 1800.1))

        assert decision.disposition == "takeover_stop_failed"
        assert "sport client unreachable" in adapter.status()["last_error"]
        black_box.close()

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# Refusal paths that survive from the single-press design
# ---------------------------------------------------------------------------


def test_a_refused_cohort_is_recorded_and_never_stacks(tmp_path) -> None:
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

        # is_busy is False here: this is the narrow race where the controller
        # believed the robot was idle and the cohort guard disagreed.
        adapter = ControllerStartAdapter(
            start_cohort=start_cohort,
            black_box=black_box,
            active_run_id=lambda: active_run_id,
            is_busy=lambda: False,
        )

        await arm(adapter, 190, 1900.0)
        await press_start(adapter, 191, 1900.1)
        await press_start(adapter, 193, 1900.2)
        refused = await press_start(adapter, 195, 1900.3)

        assert refused.disposition == "active_cohort"
        assert refused.detail == "the active cohort owns Demo Run activation"
        assert attempts == 1
        assert adapter.status()["accepted_edges"] == 0

        events = [
            event
            for event in black_box.read(active_run_id)
            if event["kind"] == "controller_start"
        ]
        assert [event["payload"]["disposition"] for event in events] == ["active_cohort"]
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

        await arm(adapter, 200, 2000.0)
        await press_start(adapter, 201, 2000.1)
        await press_start(adapter, 203, 2000.2)
        refused = await press_start(adapter, 205, 2000.3)

        assert refused.disposition == "rejected"
        assert refused.detail == "activation readiness has not passed"
        black_box.close()

    asyncio.run(scenario())


def test_duplicate_dds_delivery_is_ignored(tmp_path) -> None:
    async def scenario() -> None:
        black_box = RunBlackBox(tmp_path)
        adapter = ControllerStartAdapter(
            start_cohort=lambda: None,
            black_box=black_box,
            active_run_id=lambda: None,
        )

        await arm(adapter, 210, 2100.0)
        await adapter.observe(sample(211, START_MASK, 2100.1))
        duplicate = await adapter.observe(sample(211, START_MASK, 2100.1))

        assert duplicate.disposition == "duplicate"
        assert adapter.status()["start_presses_recorded"] == 1
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


# ---------------------------------------------------------------------------
# Whole-application behaviour
# ---------------------------------------------------------------------------


def test_controller_start_runs_one_apple_one_mango_one_pear(tmp_path) -> None:
    source = ScriptedSource(start_edges(90))
    app = create_app(
        runs_root=tmp_path / "runs",
        cohorts_root=tmp_path / "cohorts",
        hardware=ReadyHardware(),
        camera_perception_status=ready_camera,
        stage_executor=SimulatedStageExecutor(),
        controller_start_source=source,
        inter_run_pause_s=0.0,
    )

    with TestClient(app) as client:
        assert source.dispositions() == [
            "armed",
            "start_press_recorded",
            "released",
            "start_press_recorded",
            "released",
            "accepted",
            # The cohort now owns the robot, so releasing Start is read as an
            # idle controller rather than as progress toward another launch.
            "idle",
        ]
        cohort = wait_for_cohort(client)

        assert cohort["status"] == "COMPLETED"
        assert cohort["policy"]["runs"] == 3
        assert cohort["selected_fruits"] == ["apple", "mango", "pear"]
        assert sorted(cohort["fruit_sequence"]) == ["apple", "mango", "pear"]
        assert len(cohort["runs"]) == 3
        assert all(record["outcome"] == "COMPLETED" for record in cohort["runs"])
        assert "banana" not in cohort["fruit_sequence"]

        status = client.get("/api/status").json()["controller_start"]
        assert status["enabled"] is True
        assert status["adapter"]["accepted_edges"] == 1
        assert status["adapter"]["last_cohort_id"] == cohort["cohort_id"]
        assert status["adapter"]["start_presses_required"] == 3
        assert status["cohort_request"]["fruit_subset"] == ["apple", "mango", "pear"]

    assert source.closed is True


def test_two_presses_alone_never_start_a_cohort_in_the_application(tmp_path) -> None:
    source = ScriptedSource(start_edges(90, presses=2))
    app = create_app(
        runs_root=tmp_path / "runs",
        cohorts_root=tmp_path / "cohorts",
        hardware=ReadyHardware(),
        camera_perception_status=ready_camera,
        stage_executor=SimulatedStageExecutor(),
        controller_start_source=source,
        inter_run_pause_s=0.0,
    )

    with TestClient(app) as client:
        assert "accepted" not in source.dispositions()
        assert client.get("/api/cohorts/active").json()["cohort"] is None
        adapter = client.get("/api/status").json()["controller_start"]["adapter"]
        assert adapter["start_press_progress"] == "2 of 3"
        assert "1 more time" in adapter["detail"]


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


def test_controller_input_stops_a_running_cohort_through_the_safe_path(
    tmp_path,
) -> None:
    source = DrivableSource()
    executor = GatedStageExecutor()
    hardware = ReadyHardware()
    app = create_app(
        runs_root=tmp_path / "runs",
        cohorts_root=tmp_path / "cohorts",
        hardware=hardware,
        camera_perception_status=ready_camera,
        stage_executor=executor,
        controller_start_source=source,
        inter_run_pause_s=0.0,
    )

    with TestClient(app) as client:
        assert source.launch(300, 3000.0).disposition == "accepted"
        wait_until(lambda: executor.gate_reached)
        run_id = client.get("/api/status").json()["active_run_id"]
        assert run_id is not None

        # One touch of an unrelated button stops the whole cohort at once.
        stopped = source.submit(320, A_MASK, 3010.0)
        assert stopped.disposition == "takeover_stop"

        cohort = client.get("/api/cohorts/active").json()["cohort"]
        assert cohort["status"] == "STOPPED"
        status = client.get("/api/status").json()
        assert status["active_run_id"] is None
        assert status["mission"]["phase"] == "remote_takeover"
        assert status["mission"]["restart_required"] is True

        run = client.get(f"/api/results/{run_id}").json()["run"]
        assert run["outcome"] == "STOPPED"
        assert run["final_safety_state"] == "DISARMED_CONFIRMED"

        adapter = status["controller_start"]["adapter"]
        assert adapter["takeover"]["latched"] is True
        assert adapter["takeover"]["stops"] == 1
        assert adapter["start_presses_recorded"] == 0


def test_start_during_a_single_demo_run_stops_it_instead_of_counting(tmp_path) -> None:
    source = DrivableSource()
    executor = GatedStageExecutor()
    app = create_app(
        runs_root=tmp_path / "runs",
        cohorts_root=tmp_path / "cohorts",
        hardware=ReadyHardware(),
        camera_perception_status=ready_camera,
        stage_executor=executor,
        controller_start_source=source,
        inter_run_pause_s=0.0,
    )

    with TestClient(app) as client:
        assert source.submit(400, 0, 4000.0).disposition == "armed"
        started = client.post("/api/run", json={"target_fruit": "pear"})
        assert started.status_code == 201
        run_id = started.json()["run"]["run_id"]
        wait_until(lambda: executor.gate_reached)

        stopped = source.submit(401, START_MASK, 4000.1)

        assert stopped.disposition == "takeover_stop"
        assert client.get("/api/status").json()["active_run_id"] is None
        assert client.get("/api/cohorts/active").json()["cohort"] is None
        run = client.get(f"/api/results/{run_id}").json()["run"]
        assert run["outcome"] == "STOPPED"
        adapter = client.get("/api/status").json()["controller_start"]["adapter"]
        assert adapter["start_presses_recorded"] == 0
        assert adapter["accepted_edges"] == 0


def test_start_edges_while_remote_takeover_is_latched_are_refused(tmp_path) -> None:
    machine = MissionMachine()
    machine.remote_takeover(
        RemoteInput(
            source="go2_controller",
            control="wireless_remote",
            received_monotonic_s=1.0,
        )
    )
    source = ScriptedSource(start_edges(80))
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
        assert "accepted" not in source.dispositions()
        assert set(source.dispositions()[1:]) == {"takeover_latched"}
        assert client.get("/api/cohorts/active").json()["cohort"] is None
        adapter = client.get("/api/status").json()["controller_start"]["adapter"]
        assert adapter["accepted_edges"] == 0
        assert adapter["takeover"]["latched"] is True


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


# ---------------------------------------------------------------------------
# Per-fruit run triggers: X -> Pear, Y -> Mango, B -> Apple
# ---------------------------------------------------------------------------


class ActivationRecorder:
    """Record what the adapter asked for, without touching a robot."""

    def __init__(self) -> None:
        self.fruit_runs: list[tuple[str, str]] = []
        self.cohorts: list[str] = []

    async def start_fruit_run(
        self, *, target_fruit: str, activation_id: str
    ) -> dict[str, object]:
        self.fruit_runs.append((target_fruit, activation_id))
        return {
            "run": {
                "run_id": f"run-{len(self.fruit_runs)}",
                "target_fruit": target_fruit,
            },
            "idempotent_replay": False,
        }

    async def start_cohort(self) -> dict[str, object]:
        cohort_id = str(uuid4())
        self.cohorts.append(cohort_id)
        return cohort_record(cohort_id)

    def fruits(self) -> list[str]:
        return [fruit for fruit, _ in self.fruit_runs]


def make_idle_adapter(black_box: RunBlackBox):
    """An adapter with both activation paths wired and nothing running."""
    recorder = ActivationRecorder()
    adapter = ControllerStartAdapter(
        start_cohort=recorder.start_cohort,
        start_fruit_run=recorder.start_fruit_run,
        black_box=black_box,
        active_run_id=lambda: None,
    )
    return adapter, recorder


def fruit_button_params(*, with_start: bool = False):
    cases = list(FRUIT_BUTTON_CASES)
    if with_start:
        cases.append(("start", START_MASK, None))
    return pytest.mark.parametrize(
        ("button", "mask", "fruit"), cases, ids=[case[0] for case in cases]
    )


def test_fruit_button_masks_agree_with_the_decoder_and_the_sdk_bitfield() -> None:
    """The production masks match an independent restatement of the layout."""
    assert BUTTON_NAMES.index("B") == 9
    assert BUTTON_NAMES.index("X") == 10
    assert BUTTON_NAMES.index("Y") == 11
    assert FRUIT_RUN_BUTTON_MASKS == {"X": X_MASK, "Y": Y_MASK, "B": B_MASK}
    # The pre-existing masks are unchanged, so the derivation did not shift.
    assert BUTTON_MASKS["start"] == START_MASK == 1 << 2
    assert BUTTON_MASKS["select"] == SELECT_MASK
    assert BUTTON_MASKS["A"] == A_MASK

    # Every mask decodes back to exactly its own button and nothing else, so a
    # press of X can never be read as A, B or Y.
    for name, mask, _fruit in FRUIT_BUTTON_CASES:
        decoded = decode_frame(frame(mask))
        assert decoded.buttons == (name,)
        assert decoded.displaced_axes == ()
        assert decoded.neutral is False


def test_fruit_run_buttons_map_the_three_stage_fruits_and_never_banana() -> None:
    assert FRUIT_RUN_BUTTONS == {"X": "pear", "Y": "mango", "B": "apple"}
    assert sorted(FRUIT_RUN_BUTTONS.values()) == sorted(CONTROLLER_START_COHORT_FRUITS)
    assert "banana" not in FRUIT_RUN_BUTTONS.values()
    for fruit in FRUIT_RUN_BUTTONS.values():
        assert fruit in QUALIFIED_FRUITS


def test_a_controller_fruit_run_uses_the_same_tuning_as_the_start_cohort() -> None:
    """A Pear started with X must behave like the Pear inside a Start cohort."""
    assert controller_run_tuning() == three_fruit_cohort_request().tuning
    assert controller_run_tuning() == {
        "search": {"yaw_rps": CONTROLLER_START_SEARCH_YAW_RPS},
        "home": {"align_yaw_rps": CONTROLLER_START_HOME_ALIGN_YAW_RPS},
    }


@fruit_button_params()
def test_three_presses_of_a_fruit_button_start_that_one_fruit(
    tmp_path, button, mask, fruit
) -> None:
    async def scenario() -> None:
        black_box = RunBlackBox(tmp_path)
        adapter, recorder = make_idle_adapter(black_box)

        assert (await arm(adapter, 10, 100.0)).disposition == "armed"
        first = await press_button(adapter, mask, 11, 100.1)
        assert adapter.status()["pending_button"] == button
        assert adapter.status()["pending_press_progress"] == "1 of 3"
        # A partial fruit sequence must never read as progress to a cohort.
        assert adapter.status()["start_press_progress"] == "0 of 3"
        second = await press_button(adapter, mask, 13, 100.2)
        assert adapter.status()["pending_press_progress"] == "2 of 3"
        third = await press_button(adapter, mask, 15, 100.3)

        assert first.disposition == "start_press_recorded"
        assert second.disposition == "start_press_recorded"
        assert third.disposition == "accepted"
        assert third.target_fruit == fruit
        assert third.run_id == "run-1"
        # A single-fruit run is not a cohort.
        assert third.cohort_id is None
        assert third.activation_id == (
            f"go2-controller-{fruit}:{adapter.instance_id}:{TOPIC}:15"
        )
        assert recorder.fruit_runs == [(fruit, third.activation_id)]
        assert recorder.cohorts == []

        status = adapter.status()
        assert status["accepted_edges"] == 1
        assert status["last_target_fruit"] == fruit
        assert status["last_run_id"] == "run-1"
        assert status["last_cohort_id"] is None
        # The counter resets after a launch, so a fourth press cannot relaunch.
        assert status["pending_presses_recorded"] == 0
        assert status["pending_button"] is None
        black_box.close()

    asyncio.run(scenario())


@fruit_button_params(with_start=True)
def test_holding_an_activation_button_down_is_one_press_not_three(
    tmp_path, button, mask, fruit
) -> None:
    """Presses are counted as edges, so a held thumb is exactly one press."""

    async def scenario() -> None:
        black_box = RunBlackBox(tmp_path)
        adapter, recorder = make_idle_adapter(black_box)

        await arm(adapter, 30, 300.0)
        first = await adapter.observe(sample(31, mask, 300.1))
        held = [
            await adapter.observe(sample(32 + index, mask, 300.2 + index * 0.05))
            for index in range(8)
        ]

        assert first.disposition == "start_press_recorded"
        assert {decision.disposition for decision in held} == {"held"}
        assert adapter.status()["pending_presses_recorded"] == 1
        assert adapter.status()["pending_button"] == button
        assert recorder.fruit_runs == []
        assert recorder.cohorts == []
        black_box.close()

    asyncio.run(scenario())


@fruit_button_params()
def test_fruit_presses_spread_beyond_the_window_start_nothing(
    tmp_path, button, mask, fruit
) -> None:
    async def scenario() -> None:
        black_box = RunBlackBox(tmp_path)
        adapter, recorder = make_idle_adapter(black_box)

        await arm(adapter, 20, 200.0)
        await press_button(adapter, mask, 21, 200.0)
        await press_button(adapter, mask, 23, 200.1)
        assert adapter.status()["pending_presses_recorded"] == 2

        # The third press lands outside the window measured from the first, so
        # it opens a fresh sequence instead of completing the stale one.
        late = await press_button(
            adapter, mask, 25, 200.0 + START_SEQUENCE_WINDOW_S + 0.1
        )

        assert late.disposition == "start_press_recorded"
        assert adapter.status()["pending_presses_recorded"] == 1
        assert recorder.fruit_runs == []

        # The button is not disabled, only the stale sequence was: two more
        # presses inside the fresh window do launch.
        await press_button(adapter, mask, 27, 200.0 + START_SEQUENCE_WINDOW_S + 0.2)
        launched = await press_button(
            adapter, mask, 29, 200.0 + START_SEQUENCE_WINDOW_S + 0.3
        )
        assert launched.disposition == "accepted"
        assert recorder.fruits() == [fruit]
        black_box.close()

    asyncio.run(scenario())


def test_mixed_fruit_button_presses_never_accumulate(tmp_path) -> None:
    """X, X, Y starts nothing: each button owns its own count."""

    async def scenario() -> None:
        black_box = RunBlackBox(tmp_path)
        adapter, recorder = make_idle_adapter(black_box)

        await arm(adapter, 40, 400.0)
        await press_button(adapter, X_MASK, 41, 400.0)
        await press_button(adapter, X_MASK, 43, 400.1)
        assert adapter.status()["pending_button"] == "X"
        assert adapter.status()["pending_presses_recorded"] == 2

        # The documented rule: an edge on a different activation button
        # abandons the X sequence and counts as press one of Y's own sequence.
        switched = await press_button(adapter, Y_MASK, 45, 400.2)

        assert switched.disposition == "start_press_recorded"
        assert adapter.status()["pending_button"] == "Y"
        assert adapter.status()["pending_presses_recorded"] == 1
        assert recorder.fruit_runs == []

        # Two further Y presses -- three Y presses in all -- start Mango, and
        # never Pear, even though two X presses came first.
        await press_button(adapter, Y_MASK, 47, 400.3)
        launched = await press_button(adapter, Y_MASK, 49, 400.4)

        assert launched.disposition == "accepted"
        assert launched.target_fruit == "mango"
        assert recorder.fruits() == ["mango"]
        black_box.close()

    asyncio.run(scenario())


def test_start_and_fruit_presses_do_not_combine(tmp_path) -> None:
    async def scenario() -> None:
        black_box = RunBlackBox(tmp_path)
        adapter, recorder = make_idle_adapter(black_box)

        await arm(adapter, 50, 500.0)
        await press_start(adapter, 51, 500.0)
        await press_start(adapter, 53, 500.1)
        assert adapter.status()["start_presses_recorded"] == 2

        # One X press abandons the two-thirds-complete cohort sequence.
        await press_button(adapter, X_MASK, 55, 500.2)
        assert adapter.status()["start_presses_recorded"] == 0
        assert adapter.status()["pending_button"] == "X"

        # So the next Start press is Start press one, not the launch.
        back = await press_start(adapter, 57, 500.3)
        assert back.disposition == "start_press_recorded"
        assert adapter.status()["start_presses_recorded"] == 1
        assert recorder.cohorts == []
        assert recorder.fruit_runs == []
        black_box.close()

    asyncio.run(scenario())


def test_two_activation_buttons_at_once_count_for_neither(tmp_path) -> None:
    async def scenario() -> None:
        black_box = RunBlackBox(tmp_path)
        adapter, recorder = make_idle_adapter(black_box)

        await arm(adapter, 60, 600.0)
        await press_button(adapter, X_MASK, 61, 600.0)
        await press_button(adapter, X_MASK, 63, 600.1)
        assert adapter.status()["pending_presses_recorded"] == 2

        # An X+Y chord is ambiguous, so it is other input: it clears the
        # sequence and counts for nothing.
        chord = await adapter.observe(sample(65, X_MASK | Y_MASK, 600.2))

        assert chord.disposition == "other_input"
        assert adapter.status()["pending_presses_recorded"] == 0
        assert adapter.status()["pending_button"] is None
        assert recorder.fruit_runs == []
        black_box.close()

    asyncio.run(scenario())


def test_a_stick_clears_a_partial_fruit_sequence(tmp_path) -> None:
    async def scenario() -> None:
        black_box = RunBlackBox(tmp_path)
        adapter, recorder = make_idle_adapter(black_box)

        await arm(adapter, 66, 660.0)
        await press_button(adapter, B_MASK, 67, 660.0)
        await press_button(adapter, B_MASK, 69, 660.1)
        assert adapter.status()["pending_presses_recorded"] == 2

        nudged = await adapter.observe(sample(71, 0, 660.2, {"ly": 0.9}))

        assert nudged.disposition == "other_input"
        assert adapter.status()["pending_presses_recorded"] == 0
        assert recorder.fruit_runs == []
        black_box.close()

    asyncio.run(scenario())


@fruit_button_params()
def test_fruit_presses_during_an_active_run_stop_it_and_never_start(
    tmp_path, button, mask, fruit
) -> None:
    """The dangerous case: X x3 during a run must stop, not stop-then-launch."""

    async def scenario() -> None:
        black_box = RunBlackBox(tmp_path)
        recorder = ActivationRecorder()
        stops: list[RemoteInput] = []

        async def stop_active(remote_input: RemoteInput) -> dict[str, object]:
            stops.append(remote_input)
            return {}

        adapter = ControllerStartAdapter(
            start_cohort=recorder.start_cohort,
            start_fruit_run=recorder.start_fruit_run,
            black_box=black_box,
            active_run_id=lambda: "run-1",
            stop_active=stop_active,
            # Busy throughout and deliberately never latched: the worst case,
            # because the takeover latch is not there to block a second
            # activation. Only the press gating can prevent a launch here.
            is_busy=lambda: True,
            takeover_latched=lambda: False,
        )

        await arm(adapter, 70, 700.0)
        decisions = [
            await press_button(adapter, mask, 71 + index * 2, 700.1 + index * 0.1)
            for index in range(START_PRESS_COUNT)
        ]

        assert [decision.disposition for decision in decisions] == [
            "takeover_stop"
        ] * START_PRESS_COUNT
        assert [stop.control for stop in stops] == [button] * START_PRESS_COUNT
        assert recorder.fruit_runs == []
        assert recorder.cohorts == []
        assert adapter.status()["pending_presses_recorded"] == 0
        assert adapter.status()["pending_button"] is None
        assert adapter.status()["accepted_edges"] == 0
        assert adapter.status()["takeover"]["stops"] == START_PRESS_COUNT
        black_box.close()

    asyncio.run(scenario())


def test_three_start_presses_still_start_the_cohort_with_fruit_buttons_wired(
    tmp_path,
) -> None:
    async def scenario() -> None:
        black_box = RunBlackBox(tmp_path)
        adapter, recorder = make_idle_adapter(black_box)

        await arm(adapter, 80, 800.0)
        launched = await press_button_times(adapter, START_MASK, 81, 800.1)

        assert launched is not None
        assert launched.disposition == "accepted"
        assert launched.cohort_id == recorder.cohorts[0]
        assert launched.activation_id == f"go2-controller-start:{TOPIC}:85"
        # A cohort is not a single-fruit run.
        assert launched.run_id is None
        assert launched.target_fruit is None
        assert recorder.fruit_runs == []
        assert adapter.status()["last_target_fruit"] is None
        black_box.close()

    asyncio.run(scenario())


def test_a_missing_fruit_run_path_is_reported_rather_than_silently_ignored(
    tmp_path,
) -> None:
    async def scenario() -> None:
        black_box = RunBlackBox(tmp_path)
        recorder = ActivationRecorder()
        adapter = ControllerStartAdapter(
            start_cohort=recorder.start_cohort,
            black_box=black_box,
            active_run_id=lambda: None,
        )

        assert adapter.status()["fruit_run_path_connected"] is False
        await arm(adapter, 90, 900.0)
        refused = await press_button_times(adapter, X_MASK, 91, 900.1)

        assert refused is not None
        assert refused.disposition == "run_unavailable"
        assert recorder.fruit_runs == []
        status = adapter.status()
        assert status["accepted_edges"] == 0
        assert "not connected" in str(status["last_error"])
        black_box.close()

    asyncio.run(scenario())


def test_the_status_detail_names_the_fruit_a_partial_sequence_would_run(
    tmp_path,
) -> None:
    async def scenario() -> None:
        black_box = RunBlackBox(tmp_path)
        adapter, _ = make_idle_adapter(black_box)

        await arm(adapter, 95, 950.0)
        await press_button(adapter, Y_MASK, 96, 950.1)

        detail = str(adapter.status()["detail"])
        assert "Y" in detail
        assert "Mango" in detail
        assert "2 more time" in detail
        black_box.close()

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# Per-fruit triggers: whole-application behaviour
# ---------------------------------------------------------------------------


def wait_for_run(client: TestClient, run_id: str, timeout_s: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        response = client.get(f"/api/results/{run_id}")
        if response.status_code == 200 and response.json()["run"].get("outcome"):
            return response.json()["run"]
        time.sleep(0.01)
    raise AssertionError("Demo Run did not become terminal")


@fruit_button_params()
def test_three_face_button_presses_run_exactly_that_fruit(
    tmp_path, button, mask, fruit
) -> None:
    source = DrivableSource()
    app = create_app(
        runs_root=tmp_path / "runs",
        cohorts_root=tmp_path / "cohorts",
        hardware=ReadyHardware(),
        camera_perception_status=ready_camera,
        stage_executor=SimulatedStageExecutor(),
        controller_start_source=source,
        inter_run_pause_s=0.0,
    )

    with TestClient(app) as client:
        launched = source.launch_button(mask, 500, 5000.0)

        assert launched.disposition == "accepted"
        assert launched.target_fruit == fruit
        assert launched.run_id is not None
        # A face button starts one Demo Run, never a cohort.
        assert launched.cohort_id is None
        assert client.get("/api/cohorts/active").json()["cohort"] is None

        run = wait_for_run(client, launched.run_id)
        assert run["outcome"] == "COMPLETED"
        assert run["target_fruit"] == fruit
        assert run["activation_source"] == CONTROLLER_ACTIVATION_SOURCE
        assert run["final_safety_state"] == "DISARMED_CONFIRMED"
        assert run["run_tuning"]["search"]["yaw_rps"] == CONTROLLER_START_SEARCH_YAW_RPS

        controller = client.get("/api/status").json()["controller_start"]
        assert controller["fruit_run_buttons"][button] == fruit
        assert controller["run_activation_source"] == CONTROLLER_ACTIVATION_SOURCE
        assert controller["adapter"]["accepted_edges"] == 1
        assert controller["adapter"]["last_target_fruit"] == fruit
        assert controller["adapter"]["last_run_id"] == launched.run_id
        assert controller["adapter"]["fruit_run_path_connected"] is True

    assert source.closed is True


def test_two_face_button_presses_alone_never_start_a_run(tmp_path) -> None:
    source = ScriptedSource(button_edges(90, X_MASK, presses=2))
    app = create_app(
        runs_root=tmp_path / "runs",
        cohorts_root=tmp_path / "cohorts",
        hardware=ReadyHardware(),
        camera_perception_status=ready_camera,
        stage_executor=SimulatedStageExecutor(),
        controller_start_source=source,
        inter_run_pause_s=0.0,
    )

    with TestClient(app) as client:
        assert "accepted" not in source.dispositions()
        status = client.get("/api/status").json()
        assert status["active_run_id"] is None
        assert status["cohort"] is None
        adapter = status["controller_start"]["adapter"]
        assert adapter["accepted_edges"] == 0
        assert adapter["pending_press_progress"] == "2 of 3"


def test_a_face_button_during_an_active_run_stops_it_and_starts_nothing(
    tmp_path,
) -> None:
    source = DrivableSource()
    executor = GatedStageExecutor()
    app = create_app(
        runs_root=tmp_path / "runs",
        cohorts_root=tmp_path / "cohorts",
        hardware=ReadyHardware(),
        camera_perception_status=ready_camera,
        stage_executor=executor,
        controller_start_source=source,
        inter_run_pause_s=0.0,
    )

    with TestClient(app) as client:
        assert source.submit(600, 0, 6000.0).disposition == "armed"
        started = client.post("/api/run", json={"target_fruit": "mango"})
        assert started.status_code == 201
        run_id = started.json()["run"]["run_id"]
        wait_until(lambda: executor.gate_reached)

        # Three X presses during the run. The first stops it; none of the three
        # may leave a Pear run behind.
        first = source.press(X_MASK, 601, 6000.1)
        second = source.press(X_MASK, 603, 6000.2)
        third = source.press(X_MASK, 605, 6000.3)

        assert first.disposition == "takeover_stop"
        # Presses two and three land on the takeover latch, not on a counter.
        assert {second.disposition, third.disposition} == {"takeover_latched"}

        status = client.get("/api/status").json()
        assert status["active_run_id"] is None
        assert client.get("/api/cohorts/active").json()["cohort"] is None
        run = client.get(f"/api/results/{run_id}").json()["run"]
        assert run["outcome"] == "STOPPED"
        assert run["target_fruit"] == "mango"

        adapter = status["controller_start"]["adapter"]
        assert adapter["accepted_edges"] == 0
        assert adapter["pending_presses_recorded"] == 0
        assert adapter["start_presses_recorded"] == 0
        assert adapter["takeover"]["latched"] is True
        assert adapter["takeover"]["stops"] == 1


def test_a_face_button_during_a_start_cohort_stops_the_cohort(tmp_path) -> None:
    source = DrivableSource()
    executor = GatedStageExecutor()
    app = create_app(
        runs_root=tmp_path / "runs",
        cohorts_root=tmp_path / "cohorts",
        hardware=ReadyHardware(),
        camera_perception_status=ready_camera,
        stage_executor=executor,
        controller_start_source=source,
        inter_run_pause_s=0.0,
    )

    with TestClient(app) as client:
        assert source.launch(700, 7000.0).disposition == "accepted"
        wait_until(lambda: executor.gate_reached)

        stopped = source.press(B_MASK, 720, 7010.0)

        assert stopped.disposition == "takeover_stop"
        cohort = client.get("/api/cohorts/active").json()["cohort"]
        assert cohort["status"] == "STOPPED"
        status = client.get("/api/status").json()
        assert status["active_run_id"] is None
        adapter = status["controller_start"]["adapter"]
        # B stopped the cohort and did not queue an Apple run.
        assert adapter["accepted_edges"] == 1
        assert adapter["pending_presses_recorded"] == 0
        assert adapter["last_target_fruit"] is None


def test_face_button_edges_while_remote_takeover_is_latched_are_refused(
    tmp_path,
) -> None:
    machine = MissionMachine()
    machine.remote_takeover(
        RemoteInput(
            source="go2_controller",
            control="wireless_remote",
            received_monotonic_s=1.0,
        )
    )
    source = ScriptedSource(button_edges(80, Y_MASK))
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
        assert "accepted" not in source.dispositions()
        assert set(source.dispositions()[1:]) == {"takeover_latched"}
        status = client.get("/api/status").json()
        assert status["active_run_id"] is None
        adapter = status["controller_start"]["adapter"]
        assert adapter["accepted_edges"] == 0
        assert adapter["takeover"]["latched"] is True


@fruit_button_params()
def test_a_stale_fruit_press_count_cannot_survive_a_stop(
    tmp_path, button, mask, fruit
) -> None:
    """The exact hazard: two presses, a stop, one press must not launch.

    Two X presses while idle, then a run starts from the UI, then a third X
    press stops it. If the stopping press left the count at two, the operator's
    next single X press would launch a Pear run -- three presses, but only one
    of them after the stop. The count must be gone.
    """

    async def scenario() -> None:
        black_box = RunBlackBox(tmp_path)
        recorder = ActivationRecorder()
        busy = {"value": False}
        stops: list[RemoteInput] = []

        async def stop_active(remote_input: RemoteInput) -> dict[str, object]:
            stops.append(remote_input)
            busy["value"] = False
            return {}

        adapter = ControllerStartAdapter(
            start_cohort=recorder.start_cohort,
            start_fruit_run=recorder.start_fruit_run,
            black_box=black_box,
            active_run_id=lambda: "run-1" if busy["value"] else None,
            stop_active=stop_active,
            is_busy=lambda: busy["value"],
            # Never latched, so nothing but the press gating stands between the
            # stop and an unwanted launch.
            takeover_latched=lambda: False,
        )

        await arm(adapter, 120, 1200.0)
        await press_button(adapter, mask, 121, 1200.1)
        await press_button(adapter, mask, 123, 1200.2)
        assert adapter.status()["pending_presses_recorded"] == 2

        # The UI starts a run, then the operator presses the same button.
        busy["value"] = True
        stopped = await press_button(adapter, mask, 125, 1200.3)

        assert stopped.disposition == "takeover_stop"
        assert [stop.control for stop in stops] == [button]
        assert adapter.status()["pending_presses_recorded"] == 0

        # One more press must be press one of a fresh sequence, not a launch.
        again = await press_button(adapter, mask, 127, 1200.4)
        assert again.disposition == "start_press_recorded"
        assert adapter.status()["pending_presses_recorded"] == 1
        assert recorder.fruit_runs == []
        assert recorder.cohorts == []

        # It takes two further presses -- three after the stop -- to launch.
        await press_button(adapter, mask, 129, 1200.5)
        launched = await press_button(adapter, mask, 131, 1200.6)
        assert launched.disposition == "accepted"
        assert launched.target_fruit == fruit
        assert recorder.fruits() == [fruit]
        black_box.close()

    asyncio.run(scenario())


def test_a_fruit_activation_id_cannot_collide_across_a_robot_reboot(tmp_path) -> None:
    """The DDS tick restarts at zero on reboot; the activation id must not.

    activation_id is StageDemo's durable idempotency key, matched against
    persisted Run Results. If a repeated tick produced a repeated id, the
    operator's press would be answered with the earlier run and the robot would
    never move -- a silent dead press on stage.
    """

    async def scenario() -> None:
        black_box = RunBlackBox(tmp_path)
        first_adapter, first = make_idle_adapter(black_box)
        second_adapter, second = make_idle_adapter(black_box)

        # The identical sample sequence, as if the robot had rebooted and the
        # tick counter came back around to the same values.
        for adapter in (first_adapter, second_adapter):
            await arm(adapter, 140, 1400.0)
            await press_button_times(adapter, X_MASK, 141, 1400.1)

        assert first.fruits() == ["pear"]
        assert second.fruits() == ["pear"]
        first_id = first.fruit_runs[0][1]
        second_id = second.fruit_runs[0][1]

        assert first_id != second_id
        assert first_adapter.instance_id != second_adapter.instance_id
        assert first_id.startswith("go2-controller-pear:")
        assert first_id.endswith(f":{TOPIC}:145")
        assert second_id.endswith(f":{TOPIC}:145")
        black_box.close()

    asyncio.run(scenario())
