"""Back-to-back Demo Runs must each execute with the detector awake.

Five runs ran back to back on 2026-08-17 19:22:56-19:25:19 and the last two
swept a full circle producing zero detections of any confidence while the camera
source was demonstrably healthy. The suspicion was the idle inference gate: an
earlier run's lease release racing the next run's acquire, a lease expiring
mid-run, or a swallowed acquire failure leaving the detector asleep for a whole
run.

These tests replay that sequence across both real seams - the app-side
``PerceptionStatusClient`` and the sidecar's HTTP surface - and pin the
behaviour the gate actually has, including the one hazard that is currently
unreachable only because nothing calls ``release``.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from border_collie_demo.config import PerceptionConfig
from border_collie_demo.perception import PerceptionStatusClient
from media.perception_sidecar import InferenceGate, create_app

RUN_HOLD_S = 120.0
GUIDANCE_TICK_S = 0.107  # measured sample cadence in the 19:24:59 search trace
SWEEP_SAMPLES = 97  # the longest of the two zero-detection sweeps


class FakeClock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class GateRuntime:
    """Minimal sidecar runtime exposing only the gate surface."""

    def __init__(self, clock: FakeClock) -> None:
        self.gate = InferenceGate(warmup_s=0.0, clock=clock)
        self.target_fruit = "mango"
        # Every frame the detector would have processed, so a test can assert
        # inference actually ran rather than only that the gate said so.
        self.inference_frames = 0
        self.idle_frames = 0

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

    def consume_frame(self) -> None:
        """What ``_detect`` does per frame: gate decides which branch runs."""
        if self.gate.active():
            self.inference_frames += 1
        else:
            self.idle_frames += 1


def client_against(sidecar: TestClient) -> PerceptionStatusClient:
    """An app-side perception client whose transports hit the real sidecar app."""

    def fetch(url: str, _timeout: float) -> dict[str, object]:
        response = sidecar.get(url)
        response.raise_for_status()
        return response.json()

    def post_target(url: str, fruit: str, _timeout: float) -> dict[str, object]:
        response = sidecar.post(url, json={"target_fruit": fruit, "hold": "preview"})
        response.raise_for_status()
        return response.json()

    def post_json(
        url: str, body: dict[str, object], _timeout: float
    ) -> dict[str, object]:
        response = sidecar.post(url, json=body)
        response.raise_for_status()
        return response.json()

    return PerceptionStatusClient(
        PerceptionConfig(enabled=True),
        fetcher=fetch,
        target_poster=post_target,
        json_poster=post_json,
    )


def test_five_back_to_back_runs_never_execute_with_the_detector_asleep() -> None:
    """Replay 2026-08-17 19:22:56-19:25:19: activate, sweep, end, repeat."""
    clock = FakeClock()
    runtime = GateRuntime(clock)

    with TestClient(create_app(runtime)) as sidecar:
        perception = client_against(sidecar)
        asleep_samples: list[tuple[int, int]] = []

        for run in range(1, 6):
            # Activation: select the target and take the run lease, exactly as
            # StageDemo._select_target_and_read_camera does.
            activation = perception.select_run_target("mango")
            assert activation["inference"]["holds"]["run"] == pytest.approx(RUN_HOLD_S)

            # The sweep. Guidance polls perception every tick through
            # run_status, which renews the lease on the way past.
            for sample in range(1, SWEEP_SAMPLES + 1):
                status = perception.run_status()
                runtime.consume_frame()
                if status["inference"].get("active") is not True:
                    asleep_samples.append((run, sample))
                clock.advance(GUIDANCE_TICK_S)

            # The run ends. Nothing releases the lease - no caller of
            # release_inference exists - and 20 s later the next run starts,
            # which is the observed gap between the two failing runs.
            clock.advance(20.0)

        assert asleep_samples == []
        assert runtime.inference_frames == 5 * SWEEP_SAMPLES
        assert runtime.idle_frames == 0


def test_a_finished_run_cannot_strand_the_next_run_without_a_lease() -> None:
    """Even an explicit release from run N cannot outlive run N+1's acquire."""
    clock = FakeClock()
    gate = InferenceGate(warmup_s=0.0, clock=clock)

    gate.hold("run", RUN_HOLD_S)  # run N activates
    gate.release("run")  # run N ends and releases
    gate.hold("run", RUN_HOLD_S)  # run N+1 activates immediately

    assert gate.active() is True
    assert gate.status()["holds"]["run"] == pytest.approx(RUN_HOLD_S)


def test_a_late_release_from_a_finished_run_would_strand_the_next_one() -> None:
    """The latent hazard, unreachable today because nothing calls release.

    Deadlines are absolute and ``release`` is unconditional, so a release that
    arrives after the next run's acquire drops a live lease. If a run-teardown
    release is ever added, it must carry the lease it is dropping.
    """
    clock = FakeClock()
    gate = InferenceGate(warmup_s=0.0, clock=clock)

    gate.hold("run", RUN_HOLD_S)  # run N+1 activates
    gate.release("run")  # run N's teardown arrives late

    assert gate.active() is False


def test_a_long_wait_for_command_lets_the_run_lease_lapse_before_the_sweep() -> None:
    """The only way a run reaches guidance unleased: nothing polls in between.

    Activation takes the lease, but a voice run then sits in wait_for_command,
    which does not poll perception. Past 120 s the lease lapses and the
    detector sleeps until guidance's first poll re-takes it - so the first tick
    or two of the sweep can be cold, which is the gap the trace should show.
    """
    clock = FakeClock()
    runtime = GateRuntime(clock)

    with TestClient(create_app(runtime)) as sidecar:
        perception = client_against(sidecar)
        perception.select_run_target("mango")

        clock.advance(RUN_HOLD_S + 10.0)  # operator takes their time
        runtime.consume_frame()
        assert runtime.idle_frames == 1  # the detector is asleep at this instant

        first_tick = perception.run_status()
        assert first_tick["inference"]["active"] is True
        runtime.consume_frame()
        assert runtime.inference_frames == 1
