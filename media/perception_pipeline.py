"""Latest-frame perception scheduling with explicit throughput evidence.

The pipeline owns cadence and ordering while the sidecar owns fruit semantics.
Motion evidence is published before optional visual-odometry and preview work in
the throughput profile.  The baseline profile preserves the former serialized
ordering for an in-build rollback and matched A/B measurements.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import dataclass
from enum import StrEnum
from threading import Lock
from typing import Any


class PipelineProfile(StrEnum):
    BASELINE = "baseline"
    THROUGHPUT_V1 = "throughput-v1"

    @classmethod
    def parse(cls, value: str) -> PipelineProfile:
        try:
            return cls(value.strip().casefold())
        except ValueError as exc:
            choices = ", ".join(profile.value for profile in cls)
            raise ValueError(
                f"unknown perception pipeline profile {value!r}; expected {choices}"
            ) from exc


class FrameRoute(StrEnum):
    BASELINE = "baseline"
    FULL_FRAME = "full_frame"
    LOWER_CENTER_SEARCH_CROP = "lower_center_search_crop"


@dataclass(frozen=True)
class SourceFrame:
    generation: str
    frame: Any
    received_monotonic_s: float
    pts: int
    time_base: str

    def __post_init__(self) -> None:
        if not self.generation:
            raise ValueError("source frame generation is required")
        if not isinstance(self.pts, int):
            raise TypeError("source frame PTS must be an integer")
        if not self.time_base:
            raise ValueError("source frame time base is required")
        if not math.isfinite(self.received_monotonic_s):
            raise ValueError("source frame receive time must be finite")


@dataclass(frozen=True)
class InferenceOutcome:
    source: SourceFrame
    target_fruit: str
    route: FrameRoute
    payload: Any
    candidate_present: bool
    route_triggered: bool
    inference_passes: int
    inference_s: float

    def __post_init__(self) -> None:
        if self.inference_passes < 1:
            raise ValueError("inference passes must be positive")
        if not math.isfinite(self.inference_s) or self.inference_s < 0.0:
            raise ValueError("inference duration must be finite and non-negative")


@dataclass
class _AuxiliaryMetrics:
    submitted: int = 0
    dropped: int = 0
    started: int = 0
    completed: int = 0
    errors: int = 0
    total_s: float = 0.0
    maximum_s: float = 0.0
    last_error: str | None = None

    def status(self) -> dict[str, object]:
        return {
            "submitted": self.submitted,
            "dropped": self.dropped,
            "started": self.started,
            "completed": self.completed,
            "errors": self.errors,
            "average_s": (
                None if self.completed == 0 else self.total_s / self.completed
            ),
            "maximum_s": self.maximum_s,
            "last_error": self.last_error,
        }


class PerceptionPipeline:
    """Process the newest source frame through one observable interface."""

    def __init__(
        self,
        *,
        profile: PipelineProfile,
        target_fruit: Callable[[], str],
        infer: Callable[[SourceFrame, FrameRoute], InferenceOutcome | None],
        publish: Callable[[InferenceOutcome], None],
        visual_odometry: Callable[[InferenceOutcome], None],
        preview: Callable[[InferenceOutcome], None],
        inference_executor: Executor | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.profile = profile
        self._target_fruit = target_fruit
        self._infer = infer
        self._publish = publish
        self._visual_odometry = visual_odometry
        self._preview = preview
        self._clock = clock
        self._source: asyncio.Queue[SourceFrame] = asyncio.Queue(maxsize=1)
        self._odometry: asyncio.Queue[InferenceOutcome] = asyncio.Queue(maxsize=1)
        self._previews: asyncio.Queue[InferenceOutcome] = asyncio.Queue(maxsize=1)
        self._inference_executor = inference_executor or ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="border-collie-inference",
        )
        self._owns_inference_executor = inference_executor is None
        self._odometry_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="border-collie-odometry",
        )
        self._preview_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="border-collie-preview",
        )
        self._tasks: list[asyncio.Task[None]] = []
        self._closed = False
        self._lock = Lock()
        self._started_s: float | None = None
        self._source_frames = 0
        self._dropped_source_frames = 0
        self._processed_frames = 0
        self._stale_frames = 0
        self._inference_passes = 0
        self._inference_total_s = 0.0
        self._inference_maximum_s = 0.0
        self._latest_published_pts: int | None = None
        self._routes: Counter[str] = Counter()
        self._route_generation: str | None = None
        self._next_crop_target: str | None = None
        self._odometry_metrics = _AuxiliaryMetrics()
        self._preview_metrics = _AuxiliaryMetrics()

    async def start(self) -> None:
        if self._closed:
            raise RuntimeError("perception pipeline is closed")
        if self._tasks:
            return
        with self._lock:
            self._started_s = self._clock()
        self._tasks.append(asyncio.create_task(self._run(), name="perception-pipeline"))
        if self.profile is PipelineProfile.THROUGHPUT_V1:
            self._tasks.extend(
                (
                    asyncio.create_task(
                        self._run_auxiliary(
                            self._odometry,
                            self._visual_odometry,
                            self._odometry_executor,
                            self._odometry_metrics,
                        ),
                        name="perception-odometry",
                    ),
                    asyncio.create_task(
                        self._run_auxiliary(
                            self._previews,
                            self._preview,
                            self._preview_executor,
                            self._preview_metrics,
                        ),
                        name="perception-preview",
                    ),
                )
            )

    def submit(self, source: SourceFrame) -> None:
        if not self._tasks or self._closed:
            raise RuntimeError("perception pipeline is not running")
        with self._lock:
            self._source_frames += 1
        if self._source.full():
            self._source.get_nowait()
            with self._lock:
                self._dropped_source_frames += 1
        self._source.put_nowait(source)

    async def wait_until_processed(self, count: int, *, timeout_s: float) -> None:
        if count < 0 or timeout_s <= 0.0:
            raise ValueError("processed count and timeout must be positive")
        deadline = asyncio.get_running_loop().time() + timeout_s
        while True:
            with self._lock:
                processed = self._processed_frames
            if processed >= count:
                return
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError(
                    f"perception pipeline processed {processed}/{count} frames"
                )
            await asyncio.sleep(0.001)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        tasks, self._tasks = self._tasks, []
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        executors = [self._odometry_executor, self._preview_executor]
        if self._owns_inference_executor:
            executors.append(self._inference_executor)
        for executor in executors:
            await asyncio.to_thread(executor.shutdown, wait=True, cancel_futures=True)

    def status(self) -> dict[str, object]:
        with self._lock:
            now = self._clock()
            elapsed_s = (
                None
                if self._started_s is None
                else max(0.0, now - self._started_s)
            )
            processed_fps = (
                None
                if elapsed_s is None or elapsed_s == 0.0
                else self._processed_frames / elapsed_s
            )
            source_fps = (
                None
                if elapsed_s is None or elapsed_s == 0.0
                else self._source_frames / elapsed_s
            )
            return {
                "profile": self.profile.value,
                "source_frames": self._source_frames,
                "processed_frames": self._processed_frames,
                "dropped_source_frames": self._dropped_source_frames,
                "stale_frames": self._stale_frames,
                "source_fps": source_fps,
                "processed_fps": processed_fps,
                "inference_passes": self._inference_passes,
                "average_inference_s": (
                    None
                    if self._processed_frames == 0
                    else self._inference_total_s / self._processed_frames
                ),
                "maximum_inference_s": self._inference_maximum_s,
                "latest_published_pts": self._latest_published_pts,
                "routes": dict(self._routes),
                "visual_odometry": self._odometry_metrics.status(),
                "preview": self._preview_metrics.status(),
            }

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            source = await self._source.get()
            target = self._target_fruit().casefold().strip()
            route = self._route_for(target, source.generation)
            outcome = await loop.run_in_executor(
                self._inference_executor,
                self._infer,
                source,
                route,
            )
            if outcome is None:
                with self._lock:
                    self._stale_frames += 1
                continue
            if outcome.source != source:
                raise RuntimeError("inference outcome does not match its source frame")
            self._note_inference(outcome)
            self._advance_route(target, outcome)
            if self.profile is PipelineProfile.BASELINE:
                await self._run_baseline_auxiliary(outcome)
                self._publish(outcome)
                with self._lock:
                    self._latest_published_pts = source.pts
                    self._preview_metrics.submitted += 1
                    self._preview_metrics.started += 1
                preview_started = self._clock()
                try:
                    await loop.run_in_executor(
                        self._inference_executor,
                        self._preview,
                        outcome,
                    )
                except Exception as exc:  # noqa: BLE001 - preview must degrade only
                    self._note_auxiliary_error(self._preview_metrics, exc)
                else:
                    self._note_auxiliary_complete(
                        self._preview_metrics,
                        preview_started,
                    )
                continue
            self._publish(outcome)
            with self._lock:
                self._latest_published_pts = source.pts
            self._enqueue_auxiliary(
                self._odometry,
                outcome,
                self._odometry_metrics,
            )
            self._enqueue_auxiliary(
                self._previews,
                outcome,
                self._preview_metrics,
            )

    async def _run_baseline_auxiliary(self, outcome: InferenceOutcome) -> None:
        loop = asyncio.get_running_loop()
        started = self._clock()
        with self._lock:
            self._odometry_metrics.submitted += 1
            self._odometry_metrics.started += 1
        try:
            await loop.run_in_executor(
                self._inference_executor,
                self._visual_odometry,
                outcome,
            )
        except Exception as exc:  # noqa: BLE001 - odometry must degrade only
            self._note_auxiliary_error(self._odometry_metrics, exc)
        else:
            self._note_auxiliary_complete(self._odometry_metrics, started)

    def _route_for(self, target: str, generation: str) -> FrameRoute:
        if self.profile is PipelineProfile.BASELINE:
            return FrameRoute.BASELINE
        if generation != self._route_generation:
            self._route_generation = generation
            self._next_crop_target = None
        if self._next_crop_target == target:
            return FrameRoute.LOWER_CENTER_SEARCH_CROP
        return FrameRoute.FULL_FRAME

    def _advance_route(self, target: str, outcome: InferenceOutcome) -> None:
        if self.profile is PipelineProfile.BASELINE:
            return
        if outcome.route is FrameRoute.LOWER_CENTER_SEARCH_CROP:
            self._next_crop_target = target if outcome.candidate_present else None
            return
        self._next_crop_target = (
            target
            if target in {"apple", "banana"}
            and not outcome.candidate_present
            and not outcome.route_triggered
            else None
        )

    def _note_inference(self, outcome: InferenceOutcome) -> None:
        with self._lock:
            self._processed_frames += 1
            self._inference_passes += outcome.inference_passes
            self._inference_total_s += outcome.inference_s
            self._inference_maximum_s = max(
                self._inference_maximum_s,
                outcome.inference_s,
            )
            self._routes[outcome.route.value] += 1

    def _enqueue_auxiliary(
        self,
        queue: asyncio.Queue[InferenceOutcome],
        outcome: InferenceOutcome,
        metrics: _AuxiliaryMetrics,
    ) -> None:
        with self._lock:
            metrics.submitted += 1
        if queue.full():
            queue.get_nowait()
            with self._lock:
                metrics.dropped += 1
        queue.put_nowait(outcome)

    async def _run_auxiliary(
        self,
        queue: asyncio.Queue[InferenceOutcome],
        function: Callable[[InferenceOutcome], None],
        executor: Executor,
        metrics: _AuxiliaryMetrics,
    ) -> None:
        loop = asyncio.get_running_loop()
        while True:
            outcome = await queue.get()
            started = self._clock()
            with self._lock:
                metrics.started += 1
            try:
                await loop.run_in_executor(executor, function, outcome)
            except Exception as exc:  # noqa: BLE001 - auxiliary work must degrade
                self._note_auxiliary_error(metrics, exc)
            else:
                self._note_auxiliary_complete(metrics, started)

    def _note_auxiliary_complete(
        self,
        metrics: _AuxiliaryMetrics,
        started_s: float,
    ) -> None:
        elapsed_s = max(0.0, self._clock() - started_s)
        with self._lock:
            metrics.completed += 1
            metrics.total_s += elapsed_s
            metrics.maximum_s = max(metrics.maximum_s, elapsed_s)
            metrics.last_error = None

    def _note_auxiliary_error(
        self,
        metrics: _AuxiliaryMetrics,
        exc: Exception,
    ) -> None:
        with self._lock:
            metrics.errors += 1
            metrics.last_error = f"{type(exc).__name__}: {exc}"
