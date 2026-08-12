import asyncio
from threading import Event

from media.perception_pipeline import (
    FrameRoute,
    InferenceOutcome,
    PerceptionPipeline,
    PipelineProfile,
    SourceFrame,
)


def frame(pts: int) -> SourceFrame:
    return SourceFrame(
        generation="camera-1",
        frame=f"frame-{pts}",
        received_monotonic_s=float(pts),
        pts=pts,
        time_base="1/90000",
    )


def outcome(source: SourceFrame, route: FrameRoute) -> InferenceOutcome:
    return InferenceOutcome(
        source=source,
        target_fruit="apple",
        route=route,
        payload=f"result-{source.pts}",
        candidate_present=False,
        route_triggered=False,
        inference_passes=1,
        inference_s=0.060,
    )


def test_latest_frame_pipeline_replaces_pending_source_without_stale_backlog() -> None:
    started = Event()
    release = Event()
    published: list[int] = []

    def infer(source: SourceFrame, route: FrameRoute) -> InferenceOutcome:
        if source.pts == 1:
            started.set()
            assert release.wait(timeout=1.0)
        return outcome(source, route)

    async def scenario() -> dict[str, object]:
        pipeline = PerceptionPipeline(
            profile=PipelineProfile.THROUGHPUT_V1,
            target_fruit=lambda: "apple",
            infer=infer,
            publish=lambda result: published.append(result.source.pts),
            visual_odometry=lambda _result: None,
            preview=lambda _result: None,
        )
        await pipeline.start()
        try:
            pipeline.submit(frame(1))
            assert await asyncio.to_thread(started.wait, 1.0)
            pipeline.submit(frame(2))
            pipeline.submit(frame(3))
            release.set()
            await pipeline.wait_until_processed(2, timeout_s=1.0)
            return pipeline.status()
        finally:
            await pipeline.close()

    status = asyncio.run(scenario())

    assert published == [1, 3]
    assert status["source_frames"] == 3
    assert status["processed_frames"] == 2
    assert status["dropped_source_frames"] == 1
    assert status["latest_published_pts"] == 3


def test_motion_evidence_is_published_before_slow_auxiliary_work_completes() -> None:
    published = Event()
    odometry_started = Event()
    release_odometry = Event()

    def observe_odometry(_result: InferenceOutcome) -> None:
        odometry_started.set()
        assert release_odometry.wait(timeout=1.0)

    async def scenario() -> dict[str, object]:
        pipeline = PerceptionPipeline(
            profile=PipelineProfile.THROUGHPUT_V1,
            target_fruit=lambda: "pear",
            infer=lambda source, route: outcome(source, route),
            publish=lambda _result: published.set(),
            visual_odometry=observe_odometry,
            preview=lambda _result: None,
        )
        await pipeline.start()
        try:
            pipeline.submit(frame(10))
            assert await asyncio.to_thread(published.wait, 1.0)
            assert await asyncio.to_thread(odometry_started.wait, 1.0)
            assert not release_odometry.is_set()
            status = pipeline.status()
            assert status["processed_frames"] == 1
            assert status["latest_published_pts"] == 10
            return status
        finally:
            release_odometry.set()
            await pipeline.close()

    status = asyncio.run(scenario())
    assert status["visual_odometry"]["completed"] == 0


def test_throughput_profile_staggers_full_and_search_crop_passes() -> None:
    routes: list[FrameRoute] = []

    def infer(source: SourceFrame, route: FrameRoute) -> InferenceOutcome:
        routes.append(route)
        return outcome(source, route)

    async def scenario() -> dict[str, object]:
        pipeline = PerceptionPipeline(
            profile=PipelineProfile.THROUGHPUT_V1,
            target_fruit=lambda: "apple",
            infer=infer,
            publish=lambda _result: None,
            visual_odometry=lambda _result: None,
            preview=lambda _result: None,
        )
        await pipeline.start()
        try:
            for pts in (20, 21, 22):
                pipeline.submit(frame(pts))
                await pipeline.wait_until_processed(pts - 19, timeout_s=1.0)
            return pipeline.status()
        finally:
            await pipeline.close()

    status = asyncio.run(scenario())

    assert routes == [
        FrameRoute.FULL_FRAME,
        FrameRoute.LOWER_CENTER_SEARCH_CROP,
        FrameRoute.FULL_FRAME,
    ]
    assert status["inference_passes"] == 3
    assert status["routes"] == {
        "full_frame": 2,
        "lower_center_search_crop": 1,
    }


def test_crop_candidate_keeps_fresh_crop_route_until_it_is_lost() -> None:
    routes: list[FrameRoute] = []

    def infer(source: SourceFrame, route: FrameRoute) -> InferenceOutcome:
        routes.append(route)
        result = outcome(source, route)
        if source.pts in {41, 42}:
            return InferenceOutcome(
                source=result.source,
                target_fruit=result.target_fruit,
                route=result.route,
                payload=result.payload,
                candidate_present=True,
                route_triggered=False,
                inference_passes=1,
                inference_s=0.060,
            )
        return result

    async def scenario() -> None:
        pipeline = PerceptionPipeline(
            profile=PipelineProfile.THROUGHPUT_V1,
            target_fruit=lambda: "apple",
            infer=infer,
            publish=lambda _result: None,
            visual_odometry=lambda _result: None,
            preview=lambda _result: None,
        )
        await pipeline.start()
        try:
            for index, pts in enumerate((40, 41, 42, 43), start=1):
                pipeline.submit(frame(pts))
                await pipeline.wait_until_processed(index, timeout_s=1.0)
        finally:
            await pipeline.close()

    asyncio.run(scenario())
    assert routes == [
        FrameRoute.FULL_FRAME,
        FrameRoute.LOWER_CENTER_SEARCH_CROP,
        FrameRoute.LOWER_CENTER_SEARCH_CROP,
        FrameRoute.LOWER_CENTER_SEARCH_CROP,
    ]


def test_camera_generation_change_resets_latched_crop_route() -> None:
    routes: list[FrameRoute] = []

    def infer(source: SourceFrame, route: FrameRoute) -> InferenceOutcome:
        routes.append(route)
        result = outcome(source, route)
        return InferenceOutcome(
            source=result.source,
            target_fruit=result.target_fruit,
            route=result.route,
            payload=result.payload,
            candidate_present=source.pts == 51,
            route_triggered=False,
            inference_passes=1,
            inference_s=0.060,
        )

    async def scenario() -> None:
        pipeline = PerceptionPipeline(
            profile=PipelineProfile.THROUGHPUT_V1,
            target_fruit=lambda: "apple",
            infer=infer,
            publish=lambda _result: None,
            visual_odometry=lambda _result: None,
            preview=lambda _result: None,
        )
        await pipeline.start()
        try:
            for index, source in enumerate(
                (
                    frame(50),
                    frame(51),
                    SourceFrame(
                        generation="camera-2",
                        frame="frame-52",
                        received_monotonic_s=52.0,
                        pts=52,
                        time_base="1/90000",
                    ),
                ),
                start=1,
            ):
                pipeline.submit(source)
                await pipeline.wait_until_processed(index, timeout_s=1.0)
        finally:
            await pipeline.close()

    asyncio.run(scenario())
    assert routes == [
        FrameRoute.FULL_FRAME,
        FrameRoute.LOWER_CENTER_SEARCH_CROP,
        FrameRoute.FULL_FRAME,
    ]


def test_baseline_profile_preserves_same_frame_followup_route() -> None:
    routes: list[FrameRoute] = []

    async def scenario() -> dict[str, object]:
        pipeline = PerceptionPipeline(
            profile=PipelineProfile.BASELINE,
            target_fruit=lambda: "apple",
            infer=lambda source, route: (
                routes.append(route) or outcome(source, route)
            ),
            publish=lambda _result: None,
            visual_odometry=lambda _result: None,
            preview=lambda _result: None,
        )
        await pipeline.start()
        try:
            pipeline.submit(frame(30))
            await pipeline.wait_until_processed(1, timeout_s=1.0)
            deadline = asyncio.get_running_loop().time() + 0.1
            while pipeline.status()["preview"]["completed"] < 1:
                if asyncio.get_running_loop().time() >= deadline:
                    break
                await asyncio.sleep(0.001)
            return pipeline.status()
        finally:
            await pipeline.close()

    status = asyncio.run(scenario())
    assert routes == [FrameRoute.BASELINE]
    assert status["visual_odometry"]["completed"] == 1
    assert status["preview"]["completed"] == 1
