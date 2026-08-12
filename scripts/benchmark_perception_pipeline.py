"""Synthetic matched-load benchmark for perception scheduling profiles."""

from __future__ import annotations

import argparse
import asyncio
import json
import time

from media.perception_pipeline import (
    FrameRoute,
    InferenceOutcome,
    PerceptionPipeline,
    PipelineProfile,
    SourceFrame,
)


async def benchmark(
    profile: PipelineProfile,
    *,
    duration_s: float,
    source_fps: float,
    inference_pass_ms: float,
    odometry_ms: float,
    preview_ms: float,
) -> dict[str, object]:
    published = 0

    def infer(source: SourceFrame, route: FrameRoute) -> InferenceOutcome:
        passes = 2 if route is FrameRoute.BASELINE else 1
        elapsed_s = passes * inference_pass_ms / 1000.0
        time.sleep(elapsed_s)
        return InferenceOutcome(
            source=source,
            target_fruit="apple",
            route=route,
            payload=None,
            candidate_present=False,
            route_triggered=False,
            inference_passes=passes,
            inference_s=elapsed_s,
        )

    def publish(_result: InferenceOutcome) -> None:
        nonlocal published
        published += 1

    pipeline = PerceptionPipeline(
        profile=profile,
        target_fruit=lambda: "apple",
        infer=infer,
        publish=publish,
        visual_odometry=lambda _result: time.sleep(odometry_ms / 1000.0),
        preview=lambda _result: time.sleep(preview_ms / 1000.0),
    )
    await pipeline.start()
    started = time.monotonic()
    interval_s = 1.0 / source_fps
    pts = 0
    try:
        while time.monotonic() - started < duration_s:
            now = time.monotonic()
            pipeline.submit(
                SourceFrame(
                    generation="benchmark",
                    frame=None,
                    received_monotonic_s=now,
                    pts=pts,
                    time_base="1/90000",
                )
            )
            pts += 1
            await asyncio.sleep(interval_s)
        await asyncio.sleep(
            max(0.25, (2 * inference_pass_ms + odometry_ms + preview_ms) / 1000.0)
        )
        status = pipeline.status()
    finally:
        await pipeline.close()
    return {
        "profile": profile.value,
        "source_frames": pts,
        "published_frames": published,
        "published_fps": published / duration_s,
        "dropped_source_frames": status["dropped_source_frames"],
        "inference_passes": status["inference_passes"],
        "routes": status["routes"],
        "visual_odometry": status["visual_odometry"],
        "preview": status["preview"],
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration-s", type=float, default=3.0)
    parser.add_argument("--source-fps", type=float, default=15.0)
    parser.add_argument("--inference-pass-ms", type=float, default=62.5)
    parser.add_argument("--odometry-ms", type=float, default=21.0)
    parser.add_argument("--preview-ms", type=float, default=25.0)
    args = parser.parse_args()
    results = []
    for profile in (PipelineProfile.BASELINE, PipelineProfile.THROUGHPUT_V1):
        results.append(
            await benchmark(
                profile,
                duration_s=args.duration_s,
                source_fps=args.source_fps,
                inference_pass_ms=args.inference_pass_ms,
                odometry_ms=args.odometry_ms,
                preview_ms=args.preview_ms,
            )
        )
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(main())
