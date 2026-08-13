from __future__ import annotations

import asyncio

import pytest

from border_collie_demo.media import BarkFailure
from border_collie_demo.system_audio import (
    SystemAudioConfig,
    SystemAudioPolicy,
    create_system_audio_policy,
)


class FakeVui:
    def __init__(self, *, volume: int = 7) -> None:
        self.volume = volume
        self.events: list[tuple[object, ...]] = []

    def SetTimeout(self, timeout_s: float) -> None:
        self.events.append(("timeout", timeout_s))

    def Init(self) -> None:
        self.events.append(("init",))

    def GetVolume(self) -> tuple[int, int]:
        self.events.append(("get", self.volume))
        return 0, self.volume

    def SetVolume(self, volume: int) -> int:
        self.events.append(("set", volume))
        self.volume = volume
        return 0


class StuckMutedVui(FakeVui):
    def SetVolume(self, volume: int) -> int:
        self.events.append(("set", volume))
        if volume == 0:
            self.volume = 0
        return 0


class FakeBark:
    def __init__(self, events: list[tuple[object, ...]], *, fail: bool = False) -> None:
        self.events = events
        self.fail = fail

    def status(self) -> dict[str, object]:
        return {"ready": True, "detail": "bark ready"}

    async def bark(self) -> dict[str, object]:
        self.events.append(("bark",))
        if self.fail:
            raise BarkFailure("speaker request failed")
        return {"bark_played": True, "bark_uuid": "woof"}


def test_system_audio_defers_vui_construction_until_lifecycle_start() -> None:
    created: list[FakeVui] = []

    def create_vui() -> FakeVui:
        vui = FakeVui(volume=7)
        created.append(vui)
        return vui

    policy = create_system_audio_policy(
        FakeBark([]),
        SystemAudioConfig(),
        vui_factory=create_vui,
    )

    assert created == []

    asyncio.run(policy.start_muted())

    assert len(created) == 1
    assert created[0].volume == 0


def test_system_audio_stays_muted_except_for_bounded_bark_window() -> None:
    async def scenario() -> None:
        vui = FakeVui(volume=7)
        events = vui.events

        async def sleep(duration_s: float) -> None:
            events.append(("sleep", duration_s))

        policy = SystemAudioPolicy(
            vui,
            FakeBark(events),
            SystemAudioConfig(bark_volume=6, bark_audible_s=1.5),
            sleep=sleep,
        )

        await policy.start_muted()
        result = await policy.bark()

        assert events == [
            ("timeout", 3.0),
            ("init",),
            ("get", 7),
            ("set", 0),
            ("get", 0),
            ("set", 6),
            ("get", 6),
            ("bark",),
            ("sleep", 1.5),
            ("set", 0),
            ("get", 0),
        ]
        assert result == {
            "bark_played": True,
            "bark_uuid": "woof",
            "speaker_policy": "muted_except_bark",
            "bark_volume": 6,
            "bark_audible_s": 1.5,
            "speaker_remuted": True,
            "audio_trace": [
                {"state": "audible", "volume": 6},
                {"state": "bark_requested"},
                {"state": "muted", "volume": 0},
            ],
        }
        assert policy.status()["speaker_muted"] is True

    asyncio.run(scenario())


def test_system_audio_remutes_when_bark_fails() -> None:
    async def scenario() -> None:
        vui = FakeVui(volume=4)
        policy = SystemAudioPolicy(
            vui,
            FakeBark(vui.events, fail=True),
            SystemAudioConfig(bark_volume=5, bark_audible_s=1.0),
            sleep=lambda _duration: asyncio.sleep(0),
        )
        await policy.start_muted()

        with pytest.raises(BarkFailure, match="speaker request failed"):
            await policy.bark()

        assert vui.volume == 0
        assert vui.events[-2:] == [("set", 0), ("get", 0)]

    asyncio.run(scenario())


def test_system_audio_records_inaudible_bark_without_failing_mission() -> None:
    async def scenario() -> None:
        vui = StuckMutedVui(volume=0)
        policy = SystemAudioPolicy(
            vui,
            FakeBark(vui.events),
            SystemAudioConfig(bark_volume=6, bark_audible_s=1.0),
            sleep=lambda _duration: asyncio.sleep(0),
        )
        await policy.start_muted()

        result = await policy.bark()

        assert result["bark_played"] is True
        assert result["speaker_audible"] is False
        assert result["speaker_remuted"] is True
        assert "expected 6, got 0" in str(result["speaker_warning"])
        assert policy.status()["ready"] is True
        assert "expected 6, got 0" in str(policy.status()["speaker_warning"])
        assert vui.volume == 0

    asyncio.run(scenario())
