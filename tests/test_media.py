import asyncio

from border_collie_demo.media import BarkClient, BarkConfig


def test_bark_client_requires_positive_sidecar_acknowledgement() -> None:
    async def scenario() -> None:
        calls = []
        client = BarkClient(
            BarkConfig(enabled=True, url="http://127.0.0.1:8098/api/bark"),
            poster=lambda url, timeout: calls.append((url, timeout)) or {"ok": True, "uuid": "bark-1"},
        )

        result = await client.bark()

        assert calls == [("http://127.0.0.1:8098/api/bark", 2.0)]
        assert result == {
            "bark_played": True,
            "bark_uuid": "bark-1",
            "bark_sound": None,
            "bark_source": None,
        }

    asyncio.run(scenario())


def test_bark_readiness_is_checked_before_demo_preflight() -> None:
    client = BarkClient(
        BarkConfig(enabled=True),
        fetcher=lambda _url, _timeout: {"bark_ready": True},
    )

    assert client.status() == {
        "ready": True,
        "detail": "Go2 bark sidecar is ready",
    }


def test_thermal_beep_client_posts_to_existing_media_owner() -> None:
    calls: list[tuple[str, float]] = []
    client = BarkClient(
        BarkConfig(
            enabled=True,
            thermal_beep_url="http://127.0.0.1:8111/api/thermal/beep",
        ),
        poster=lambda url, timeout: (
            calls.append((url, timeout))
            or {"ok": True, "uuid": "thermal-beep"}
        ),
    )

    result = asyncio.run(client.thermal_beep())

    assert calls == [("http://127.0.0.1:8111/api/thermal/beep", 2.0)]
    assert result == {
        "thermal_beep_played": True,
        "thermal_beep_uuid": "thermal-beep",
    }
