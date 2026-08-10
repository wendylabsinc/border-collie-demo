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
        assert result == {"bark_played": True, "bark_uuid": "bark-1"}

    asyncio.run(scenario())


def test_bark_readiness_is_checked_before_demo_preflight() -> None:
    client = BarkClient(
        BarkConfig(enabled=True),
        fetcher=lambda _url, _timeout: {
            "bark_ready": True,
            "supervision": {
                "state": "ready",
                "ready": True,
                "generation": "camera-1",
            },
        },
    )

    assert client.status() == {
        "ready": True,
        "detail": "Go2 bark sidecar is ready",
    }


def test_bark_readiness_fails_closed_while_media_is_reconnecting() -> None:
    client = BarkClient(
        BarkConfig(enabled=True),
        fetcher=lambda _url, _timeout: {
            "bark_ready": True,
            "supervision": {
                "state": "degraded",
                "ready": False,
                "generation": "camera-2",
                "last_error": "waiting for stable frames",
            },
        },
    )

    assert client.status() == {
        "ready": False,
        "detail": "waiting for stable frames",
    }
