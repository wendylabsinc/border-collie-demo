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
        fetcher=lambda _url, _timeout: {"bark_ready": True},
    )

    assert client.status() == {
        "ready": True,
        "detail": "Go2 bark sidecar is ready",
    }
