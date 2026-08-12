from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from fastapi.testclient import TestClient

from robotkit.contracts import ObservationRecord, PublishResult, WorldSnapshot
from robotkit.perception.website.app import create_app
from robotkit.perception.website.diagnostics import build_debug_snapshot
from robotkit.perception.website.producer import WebsiteCommandProducer


NOW = datetime(2026, 8, 12, 12, 0, tzinfo=timezone.utc)
REQUEST_ID = UUID("93070866-7198-4c5e-8a6f-38cdfbf3295d")


class FakePublisher:
    def __init__(self) -> None:
        self.observations = []

    def publish_observation(self, observation):
        self.observations.append(observation)
        return PublishResult(revision=17, duplicate=False)


class FakeStateReader:
    def __init__(self, snapshot):
        self._snapshot = snapshot

    def snapshot(self):
        return self._snapshot

    def current_goal(self):
        return None

    def latest_effect(self, goal_id=None):
        return None

    def events(self, *, after=0, limit=100):
        return []


def test_pure_producer_builds_auditable_website_intent():
    producer = WebsiteCommandProducer(
        producer_id="website-test",
        instance_id="green-2",
        deployment_generation=4,
    )

    observation = producer.observation_from_command(
        "  Go to the apple! ", request_id=REQUEST_ID, observed_at=NOW
    )

    assert observation.stream == "website.intent"
    assert observation.observation_type == "website.intent.command"
    assert observation.idempotency_key == f"website-test:{REQUEST_ID}:intent"
    assert observation.deployment_generation == 4
    assert observation.payload == {
        "intent": "find",
        "slots": {"target": "apple"},
        "source_text": "Go to the apple!",
        "request_id": str(REQUEST_ID),
    }


def test_website_endpoint_publishes_without_authentication():
    publisher = FakePublisher()
    producer = WebsiteCommandProducer(producer_id="website-test")
    app = create_app(publisher, producer=producer)

    with TestClient(app) as client:
        accepted = client.post(
            "/v1/command",
            json={"command": "find apple", "request_id": str(REQUEST_ID)},
        )

    assert accepted.status_code == 201
    assert accepted.json() == {
        "event_id": str(publisher.observations[0].event_id),
        "revision": 17,
        "duplicate": False,
        "stream": "website.intent",
        "intent": "find",
        "slots": {"target": "apple"},
    }
    assert len(publisher.observations) == 1


def test_page_is_self_contained_and_has_no_token_prompt():
    app = create_app(FakePublisher())

    with TestClient(app) as client:
        response = client.get("/")
        health = client.get("/healthz")

    assert response.status_code == 200
    assert "RobotKit command" in response.text
    assert "Command token" not in response.text
    assert "Authorization" not in response.text
    assert "crypto.randomUUID" not in response.text
    assert health.json() == {"status": "ok"}


def test_debug_page_and_api_are_read_only_views_of_a():
    snapshot = WorldSnapshot(
        revision=7,
        state_revision=5,
        captured_at=NOW,
        observations=[],
    )
    app = create_app(FakePublisher(), state_reader=FakeStateReader(snapshot))

    with TestClient(app) as client:
        page = client.get("/debug")
        debug = client.get("/v1/debug")

    assert page.status_code == 200
    assert "RobotKit live diagnostics" in page.text
    assert "setInterval(refresh,1000)" in page.text
    assert "details.open = openStages.has(stage.name)" in page.text
    assert debug.status_code == 200
    assert debug.json()["revision"] == 7
    assert [stage["name"] for stage in debug.json()["stages"]] == [
        "command",
        "perception",
        "planner",
        "controller",
        "executor",
    ]


def test_debug_reducer_explains_unsupported_pear_without_shell_access():
    producer = WebsiteCommandProducer(instance_id="test")
    command = producer.observation_from_command("find pear", observed_at=NOW)
    record = ObservationRecord(
        **command.model_dump(), revision=4, received_at=NOW
    )
    snapshot = WorldSnapshot(
        revision=4,
        state_revision=4,
        captured_at=NOW,
        observations=[record],
    )

    debug = build_debug_snapshot(
        snapshot,
        current_goal=None,
        latest_effect=None,
        supported_targets=("apple",),
    )

    perception = next(stage for stage in debug["stages"] if stage["name"] == "perception")
    planner = next(stage for stage in debug["stages"] if stage["name"] == "planner")
    assert debug["requested_target"] == "pear"
    assert perception["status"] == "error"
    assert "Unsupported target 'pear'" in perception["summary"]
    assert planner["summary"] == "Planner rejected unsupported target 'pear'"
