from fastapi.testclient import TestClient

from robotkit.world_state.app import create_app
from tests.test_store import observation


def test_world_state_http_contract(tmp_path):
    with TestClient(create_app(tmp_path / "api.sqlite3")) as client:
        assert client.get("/healthz").json() == {"status": "ok"}
        response = client.post(
            "/v1/observations",
            json=observation(key="frame-1").model_dump(mode="json"),
        )
        assert response.status_code == 201

        duplicate = client.post(
            "/v1/observations",
            json=observation(key="frame-1").model_dump(mode="json"),
        )
        assert duplicate.status_code == 200
        assert duplicate.json()["duplicate"] is True

        snapshot = client.get("/v1/state").json()
        assert snapshot["state_revision"] == 1
        assert snapshot["observations"][0]["producer_id"] == "yolo"
        assert client.get("/v1/events").json()[0]["category"] == "observation.published"


def test_latest_effect_endpoint_returns_none_without_effects(tmp_path):
    with TestClient(create_app(tmp_path / "api.sqlite3")) as client:
        response = client.get("/v1/effects/latest")
        assert response.status_code == 200
        assert response.json() is None
