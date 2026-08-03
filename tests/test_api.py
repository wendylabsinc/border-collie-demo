from fastapi.testclient import TestClient

from border_collie_demo.api import create_app


def test_status_is_explicitly_non_operational() -> None:
    response = TestClient(create_app()).get("/api/status")

    assert response.status_code == 200
    assert response.json()["hardware_enabled"] is False
    assert response.json()["can_move"] is False


def test_scaffold_has_no_activate_endpoint() -> None:
    response = TestClient(create_app()).post("/api/run")

    assert response.status_code == 404
