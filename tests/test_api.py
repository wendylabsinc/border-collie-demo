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


def test_operator_page_uses_wendy_brand_and_home_navigation() -> None:
    response = TestClient(create_app()).get("/")

    assert response.status_code == 200
    assert "WENDY" in response.text
    assert "#f1eee7" in response.text
    assert "http://127.0.0.1:8088/" in response.text
    assert "Hardware control surface" in response.text
