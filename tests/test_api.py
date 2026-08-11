import json
from pathlib import Path

from fastapi.testclient import TestClient

from border_collie_demo.api import create_app


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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


def test_demo_advertises_control_ui_before_diagnostics() -> None:
    manifest = json.loads((PROJECT_ROOT / "wendy.json").read_text(encoding="utf-8"))
    metadata = json.loads(
        (PROJECT_ROOT / "wendy-demo.json").read_text(encoding="utf-8")
    )
    service = manifest["services"]["app"]

    assert {"type": "http", "port": 8096} in service["entitlements"]
    assert metadata["safety"] == "control"
    assert [link["kind"] for link in metadata["links"]] == ["control", "debug"]
