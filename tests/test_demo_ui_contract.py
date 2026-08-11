import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_operator_and_diagnostics_pages_use_wendy_contract() -> None:
    for filename in ("index.html", "debug.html"):
        html = (PROJECT_ROOT / "web" / filename).read_text(encoding="utf-8")
        assert "WENDY" in html
        assert "viewport-fit=cover" in html
        assert 'content="#f1eee7"' in html
        assert "http://127.0.0.1:8088/" in html
        assert "min-height: 48px" in html

    operator = (PROJECT_ROOT / "web" / "index.html").read_text(encoding="utf-8")
    assert "factory obstacle avoidance" in operator
    assert "min-height: 56px" in operator


def test_deployed_demo_advertises_control_ui_before_diagnostics() -> None:
    manifest = json.loads((PROJECT_ROOT / "wendy.json").read_text(encoding="utf-8"))
    metadata = json.loads(
        (PROJECT_ROOT / "wendy-demo.json").read_text(encoding="utf-8")
    )
    service = manifest["services"]["app"]

    assert {"type": "http", "port": 8110} in service["entitlements"]
    assert metadata["safety"] == "control"
    assert [link["kind"] for link in metadata["links"]] == ["control", "debug"]
    assert metadata["links"][0]["path"] == "/"
