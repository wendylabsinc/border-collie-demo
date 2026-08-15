from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_deployment_services_are_stagefile_only() -> None:
    for service_dir in (
        ROOT,
        ROOT / "media",
        ROOT / "voice",
        ROOT / "home-recorder",
    ):
        assert (service_dir / "build.stagefile.yaml").is_file()
        assert (service_dir / "build.stagefile.lock.yaml").is_file()
        assert not (service_dir / "Dockerfile").exists()


def test_all_stagefile_locks_match_their_exact_sources() -> None:
    for service_dir in (
        ROOT,
        ROOT / "media",
        ROOT / "voice",
        ROOT / "home-recorder",
    ):
        source = service_dir / "build.stagefile.yaml"
        lock = yaml.safe_load(
            (service_dir / "build.stagefile.lock.yaml").read_text()
        )
        expected = "sha256:" + hashlib.sha256(source.read_bytes()).hexdigest()

        assert lock["version"] == 1
        assert lock["sourceHash"] == expected
        assert all(value.startswith("sha256:") for value in lock["images"].values())


def test_media_pip_overlay_is_visible_to_dustynv_virtualenv() -> None:
    stagefile = yaml.safe_load((ROOT / "media/build.stagefile.yaml").read_text())
    media = stagefile["stages"][-1]

    assert media["env"]["PYTHONPATH"] == "/usr/local/lib/python3.12/site-packages"


def test_media_stagefile_packages_the_named_bark_asset_not_a_robot_uuid() -> None:
    stagefile = yaml.safe_load((ROOT / "media/build.stagefile.yaml").read_text())
    media = stagefile["stages"][-1]
    copied_paths = {
        path
        for operation in media["copy"]
        for path in operation["paths"]
    }

    assert "assets/border_collie_demo_bark.wav.b64" in copied_paths
    assert "BORDER_COLLIE_BARK_UUID" not in media["env"]


def test_home_recorder_is_a_separate_passive_service_with_shared_durable_state() -> None:
    descriptor = yaml.safe_load((ROOT / "wendy.json").read_text())
    app = descriptor["services"]["app"]
    recorder = descriptor["services"]["home-recorder"]
    stagefile = yaml.safe_load(
        (ROOT / "home-recorder/build.stagefile.yaml").read_text()
    )

    assert recorder["context"] == "./home-recorder"
    assert recorder.get("dependsOn") is None
    assert {item["type"] for item in recorder["entitlements"]} == {
        "network",
        "persist",
    }
    app_persist = next(item for item in app["entitlements"] if item["type"] == "persist")
    recorder_persist = next(
        item for item in recorder["entitlements"] if item["type"] == "persist"
    )
    assert app_persist == recorder_persist == {
        "type": "persist",
        "name": "border-collie-demo-recordings",
        "path": "/state",
    }
    assert stagefile["stages"][-1]["cmd"] == ["python", "home_recorder.py"]
    assert recorder["env"]["HOME_RECORDER_YAW_DRIFT_THRESHOLD_M"] == "0.03"
    assert recorder["env"]["HOME_RECORDER_COMMAND_ACTIVE_S"] == "0.50"


def test_app_enables_read_only_controller_start_subscription() -> None:
    descriptor = yaml.safe_load((ROOT / "wendy.json").read_text())
    app = descriptor["services"]["app"]

    assert app["env"]["BORDER_COLLIE_CONTROLLER_START_ENABLED"] == "1"
    assert {item["type"] for item in app["entitlements"]} >= {"network", "persist"}
