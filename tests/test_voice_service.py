from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
HEY_WENDY_SHA256 = "c28c91e79cb7fea92c420aa5bf91c3cd99e2f36921e38f8b614ba9d12c0b5acf"


def test_voice_service_uses_the_stage_default_dog_identity_and_local_boundaries() -> None:
    descriptor = json.loads((ROOT / "wendy.json").read_text())
    app_env = descriptor["services"]["app"]["env"]
    voice = descriptor["services"]["voice"]
    voice_env = voice["env"]

    assert voice["dependsOn"] == ["app", "media"]
    assert voice_env["ACTION_MODE"] == "border_collie"
    assert voice_env["BORDER_COLLIE_URL"] == "http://127.0.0.1:8110"
    assert voice_env["BORDER_COLLIE_EXPECTED_BUILD_LABEL"] == app_env[
        "BORDER_COLLIE_BUILD_LABEL"
    ]
    assert "BORDER_COLLIE_EXPECTED_RELEASE_ID" not in voice_env
    assert "BORDER_COLLIE_EXPECTED_CONFIG_SCHEMA" not in voice_env
    assert voice_env["WAKE_WORD"] == "/app/hey_wendy.onnx"
    assert voice_env["CONTINUOUS_TRANSCRIPTION"] == "0"
    assert voice_env["AUTO_ARM_ACTIONS"] == "1"
    assert voice_env["AUDIO_DEVICE"] == "DJI MIC MINI"
    assert voice_env["MICROPHONE_RETRY_INTERVAL_S"] == "10.0"
    assert {item["type"] for item in voice["entitlements"]} == {
        "network",
        "audio",
        "persist",
    }
    assert next(
        item for item in voice["entitlements"] if item["type"] == "persist"
    )["path"] == "/models"


def test_voice_build_is_stagefile_only() -> None:
    stagefile = yaml.safe_load((ROOT / "voice/build.stagefile.yaml").read_text())
    assert not (ROOT / "voice/Dockerfile").exists()
    assert (ROOT / "voice/build.stagefile.lock.yaml").is_file()

    stage = stagefile["stages"][-1]
    assert stage["cmd"] == ["python", "app.py"]
    assert any(
        item.get("paths") == ["hey_wendy.onnx"]
        for item in stage["copy"]
        if item.get("from") == "local"
    )
    downloads = {item["dest"]: item for item in stage["download"]}
    assert downloads["/opt/openwakeword-models/melspectrogram.onnx"]["sha256"]
    assert downloads["/opt/openwakeword-models/embedding_model.onnx"]["sha256"]


def test_bundled_hey_wendy_model_has_the_reviewed_digest() -> None:
    digest = hashlib.sha256((ROOT / "voice/hey_wendy.onnx").read_bytes()).hexdigest()
    assert digest == HEY_WENDY_SHA256


def test_audience_page_links_the_local_voice_dashboard() -> None:
    page = (ROOT / "web/index.html").read_text()
    assert "Open Hey Wendy voice dashboard" in page
    assert "http://woof.local:8092/" in page
