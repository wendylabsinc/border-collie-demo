from __future__ import annotations

import json
from pathlib import Path


def test_motion_defaults_are_owned_only_by_the_app_service() -> None:
    descriptor = json.loads(
        (Path(__file__).parents[1] / "wendy.json").read_text(encoding="utf-8")
    )
    app_env = descriptor["services"]["app"]["env"]
    media_env = descriptor["services"]["media"]["env"]

    assert app_env["BORDER_COLLIE_APPLE_FOCUS_CONFIDENCE"] == "0.40"
    assert app_env["BORDER_COLLIE_APPLE_ACQUISITION_CONFIDENCE"] == "0.40"
    assert app_env["BORDER_COLLIE_STAGE_HOME_MARGIN_M"] == "0.50"
    assert app_env["BORDER_COLLIE_HOME_ALIGN_YAW_RPS"] == "0.80"
    assert app_env["BORDER_COLLIE_HOME_STALL_TIMEOUT_S"] == "5.0"
    assert app_env["BORDER_COLLIE_SEARCH_MOTION_PATH"] == "sport_yaw"
    assert app_env["BORDER_COLLIE_GUIDANCE_SEARCH_YAW_RPS"] == "0.40"
    assert app_env["BORDER_COLLIE_GUIDANCE_FOCUS_YAW_RPS"] == "0.40"
    assert app_env["BORDER_COLLIE_GUIDANCE_FOCUS_MINIMUM_YAW_RPS"] == "0.40"
    assert app_env["BORDER_COLLIE_GUIDANCE_FINAL_PUSH_MPS"] == "0.60"
    assert app_env["BORDER_COLLIE_GUIDANCE_FINAL_PUSH_DURATION_S"] == "1.0"
    assert "BORDER_COLLIE_SYSTEM_AUDIO_ENABLED" not in app_env
    assert "BORDER_COLLIE_SYSTEM_AUDIO_QUIET_VOLUME" not in app_env
    assert "BORDER_COLLIE_APPLE_FOCUS_CONFIDENCE" not in media_env
    assert "BORDER_COLLIE_APPLE_ACQUISITION_CONFIDENCE" not in media_env
