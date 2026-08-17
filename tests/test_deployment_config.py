from __future__ import annotations

import json
from pathlib import Path, PurePosixPath


def _descriptor() -> dict:
    return json.loads(
        (Path(__file__).parents[1] / "wendy.json").read_text(encoding="utf-8")
    )


def test_apple_motion_thresholds_are_owned_only_by_the_app_service() -> None:
    descriptor = json.loads(
        (Path(__file__).parents[1] / "wendy.json").read_text(encoding="utf-8")
    )
    app_env = descriptor["services"]["app"]["env"]
    media_env = descriptor["services"]["media"]["env"]

    assert app_env["BORDER_COLLIE_APPLE_FOCUS_CONFIDENCE"] == "0.40"
    assert app_env["BORDER_COLLIE_APPLE_ACQUISITION_CONFIDENCE"] == "0.40"
    assert app_env["BORDER_COLLIE_STAGE_HOME_MARGIN_M"] == "0.50"
    assert app_env["BORDER_COLLIE_GUIDANCE_FINAL_PUSH_MPS"] == "0.60"
    assert app_env["BORDER_COLLIE_GUIDANCE_FINAL_PUSH_DURATION_S"] == "1.0"
    assert "BORDER_COLLIE_SYSTEM_AUDIO_ENABLED" not in app_env
    assert "BORDER_COLLIE_SYSTEM_AUDIO_QUIET_VOLUME" not in app_env
    assert "BORDER_COLLIE_APPLE_FOCUS_CONFIDENCE" not in media_env
    assert "BORDER_COLLIE_APPLE_ACQUISITION_CONFIDENCE" not in media_env


def test_app_run_store_is_mounted_on_its_own_persist_volume() -> None:
    """Run records must outlive the container, or every redeploy wipes them."""
    descriptor = _descriptor()
    app = descriptor["services"]["app"]

    volumes = [
        entitlement
        for entitlement in app["entitlements"]
        if entitlement["type"] == "persist"
    ]
    assert len(volumes) == 1
    volume = volumes[0]
    assert volume["name"] == "border-collie-run-store"
    assert volume["path"] == "/run-store"

    # The volume must not be shared with the voice service's model cache.
    voice_volumes = {
        entitlement["name"]
        for entitlement in descriptor["services"]["voice"]["entitlements"]
        if entitlement["type"] == "persist"
    }
    assert volume["name"] not in voice_volumes


def test_run_and_cohort_directories_sit_inside_the_persist_mount() -> None:
    """A path outside the mount is silently ephemeral again."""
    descriptor = _descriptor()
    app = descriptor["services"]["app"]
    mount = PurePosixPath("/run-store")

    runs_dir = PurePosixPath(app["env"]["BORDER_COLLIE_RUNS_DIR"])
    cohorts_dir = PurePosixPath(app["env"]["BORDER_COLLIE_COHORTS_DIR"])

    assert runs_dir == PurePosixPath("/run-store/runs")
    assert cohorts_dir == PurePosixPath("/run-store/cohorts")
    assert runs_dir.is_relative_to(mount)
    assert cohorts_dir.is_relative_to(mount)
    # Cohort records default to a `cohorts/` sibling of the runs directory, so
    # the runs directory has to be one level inside the mount. Pointing it at
    # the mount root would derive `/cohorts` and drop cohort policy back onto
    # the container's ephemeral filesystem.
    assert runs_dir.parent == mount
    assert cohorts_dir.parent == mount
