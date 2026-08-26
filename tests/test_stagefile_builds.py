from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_deployment_services_are_stagefile_only() -> None:
    for service_dir in (ROOT, ROOT / "media", ROOT / "voice"):
        assert (service_dir / "build.stagefile.yaml").is_file()
        assert (service_dir / "build.stagefile.lock.yaml").is_file()
        assert not (service_dir / "Dockerfile").exists()


def test_media_pip_overlay_is_visible_to_dustynv_virtualenv() -> None:
    stagefile = yaml.safe_load((ROOT / "media/build.stagefile.yaml").read_text())
    media = stagefile["stages"][-1]

    assert media["env"]["PYTHONPATH"] == "/usr/local/lib/python3.12/site-packages"


def test_coco_replacement_model_is_content_pinned_in_the_media_stagefile() -> None:
    stagefile = yaml.safe_load((ROOT / "media/build.stagefile.yaml").read_text())
    media = stagefile["stages"][-1]

    assert media["download"] == [
        {
            "url": (
                "https://github.com/ultralytics/assets/releases/download/"
                "v8.3.0/yolo11n.pt"
            ),
            "sha256": (
                "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1"
            ),
            "dest": "/models/yolo11n.pt",
        }
    ]
