from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

import pytest


LABELER_ROOT = Path(__file__).parents[1] / "lab" / "run-labeler"


def load_labeler_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "run_labeler_server",
        LABELER_ROOT / "server.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_manifest_accepts_a_fieldmark_frame_archive(tmp_path: Path) -> None:
    server = load_labeler_module()
    archive_path = tmp_path / "evidence.zip"
    manifest = {
        "format": "fieldmark-image-sequence",
        "frames": [{"filename": "frames/000001.jpg", "width": 1280, "height": 720}],
    }
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        archive.writestr("frames/000001.jpg", b"jpeg")

    assert server.load_manifest(archive_path) == manifest


def test_load_manifest_rejects_unsafe_frame_names(tmp_path: Path) -> None:
    server = load_labeler_module()
    archive_path = tmp_path / "evidence.zip"
    manifest = {"frames": [{"filename": "frames/../secret.jpg"}]}
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))

    with pytest.raises(ValueError, match="unsafe"):
        server.load_manifest(archive_path)
