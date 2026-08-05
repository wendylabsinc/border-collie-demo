from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest

PREPARER_PATH = Path(__file__).parents[1] / "training" / "prepare_open_images.py"


def load_preparer_module():
    spec = importlib.util.spec_from_file_location("prepare_open_images", PREPARER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_build_records_keeps_boxes_and_attribution(tmp_path: Path) -> None:
    preparer = load_preparer_module()
    annotations = tmp_path / "boxes.csv"
    metadata = tmp_path / "images.csv"
    write_csv(
        annotations,
        [
            {
                "ImageID": "image-1",
                "LabelName": "/m/09qck",
                "XMin": "0.1",
                "XMax": "0.3",
                "YMin": "0.2",
                "YMax": "0.6",
                "IsOccluded": "0",
                "IsTruncated": "0",
                "IsGroupOf": "0",
                "IsDepiction": "0",
                "IsInside": "0",
            }
        ],
    )
    write_csv(
        metadata,
        [
            {
                "ImageID": "image-1",
                "Subset": "validation",
                "OriginalURL": "https://example.test/image.jpg",
                "OriginalLandingURL": "https://example.test/page",
                "License": "https://creativecommons.org/licenses/by/2.0/",
                "AuthorProfileURL": "https://example.test/author",
                "Author": "Example Author",
                "Title": "Banana",
                "OriginalMD5": "md5",
                "Rotation": "0.0",
            }
        ],
    )

    records = preparer.build_records(annotations, metadata)

    assert records[0]["author"] == "Example Author"
    assert records[0]["annotations"][0]["class_name"] == "banana"
    assert records[0]["annotations"][0]["bbox_xyxy_normalized"] == [
        0.1,
        0.2,
        0.3,
        0.6,
    ]


def test_build_records_excludes_group_and_depiction_boxes(tmp_path: Path) -> None:
    preparer = load_preparer_module()
    annotations = tmp_path / "boxes.csv"
    metadata = tmp_path / "images.csv"
    write_csv(
        annotations,
        [
            {
                "ImageID": image_id,
                "LabelName": "/m/09qck",
                "XMin": "0.1",
                "XMax": "0.3",
                "YMin": "0.2",
                "YMax": "0.6",
                "IsOccluded": "0",
                "IsTruncated": "0",
                "IsGroupOf": group,
                "IsDepiction": depiction,
                "IsInside": "0",
            }
            for image_id, group, depiction in (
                ("group", "1", "0"),
                ("depiction", "0", "1"),
            )
        ],
    )
    write_csv(
        metadata,
        [
            {
                "ImageID": image_id,
                "Subset": "validation",
                "OriginalURL": "https://example.test/image.jpg",
                "OriginalLandingURL": "https://example.test/page",
                "License": "https://creativecommons.org/licenses/by/2.0/",
                "AuthorProfileURL": "",
                "Author": "",
                "Title": "",
                "OriginalMD5": "",
                "Rotation": "0",
            }
            for image_id in ("group", "depiction")
        ],
    )

    assert preparer.build_records(annotations, metadata) == []


def test_build_records_rejects_an_unapproved_license(tmp_path: Path) -> None:
    preparer = load_preparer_module()
    annotations = tmp_path / "boxes.csv"
    metadata = tmp_path / "images.csv"
    write_csv(
        annotations,
        [
            {
                "ImageID": "image-1",
                "LabelName": "/m/014j1m",
                "XMin": "0.1",
                "XMax": "0.3",
                "YMin": "0.2",
                "YMax": "0.6",
                "IsOccluded": "0",
                "IsTruncated": "0",
                "IsGroupOf": "0",
                "IsDepiction": "0",
                "IsInside": "0",
            }
        ],
    )
    write_csv(
        metadata,
        [
            {
                "ImageID": "image-1",
                "Subset": "validation",
                "OriginalURL": "https://example.test/image.jpg",
                "OriginalLandingURL": "https://example.test/page",
                "License": "https://example.test/unknown-license",
                "AuthorProfileURL": "",
                "Author": "",
                "Title": "",
                "OriginalMD5": "",
                "Rotation": "0",
            }
        ],
    )

    with pytest.raises(ValueError, match="unapproved image license"):
        preparer.build_records(annotations, metadata)
