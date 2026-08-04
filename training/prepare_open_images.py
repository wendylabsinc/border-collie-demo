from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen

TARGET_CLASSES = {
    "/m/014j1m": (0, "apple"),
    "/m/09qck": (1, "banana"),
    "/m/061_f": (2, "pear"),
}
ALLOWED_LICENSES = {"https://creativecommons.org/licenses/by/2.0/"}
IMAGE_URL_TEMPLATE = (
    "https://open-images-dataset.s3.amazonaws.com/{source_split}/{image_id}.jpg"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_records(
    annotations_path: Path, metadata_path: Path
) -> list[dict[str, object]]:
    annotations: dict[str, list[dict[str, object]]] = {}
    with annotations_path.open(newline="") as source:
        for row in csv.DictReader(source):
            target = TARGET_CLASSES.get(row["LabelName"])
            if target is None:
                continue
            if row["IsGroupOf"] == "1" or row["IsDepiction"] == "1":
                continue
            class_id, class_name = target
            annotations.setdefault(row["ImageID"], []).append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "bbox_xyxy_normalized": [
                        float(row["XMin"]),
                        float(row["YMin"]),
                        float(row["XMax"]),
                        float(row["YMax"]),
                    ],
                    "is_occluded": row["IsOccluded"] == "1",
                    "is_truncated": row["IsTruncated"] == "1",
                    "is_group_of": row["IsGroupOf"] == "1",
                    "is_depiction": row["IsDepiction"] == "1",
                    "is_inside": row["IsInside"] == "1",
                }
            )

    records: list[dict[str, object]] = []
    with metadata_path.open(newline="") as source:
        for row in csv.DictReader(source):
            image_annotations = annotations.get(row["ImageID"])
            if image_annotations is None:
                continue
            if row["License"] not in ALLOWED_LICENSES:
                raise ValueError(
                    f"unapproved image license for {row['ImageID']}: {row['License']}"
                )
            records.append(
                {
                    "image_id": row["ImageID"],
                    "source_split": row["Subset"],
                    "download_url": IMAGE_URL_TEMPLATE.format(
                        source_split=row["Subset"], image_id=row["ImageID"]
                    ),
                    "original_url": row["OriginalURL"],
                    "original_landing_url": row["OriginalLandingURL"],
                    "license_url": row["License"],
                    "author_profile_url": row["AuthorProfileURL"],
                    "author": row["Author"],
                    "title": row["Title"],
                    "original_md5": row["OriginalMD5"],
                    "rotation_degrees": float(row["Rotation"] or 0),
                    "annotations": image_annotations,
                }
            )

    missing_metadata = sorted(
        set(annotations) - {record["image_id"] for record in records}
    )
    if missing_metadata:
        raise ValueError(f"metadata missing for {len(missing_metadata)} target images")
    return sorted(records, key=lambda record: str(record["image_id"]))


def _download(record: dict[str, object], image_dir: Path) -> dict[str, object]:
    image_id = str(record["image_id"])
    destination = image_dir / f"{image_id}.jpg"
    if not destination.exists():
        request = Request(
            str(record["download_url"]), headers={"User-Agent": "WendyLabs/1"}
        )
        with urlopen(request, timeout=30) as response:
            content = response.read()
        destination.write_bytes(content)
    return {
        **record,
        "local_image": f"images/candidates/{destination.name}",
        "image_bytes": destination.stat().st_size,
        "image_sha256": _sha256(destination),
    }


def _write_yolo_label(record: dict[str, object], labels_dir: Path) -> None:
    lines = []
    for annotation in record["annotations"]:
        x1, y1, x2, y2 = annotation["bbox_xyxy_normalized"]
        lines.append(
            " ".join(
                (
                    str(annotation["class_id"]),
                    f"{(x1 + x2) / 2:.8f}",
                    f"{(y1 + y2) / 2:.8f}",
                    f"{x2 - x1:.8f}",
                    f"{y2 - y1:.8f}",
                )
            )
        )
    (labels_dir / f"{record['image_id']}.txt").write_text("\n".join(lines) + "\n")


def prepare_dataset(
    annotations_path: Path,
    metadata_path: Path,
    output_path: Path,
    *,
    workers: int = 8,
) -> dict[str, object]:
    records = build_records(annotations_path, metadata_path)
    image_dir = output_path / "images" / "candidates"
    labels_dir = output_path / "labels" / "candidates"
    image_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        downloaded = list(
            executor.map(lambda record: _download(record, image_dir), records)
        )
    for record in downloaded:
        _write_yolo_label(record, labels_dir)

    class_boxes: Counter[str] = Counter()
    class_images: dict[str, set[str]] = {
        class_name: set() for _, class_name in TARGET_CLASSES.values()
    }
    for record in downloaded:
        for annotation in record["annotations"]:
            class_name = str(annotation["class_name"])
            class_boxes[class_name] += 1
            class_images[class_name].add(str(record["image_id"]))
    manifest = {
        "schema_version": 1,
        "dataset": "Open Images V7",
        "source_annotation_splits": sorted(
            {str(record["source_split"]) for record in downloaded}
        ),
        "corpus_role": "public curation candidate pool",
        "class_map": {name: class_id for _, (class_id, name) in TARGET_CLASSES.items()},
        "source_files": {
            "annotations_sha256": _sha256(annotations_path),
            "metadata_sha256": _sha256(metadata_path),
        },
        "counts": {
            "images": len(downloaded),
            "boxes_by_class": dict(sorted(class_boxes.items())),
            "images_by_class": {
                name: len(image_ids) for name, image_ids in sorted(class_images.items())
            },
        },
        "license_audit": {
            "allowed_license_urls": sorted(ALLOWED_LICENSES),
            "metadata_complete": True,
            "independent_rights_verification_complete": False,
        },
        "annotation_policy": {
            "group_of_boxes_included": False,
            "depiction_boxes_included": False,
        },
        "whole_fruit_curation": {
            "required": True,
            "complete": False,
            "training_eligible": False,
            "policy": "training/whole-fruit-policy.md",
        },
        "records": downloaded,
    }
    (output_path / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare an audited Open Images fruit subset"
    )
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", default=8, type=int)
    args = parser.parse_args()
    manifest = prepare_dataset(
        args.annotations,
        args.metadata,
        args.output,
        workers=args.workers,
    )
    print(json.dumps(manifest["counts"], indent=2))


if __name__ == "__main__":
    main()
