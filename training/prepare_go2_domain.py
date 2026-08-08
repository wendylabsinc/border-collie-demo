"""Build a leak-checked Go2-camera domain adaptation dataset.

The three stage-critical tournament fixtures are never copied into this
dataset. Pear boxes come from the production TensorRT evidence manifest,
banana boxes are human-reviewed, and the static green-apple sequence is boxed
with a deterministic color/component rule that fails closed on ambiguity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

CLASS_IDS = {"pear": 0, "apple": 1, "banana": 2}


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _split(frame_index: int) -> str:
    return "val" if frame_index % 5 == 0 else "train"


def _yolo_line(
    class_name: str,
    bbox_xyxy: list[float] | tuple[float, float, float, float],
    *,
    width: int,
    height: int,
) -> str:
    x1, y1, x2, y2 = (float(value) for value in bbox_xyxy)
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"invalid {class_name} box {bbox_xyxy}")
    return (
        f"{CLASS_IDS[class_name]} {(x1 + x2) / (2 * width):.8f} "
        f"{(y1 + y2) / (2 * height):.8f} {(x2 - x1) / width:.8f} "
        f"{(y2 - y1) / height:.8f}\n"
    )


def detect_green_apple_bbox(encoded_jpeg: bytes) -> list[int]:
    image = cv2.imdecode(np.frombuffer(encoded_jpeg, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("green-apple frame is not a readable JPEG")
    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (35, 70, 35), (95, 255, 255))
    region = np.zeros_like(mask)
    region[round(0.45 * height) : round(0.90 * height), round(0.25 * width) : round(0.75 * width)] = 255
    mask = cv2.bitwise_and(mask, region)
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        np.ones((3, 3), dtype=np.uint8),
    )
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    candidates: list[tuple[int, list[int]]] = []
    for component in range(1, count):
        x, y, box_width, box_height, area = (
            int(value) for value in stats[component]
        )
        aspect = box_width / max(1, box_height)
        if 50 <= area <= 5000 and 0.4 <= aspect <= 2.0:
            candidates.append(
                (area, [x, y, x + box_width, y + box_height])
            )
    if not candidates:
        raise ValueError("green-apple component was not found")
    candidates.sort(reverse=True)
    if len(candidates) > 1 and candidates[1][0] >= candidates[0][0] * 0.75:
        raise ValueError("green-apple component is ambiguous")
    return candidates[0][1]


def _write_example(
    output: Path,
    *,
    source_id: str,
    frame_index: int,
    image: bytes,
    class_name: str,
    box: list[float],
    width: int,
    height: int,
) -> dict[str, Any]:
    split = _split(frame_index)
    stem = f"go2-{source_id}-{frame_index:06d}"
    image_path = output / "images" / split / f"{stem}.jpg"
    label_path = output / "labels" / split / f"{stem}.txt"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(image)
    label_path.write_text(
        _yolo_line(class_name, box, width=width, height=height),
        encoding="utf-8",
    )
    return {
        "source_id": source_id,
        "frame_index": frame_index,
        "split": split,
        "class_name": class_name,
        "bbox_xyxy": box,
        "image_sha256": _sha256(image),
    }


def _copy_public_split(public: Path, output: Path, split: str) -> int:
    count = 0
    for source_image in sorted((public / "images" / split).glob("*")):
        source_label = public / "labels" / split / f"{source_image.stem}.txt"
        if not source_label.is_file():
            raise FileNotFoundError(f"missing public label {source_label}")
        for source, destination in (
            (
                source_image,
                output / "images" / split / f"public-{source_image.name}",
            ),
            (
                source_label,
                output / "labels" / split / f"public-{source_label.name}",
            ),
        ):
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                continue
            try:
                os.link(source, destination)
            except OSError:
                shutil.copy2(source, destination)
        count += 1
    return count


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    forbidden = {str(record["sha256"]) for record in corpus["fixtures"]}
    records: list[dict[str, Any]] = []
    excluded_acceptance_frames: list[dict[str, Any]] = []

    def is_acceptance_fixture(
        *, source_id: str, frame_index: int, image: bytes
    ) -> bool:
        digest = _sha256(image)
        if digest not in forbidden:
            return False
        excluded_acceptance_frames.append(
            {
                "source_id": source_id,
                "frame_index": frame_index,
                "image_sha256": digest,
            }
        )
        return True

    with zipfile.ZipFile(args.pear_archive) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        for frame_index, frame in enumerate(manifest["frames"], start=1):
            image = archive.read(frame["filename"])
            if is_acceptance_fixture(
                source_id="pear-distill", frame_index=frame_index, image=image
            ):
                continue
            detection = frame.get("detection", {})
            box = detection.get("bbox_xyxy")
            if box is None or detection.get("label") != "pear":
                raise ValueError(f"pear frame {frame_index} lacks a pear box")
            records.append(
                _write_example(
                    args.output,
                    source_id="pear-distill",
                    frame_index=frame_index,
                    image=image,
                    class_name="pear",
                    box=list(box),
                    width=int(frame["width"]),
                    height=int(frame["height"]),
                )
            )

    with zipfile.ZipFile(args.apple_archive) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        for frame_index, frame in enumerate(manifest["frames"], start=1):
            image = archive.read(frame["filename"])
            if is_acceptance_fixture(
                source_id="green-apple", frame_index=frame_index, image=image
            ):
                continue
            records.append(
                _write_example(
                    args.output,
                    source_id="green-apple",
                    frame_index=frame_index,
                    image=image,
                    class_name="apple",
                    box=detect_green_apple_bbox(image),
                    width=int(frame["width"]),
                    height=int(frame["height"]),
                )
            )

    banana_labels = json.loads(args.banana_labels.read_text(encoding="utf-8"))
    labels_by_name = {
        str(record["filename"]): record for record in banana_labels["records"]
    }
    with zipfile.ZipFile(args.banana_archive) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        for frame_index, frame in enumerate(manifest["frames"], start=1):
            image = archive.read(frame["filename"])
            if is_acceptance_fixture(
                source_id="banana-human", frame_index=frame_index, image=image
            ):
                continue
            label = labels_by_name.get(str(frame["filename"]))
            if label is None or not label.get("reviewed"):
                raise ValueError(f"banana frame {frame_index} is not human-reviewed")
            records.append(
                _write_example(
                    args.output,
                    source_id="banana-human",
                    frame_index=frame_index,
                    image=image,
                    class_name="banana",
                    box=list(label["bbox_xyxy"]),
                    width=int(label["width"]),
                    height=int(label["height"]),
                )
            )

    leaked = sorted({record["image_sha256"] for record in records} & forbidden)
    if leaked:
        raise RuntimeError(f"acceptance corpus leaked into training data: {leaked}")

    public_counts = {
        split: _copy_public_split(args.public_dataset, args.output, split)
        for split in ("train", "val")
    }
    yaml_path = args.output / "data.yaml"
    yaml_path.write_text(
        f"path: {args.output.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: pear\n"
        "  1: apple\n"
        "  2: banana\n",
        encoding="utf-8",
    )
    result = {
        "schema_version": 1,
        "policy": "whole-fruit-only",
        "data_yaml": str(yaml_path),
        "public_images": public_counts,
        "go2_images": {
            "train": sum(record["split"] == "train" for record in records),
            "val": sum(record["split"] == "val" for record in records),
            "by_class": {
                name: sum(record["class_name"] == name for record in records)
                for name in CLASS_IDS
            },
        },
        "acceptance_fixture_leaks": leaked,
        "excluded_acceptance_frames": excluded_acceptance_frames,
        "sources": {
            "pear_archive_sha256": _sha256(args.pear_archive.read_bytes()),
            "apple_archive_sha256": _sha256(args.apple_archive.read_bytes()),
            "banana_archive_sha256": _sha256(args.banana_archive.read_bytes()),
            "banana_labels_sha256": _sha256(args.banana_labels.read_bytes()),
        },
        "records": records,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-dataset", type=Path, required=True)
    parser.add_argument("--pear-archive", type=Path, required=True)
    parser.add_argument("--apple-archive", type=Path, required=True)
    parser.add_argument("--banana-archive", type=Path, required=True)
    parser.add_argument("--banana-labels", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    prepare(parse_args())
