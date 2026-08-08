"""Materialize the strict whole-fruit split for a pretrained YOLO fine-tune."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from .data import FruitRecord, select_whole_fruit_records
from .model import DEFAULT_SPEC


def _label_line(class_id: int, box: tuple[float, float, float, float]) -> str:
    x1, y1, x2, y2 = box
    return (
        f"{class_id} {(x1 + x2) / 2.0:.8f} {(y1 + y2) / 2.0:.8f} "
        f"{x2 - x1:.8f} {y2 - y1:.8f}"
    )


def _materialize(records: list[FruitRecord], output: Path, split: str) -> int:
    image_dir = output / "images" / split
    label_dir = output / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    boxes = 0
    for record in records:
        destination = image_dir / f"{record.image_id}{record.image_path.suffix.lower()}"
        if not destination.exists():
            try:
                os.link(record.image_path, destination)
            except OSError:
                shutil.copy2(record.image_path, destination)
        lines = [
            _label_line(class_id, box)
            for class_id, box in zip(record.class_ids, record.boxes, strict=True)
        ]
        (label_dir / f"{record.image_id}.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        boxes += len(lines)
    return boxes


def prepare(args: argparse.Namespace) -> dict[str, object]:
    common = {
        "minimum_whole_score": args.minimum_whole_score,
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
    }
    train = select_whole_fruit_records(
        args.manifest, args.ranking, split="train", **common
    )
    validation = select_whole_fruit_records(
        args.manifest, args.ranking, split="val", **common
    )
    args.output.mkdir(parents=True, exist_ok=True)
    train_boxes = _materialize(train, args.output, "train")
    validation_boxes = _materialize(validation, args.output, "val")
    yaml_path = args.output / "data.yaml"
    names = "\n".join(
        f"  {index}: {name}" for index, name in enumerate(DEFAULT_SPEC.classes)
    )
    yaml_path.write_text(
        f"path: {args.output.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        f"{names}\n",
        encoding="utf-8",
    )
    result = {
        "schema_version": 1,
        "classes": list(DEFAULT_SPEC.classes),
        "minimum_whole_score": args.minimum_whole_score,
        "train_images": len(train),
        "train_boxes": train_boxes,
        "validation_images": len(validation),
        "validation_boxes": validation_boxes,
        "data_yaml": str(yaml_path),
        "experimental_only": True,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--ranking", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--minimum-whole-score", type=float, default=0.75)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260806)
    return parser.parse_args()


if __name__ == "__main__":
    prepare(parse_args())
