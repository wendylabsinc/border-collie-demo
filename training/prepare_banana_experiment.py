from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from zipfile import ZipFile


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _label_line(annotation: dict[str, object]) -> str:
    x1, y1, x2, y2 = annotation["bbox_xyxy_normalized"]
    return " ".join(
        (
            "0",
            f"{(x1 + x2) / 2:.8f}",
            f"{(y1 + y2) / 2:.8f}",
            f"{x2 - x1:.8f}",
            f"{y2 - y1:.8f}",
        )
    )


def prepare_experiment(
    curated_manifest_path: Path,
    evaluation_archive_path: Path,
    evaluation_labels_path: Path,
    output_path: Path,
    *,
    seed: int = 24601,
) -> dict[str, object]:
    curated = json.loads(curated_manifest_path.read_text())
    evaluation_labels = json.loads(evaluation_labels_path.read_text())
    records = list(curated["records"])
    ordered = sorted(
        records,
        key=lambda record: hashlib.sha256(
            f"{seed}:{record['image_id']}".encode()
        ).hexdigest(),
    )
    validation_count = max(1, round(len(ordered) * 0.2))
    validation_ids = {record["image_id"] for record in ordered[:validation_count]}
    split_counts = {"train": 0, "val": 0}
    split_boxes = {"train": 0, "val": 0}
    source_hashes: set[str] = set()

    for record in records:
        split = "val" if record["image_id"] in validation_ids else "train"
        images_dir = output_path / "images" / split
        labels_dir = output_path / "labels" / split
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        source = (curated_manifest_path.parent / str(record["local_image"])).resolve()
        source_hashes.add(_sha256(source))
        shutil.copy2(source, images_dir / source.name)
        labels = [_label_line(annotation) for annotation in record["annotations"]]
        (labels_dir / f"{record['image_id']}.txt").write_text("\n".join(labels) + "\n")
        split_counts[split] += 1
        split_boxes[split] += len(labels)

    evaluation_images = output_path / "evaluation" / "images"
    evaluation_label_dir = output_path / "evaluation" / "labels"
    evaluation_images.mkdir(parents=True, exist_ok=True)
    evaluation_label_dir.mkdir(parents=True, exist_ok=True)
    with ZipFile(evaluation_archive_path) as archive:
        archive_manifest = json.loads(archive.read("manifest.json"))
        if evaluation_labels["source_generation"] != archive_manifest["generation"]:
            raise ValueError("evaluation labels do not match evaluation archive")
        for record in evaluation_labels["records"]:
            filename = Path(record["filename"]).name
            (evaluation_images / filename).write_bytes(archive.read(record["filename"]))
            bbox = record.get("bbox_xyxy")
            if bbox is None:
                (evaluation_label_dir / f"{Path(filename).stem}.txt").write_text("")
                continue
            x1, y1, x2, y2 = bbox
            width = record["width"]
            height = record["height"]
            annotation = {
                "bbox_xyxy_normalized": [
                    x1 / width,
                    y1 / height,
                    x2 / width,
                    y2 / height,
                ]
            }
            (evaluation_label_dir / f"{Path(filename).stem}.txt").write_text(
                _label_line(annotation) + "\n"
            )

    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "data.yaml").write_text(
        f"path: {output_path.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: banana\n"
    )
    (output_path / "evaluation.yaml").write_text(
        f"path: {output_path.resolve()}\ntrain: evaluation/images\nval: evaluation/images\nnames:\n  0: banana\n"
    )
    result = {
        "schema_version": 1,
        "seed": seed,
        "curated_manifest_sha256": _sha256(curated_manifest_path),
        "evaluation_archive_sha256": _sha256(evaluation_archive_path),
        "evaluation_labels_sha256": _sha256(evaluation_labels_path),
        "training_use_allowed_for_evaluation": False,
        "split_images": split_counts,
        "split_boxes": split_boxes,
        "evaluation_frames": len(evaluation_labels["records"]),
        "unique_training_image_hashes": len(source_hashes),
    }
    (output_path / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the BANANA-001 experiment")
    parser.add_argument("--curated-manifest", required=True, type=Path)
    parser.add_argument("--evaluation-archive", required=True, type=Path)
    parser.add_argument("--evaluation-labels", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", default=24601, type=int)
    args = parser.parse_args()
    result = prepare_experiment(
        args.curated_manifest,
        args.evaluation_archive,
        args.evaluation_labels,
        args.output,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
