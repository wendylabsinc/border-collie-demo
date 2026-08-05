from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

FINAL_DECISIONS = {
    "accept_whole",
    "reject_cut_or_food",
    "reject_bunch",
    "reject_bad_box",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_yolo_labels(path: Path, annotations: list[dict[str, object]]) -> None:
    lines = []
    for annotation in annotations:
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
    path.write_text("\n".join(lines) + "\n")


def apply_review(
    manifest_path: Path,
    ranking_path: Path,
    review_path: Path,
    output_path: Path,
) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text())
    ranking = json.loads(ranking_path.read_text())
    review = json.loads(review_path.read_text())
    manifest_hash = _sha256(manifest_path)
    if ranking.get("source_manifest_sha256") != manifest_hash:
        raise ValueError("ranking does not match the source manifest")
    if review.get("source_manifest_sha256") != manifest_hash:
        raise ValueError("review does not match the source manifest")
    target_class = str(review.get("review_class") or "").casefold()
    if not target_class:
        raise ValueError("review class is missing")
    decisions = {
        candidate_id: value.get("decision")
        for candidate_id, value in review.get("decisions", {}).items()
    }
    valid_candidates = {
        str(candidate["candidate_id"]): candidate
        for candidate in ranking["candidates"]
        if str(candidate["class_name"]).casefold() == target_class
    }
    if any(candidate_id not in valid_candidates for candidate_id in decisions):
        raise ValueError("review contains an unknown candidate")

    images_dir = output_path / "images"
    labels_dir = output_path / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    included = []
    withheld: Counter[str] = Counter()
    hard_negative_reasons: Counter[str] = Counter()
    accepted_boxes = 0
    positive_images = 0
    hard_negative_images = 0

    for record in manifest["records"]:
        annotations = record["annotations"]
        target_indices = [
            index
            for index, annotation in enumerate(annotations)
            if str(annotation["class_name"]).casefold() == target_class
        ]
        candidate_ids = [f"{record['image_id']}:{index}" for index in target_indices]
        states = [decisions.get(candidate_id) for candidate_id in candidate_ids]
        if not any(state in FINAL_DECISIONS for state in states):
            continue
        if any(
            str(annotation["class_name"]).casefold() != target_class
            for annotation in annotations
        ):
            withheld["contains_unreviewed_target_class"] += 1
            continue
        if any(
            decisions.get(candidate_id) not in FINAL_DECISIONS
            for candidate_id in candidate_ids
        ):
            withheld["incomplete_image_review"] += 1
            continue

        accepted = [
            annotations[index]
            for index, candidate_id in zip(target_indices, candidate_ids, strict=True)
            if decisions[candidate_id] == "accept_whole"
        ]
        if not accepted and not any(
            str(state).startswith("reject_") for state in states
        ):
            continue
        source_image = (manifest_path.parent / str(record["local_image"])).resolve()
        destination_image = images_dir / source_image.name
        shutil.copy2(source_image, destination_image)
        _write_yolo_labels(labels_dir / f"{record['image_id']}.txt", accepted)
        accepted_boxes += len(accepted)
        if accepted:
            positive_images += 1
            record_role = "whole_fruit_positive"
        else:
            hard_negative_images += 1
            record_role = "reviewed_non_whole_hard_negative"
            hard_negative_reasons.update(str(state) for state in states)
        included.append(
            {
                **record,
                "local_image": f"images/{destination_image.name}",
                "local_label": f"labels/{record['image_id']}.txt",
                "record_role": record_role,
                "annotations": accepted,
            }
        )

    result = {
        "schema_version": 1,
        "source_manifest_sha256": manifest_hash,
        "ranking_sha256": _sha256(ranking_path),
        "review_sha256": _sha256(review_path),
        "target_class": target_class,
        "content_policy": "training/whole-fruit-policy.md",
        "content_review_complete": True,
        "experimental_training_eligible": True,
        "production_training_eligible": False,
        "production_blocker": "independent source-image rights verification remains incomplete",
        "counts": {
            "images": len(included),
            "positive_images": positive_images,
            "hard_negative_images": hard_negative_images,
            "accepted_boxes": accepted_boxes,
            "hard_negative_boxes_by_reason": dict(
                sorted(hard_negative_reasons.items())
            ),
            "withheld_images_by_reason": dict(sorted(withheld.items())),
        },
        "records": included,
    }
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    shutil.copy2(review_path, output_path / "review.json")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply whole-fruit review decisions")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--ranking", required=True, type=Path)
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = apply_review(args.manifest, args.ranking, args.review, args.output)
    print(json.dumps(result["counts"], indent=2))


if __name__ == "__main__":
    main()
