from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import open_clip
import torch
from PIL import Image

POSITIVE_PROMPTS = (
    "a photo of one whole intact uncut {fruit}",
    "a photo of a whole unpeeled {fruit} used as a physical prop",
    "a complete {fruit} with its full outer shape visible",
)
NEGATIVE_PROMPTS = (
    "a photo of sliced or cut {fruit} pieces",
    "a photo of peeled or partly eaten {fruit}",
    "a photo of {fruit} in a cooked dish, dessert, or drink",
    "a photo of a dense bunch or pile of {fruit}",
    "a drawing, screen, logo, or package showing {fruit}",
    "a partial fragment of {fruit} without the whole fruit shape",
)


def _crop(image: Image.Image, bbox: list[float], padding: float = 0.08) -> Image.Image:
    width, height = image.size
    x1, y1, x2, y2 = bbox
    box_width = (x2 - x1) * width
    box_height = (y2 - y1) * height
    return image.crop(
        (
            max(0, x1 * width - box_width * padding),
            max(0, y1 * height - box_height * padding),
            min(width, x2 * width + box_width * padding),
            min(height, y2 * height + box_height * padding),
        )
    )


def _whole_probability(logits: torch.Tensor) -> torch.Tensor:
    positive = torch.logsumexp(logits[:, : len(POSITIVE_PROMPTS)], dim=1) - math.log(
        len(POSITIVE_PROMPTS)
    )
    negative = torch.logsumexp(logits[:, len(POSITIVE_PROMPTS) :], dim=1) - math.log(
        len(NEGATIVE_PROMPTS)
    )
    return torch.stack((positive, negative), dim=1).softmax(dim=1)[:, 0]


def _quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.floor(fraction * len(ordered)))]


def rank_manifest(
    manifest_path: Path,
    output_path: Path,
    *,
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
    batch_size: int = 64,
) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text())
    dataset_root = manifest_path.parent
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)

    prompt_features: dict[str, torch.Tensor] = {}
    class_names = sorted(
        {
            str(annotation["class_name"])
            for record in manifest["records"]
            for annotation in record["annotations"]
        }
    )
    with torch.inference_mode():
        for class_name in class_names:
            prompts = [
                template.format(fruit=class_name)
                for template in (*POSITIVE_PROMPTS, *NEGATIVE_PROMPTS)
            ]
            features = model.encode_text(tokenizer(prompts).to(device))
            prompt_features[class_name] = features / features.norm(dim=-1, keepdim=True)

    candidates: list[dict[str, object]] = []
    pending_images: list[torch.Tensor] = []
    pending_records: list[dict[str, object]] = []

    def flush() -> None:
        if not pending_images:
            return
        images = torch.stack(pending_images).to(device)
        with torch.inference_mode():
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            for index, record in enumerate(pending_records):
                text_features = prompt_features[str(record["class_name"])]
                logits = 100 * image_features[index : index + 1] @ text_features.T
                score = float(_whole_probability(logits)[0].cpu())
                candidates.append({**record, "whole_fruit_score": score})
        pending_images.clear()
        pending_records.clear()

    for record in manifest["records"]:
        image = Image.open(dataset_root / record["local_image"]).convert("RGB")
        for annotation_index, annotation in enumerate(record["annotations"]):
            crop = _crop(image, annotation["bbox_xyxy_normalized"])
            pending_images.append(preprocess(crop))
            pending_records.append(
                {
                    "candidate_id": f"{record['image_id']}:{annotation_index}",
                    "image_id": record["image_id"],
                    "local_image": record["local_image"],
                    "class_id": annotation["class_id"],
                    "class_name": annotation["class_name"],
                    "bbox_xyxy_normalized": annotation["bbox_xyxy_normalized"],
                }
            )
            if len(pending_images) >= batch_size:
                flush()
    flush()

    scores_by_class: dict[str, list[float]] = defaultdict(list)
    for candidate in candidates:
        scores_by_class[str(candidate["class_name"])].append(
            float(candidate["whole_fruit_score"])
        )
    summary = {
        class_name: {
            "candidates": len(scores),
            "score_minimum": min(scores),
            "score_p25": _quantile(scores, 0.25),
            "score_median": _quantile(scores, 0.5),
            "score_p75": _quantile(scores, 0.75),
            "score_maximum": max(scores),
        }
        for class_name, scores in sorted(scores_by_class.items())
    }
    result = {
        "schema_version": 1,
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": __import__("hashlib")
        .sha256(manifest_path.read_bytes())
        .hexdigest(),
        "purpose": "whole-fruit review ranking only",
        "training_eligible": False,
        "ranking_model": {
            "library": "open-clip-torch",
            "model": model_name,
            "pretrained": pretrained,
            "device": device,
            "positive_prompts": list(POSITIVE_PROMPTS),
            "negative_prompts": list(NEGATIVE_PROMPTS),
        },
        "summary": summary,
        "candidates": sorted(
            candidates,
            key=lambda candidate: (
                str(candidate["class_name"]),
                -float(candidate["whole_fruit_score"]),
            ),
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rank Open Images crops for whole-fruit review"
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", default=64, type=int)
    args = parser.parse_args()
    result = rank_manifest(
        args.manifest,
        args.output,
        batch_size=args.batch_size,
    )
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
