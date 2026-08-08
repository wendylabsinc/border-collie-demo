"""Whole-fruit dataset selection and 640px detector inputs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from torch import Tensor
from torch.utils.data import Dataset

from .model import DEFAULT_SPEC


@dataclass(frozen=True)
class FruitRecord:
    image_id: str
    image_path: Path
    boxes: tuple[tuple[float, float, float, float], ...]
    class_ids: tuple[int, ...]


def _is_validation(image_id: str, seed: int, validation_fraction: float) -> bool:
    digest = hashlib.sha256(f"{seed}:{image_id}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    return value < validation_fraction


def select_whole_fruit_records(
    manifest_path: Path,
    ranking_path: Path,
    *,
    minimum_whole_score: float,
    split: Literal["train", "val", "all"],
    seed: int = 20260806,
    validation_fraction: float = 0.15,
) -> list[FruitRecord]:
    """Select only high-ranked whole-fruit annotations with image-level splits."""

    if not 0.0 <= minimum_whole_score <= 1.0:
        raise ValueError("minimum_whole_score must be between zero and one")
    if split not in {"train", "val", "all"}:
        raise ValueError("split must be train, val, or all")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ranking = json.loads(ranking_path.read_text(encoding="utf-8"))
    scores = {
        str(candidate["candidate_id"]): float(candidate["whole_fruit_score"])
        for candidate in ranking["candidates"]
    }
    class_map = {name: index for index, name in enumerate(DEFAULT_SPEC.classes)}
    records: list[FruitRecord] = []
    for source in manifest["records"]:
        image_id = str(source["image_id"])
        is_validation = _is_validation(image_id, seed, validation_fraction)
        if split == "train" and is_validation:
            continue
        if split == "val" and not is_validation:
            continue
        boxes: list[tuple[float, float, float, float]] = []
        class_ids: list[int] = []
        for index, annotation in enumerate(source["annotations"]):
            if scores.get(f"{image_id}:{index}", 0.0) < minimum_whole_score:
                continue
            class_name = str(annotation["class_name"]).casefold().strip()
            if class_name not in class_map:
                continue
            box = tuple(float(value) for value in annotation["bbox_xyxy_normalized"])
            if len(box) != 4 or box[2] <= box[0] or box[3] <= box[1]:
                continue
            boxes.append(box)
            class_ids.append(class_map[class_name])
        if boxes:
            records.append(
                FruitRecord(
                    image_id=image_id,
                    image_path=(manifest_path.parent / str(source["local_image"])),
                    boxes=tuple(boxes),
                    class_ids=tuple(class_ids),
                )
            )
    return sorted(records, key=lambda record: record.image_id)


class WholeFruitDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    def __init__(
        self,
        records: list[FruitRecord],
        *,
        input_size: int = 640,
        augment: bool = False,
    ) -> None:
        self.records = records
        self.input_size = input_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        record = self.records[index]
        image = ImageOps.exif_transpose(Image.open(record.image_path)).convert("RGB")
        source_width, source_height = image.size
        scale = min(self.input_size / source_width, self.input_size / source_height)
        resized_width = max(1, round(source_width * scale))
        resized_height = max(1, round(source_height * scale))
        image = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
        pad_x = (self.input_size - resized_width) // 2
        pad_y = (self.input_size - resized_height) // 2
        canvas = Image.new("RGB", (self.input_size, self.input_size), (114, 114, 114))
        canvas.paste(image, (pad_x, pad_y))
        boxes = torch.tensor(record.boxes, dtype=torch.float32)
        boxes[:, (0, 2)] = boxes[:, (0, 2)] * source_width * scale + pad_x
        boxes[:, (1, 3)] = boxes[:, (1, 3)] * source_height * scale + pad_y

        if self.augment and bool(torch.rand(()) < 0.5):
            canvas = ImageOps.mirror(canvas)
            left = self.input_size - boxes[:, 2].clone()
            right = self.input_size - boxes[:, 0].clone()
            boxes[:, 0], boxes[:, 2] = left, right
        if self.augment:
            brightness = 0.85 + 0.30 * float(torch.rand(()))
            contrast = 0.85 + 0.30 * float(torch.rand(()))
            saturation = 0.85 + 0.30 * float(torch.rand(()))
            canvas = ImageEnhance.Brightness(canvas).enhance(brightness)
            canvas = ImageEnhance.Contrast(canvas).enhance(contrast)
            canvas = ImageEnhance.Color(canvas).enhance(saturation)

        array = np.asarray(canvas, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))
        return tensor, boxes, torch.tensor(record.class_ids, dtype=torch.long)

    def sampling_weights(self) -> Tensor:
        counts = torch.zeros(len(DEFAULT_SPEC.classes), dtype=torch.float64)
        for record in self.records:
            for class_id in set(record.class_ids):
                counts[class_id] += 1
        inverse = torch.where(counts > 0, 1.0 / counts, torch.zeros_like(counts))
        return torch.tensor(
            [max(float(inverse[class_id]) for class_id in record.class_ids) for record in self.records],
            dtype=torch.float64,
        )


def collate_batch(
    batch: list[tuple[Tensor, Tensor, Tensor]],
) -> tuple[Tensor, list[Tensor], list[Tensor]]:
    images, boxes, class_ids = zip(*batch, strict=True)
    return torch.stack(images), list(boxes), list(class_ids)
