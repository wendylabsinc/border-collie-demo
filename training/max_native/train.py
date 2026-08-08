"""Train the exact MAX-native fruit detector twin on whole-fruit images."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from .data import WholeFruitDataset, collate_batch, select_whole_fruit_records
from .detection import (
    align_teacher_predictions,
    detection_loss,
    distillation_loss,
    encode_targets,
)
from .model import ARCHITECTURE_SHA256, DEFAULT_SPEC, ConvNorm, TrainableFruitDetector


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_conv_norm(destination: ConvNorm, source: nn.Module) -> bool:
    children = list(source.children())
    if len(children) < 2:
        return False
    source_conv, source_norm = children[0], children[1]
    if not isinstance(source_conv, nn.Conv2d) or not isinstance(
        source_norm, nn.BatchNorm2d
    ):
        return False
    if destination.conv.weight.shape != source_conv.weight.shape:
        return False
    with torch.no_grad():
        destination.conv.weight.copy_(source_conv.weight)
        destination.norm.weight.copy_(source_norm.weight)
        destination.norm.bias.copy_(source_norm.bias)
        destination.norm.running_mean.copy_(source_norm.running_mean)
        destination.norm.running_var.copy_(source_norm.running_var)
        destination.norm.num_batches_tracked.copy_(source_norm.num_batches_tracked)
    return True


def initialize_from_mobilenet_v3_large(model: TrainableFruitDetector) -> dict[str, Any]:
    """Reuse matching ImageNet convolutions; leave changed 3x3/heads trainable."""

    from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

    source = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
    copied: list[str] = []
    skipped: list[str] = []
    if _copy_conv_norm(model.stem, source.features[0]):
        copied.append("stem")
    else:
        skipped.append("stem")
    for index, destination_block in enumerate(model.blocks):
        source_block = source.features[index + 1].block
        source_layers = []
        for candidate in source_block.children():
            children = list(candidate.children())
            if (
                len(children) >= 2
                and isinstance(children[0], nn.Conv2d)
                and isinstance(children[1], nn.BatchNorm2d)
            ):
                source_layers.append(candidate)
        destinations: list[tuple[str, ConvNorm]] = []
        if destination_block.expand is not None:
            destinations.append((f"blocks.{index}.expand", destination_block.expand))
        destinations.extend(
            (
                (f"blocks.{index}.depthwise", destination_block.depthwise),
                (f"blocks.{index}.project", destination_block.project),
            )
        )
        if len(source_layers) != len(destinations):
            raise RuntimeError(f"unexpected torchvision block layout at index {index}")
        for (name, destination), source_layer in zip(
            destinations, source_layers, strict=True
        ):
            if _copy_conv_norm(destination, source_layer):
                copied.append(name)
            else:
                skipped.append(name)
    return {
        "source": "torchvision MobileNet_V3_Large_Weights.DEFAULT",
        "copied_layers": copied,
        "skipped_layers": skipped,
        "copied_count": len(copied),
        "skipped_count": len(skipped),
    }


def _device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _class_counts(records: list[Any]) -> dict[str, int]:
    counts: Counter[int] = Counter()
    for record in records:
        counts.update(record.class_ids)
    return {
        class_name: counts[index]
        for index, class_name in enumerate(DEFAULT_SPEC.classes)
    }


def _epoch(
    model: TrainableFruitDetector,
    loader: DataLoader,
    *,
    device: torch.device,
    input_size: int,
    optimizer: torch.optim.Optimizer | None,
    teacher: nn.Module | None = None,
    distillation_weight: float = 1.0,
    teacher_box_threshold: float = 0.25,
    train_batch_norm: bool = False,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    if training and not train_batch_norm:
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
    totals = Counter()
    batches = 0
    started = time.perf_counter()
    context = torch.enable_grad if training else torch.inference_mode
    with context():
        for images, boxes, class_ids in loader:
            images = images.to(device)
            targets = encode_targets(
                boxes,
                class_ids,
                input_size=input_size,
                device=device,
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss, metrics = detection_loss(outputs, targets)
            if teacher is not None:
                with torch.no_grad():
                    teacher_result = teacher(images)
                    teacher_predictions = (
                        teacher_result[0]
                        if isinstance(teacher_result, (tuple, list))
                        else teacher_result
                    )
                dense_teacher_targets = align_teacher_predictions(
                    teacher_predictions,
                    input_size=input_size,
                )
                teacher_loss, teacher_metrics = distillation_loss(
                    outputs,
                    dense_teacher_targets,
                    input_size=input_size,
                    box_confidence_threshold=teacher_box_threshold,
                )
                loss = loss + distillation_weight * teacher_loss
                metrics = {
                    **metrics,
                    **teacher_metrics,
                    "supervised_loss": metrics["loss"],
                    "loss": float(loss.detach().cpu()),
                }
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite training loss: {float(loss)}")
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()
            for name, value in metrics.items():
                totals[name] += value
            batches += 1
    elapsed = time.perf_counter() - started
    return {
        **{name: value / max(1, batches) for name, value in totals.items()},
        "batches": float(batches),
        "seconds": elapsed,
        "images_per_second": len(loader.dataset) / max(elapsed, 1e-9),
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = _device(args.device)
    train_records = select_whole_fruit_records(
        args.manifest,
        args.ranking,
        minimum_whole_score=args.minimum_whole_score,
        split="train",
        seed=args.seed,
        validation_fraction=args.validation_fraction,
    )
    validation_records = select_whole_fruit_records(
        args.manifest,
        args.ranking,
        minimum_whole_score=args.minimum_whole_score,
        split="val",
        seed=args.seed,
        validation_fraction=args.validation_fraction,
    )
    if not train_records or not validation_records:
        raise RuntimeError("whole-fruit selection produced an empty train or validation split")
    train_dataset = WholeFruitDataset(
        train_records, input_size=args.input_size, augment=True
    )
    validation_dataset = WholeFruitDataset(
        validation_records, input_size=args.input_size, augment=False
    )
    generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(
        train_dataset.sampling_weights(),
        num_samples=len(train_dataset),
        replacement=True,
        generator=generator,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.workers,
        collate_fn=collate_batch,
        persistent_workers=args.workers > 0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_batch,
        persistent_workers=args.workers > 0,
    )
    model = TrainableFruitDetector(DEFAULT_SPEC)
    if args.initial_checkpoint is not None:
        initial_checkpoint = torch.load(
            args.initial_checkpoint, map_location="cpu", weights_only=True
        )
        if initial_checkpoint["architecture_sha256"] != ARCHITECTURE_SHA256:
            raise RuntimeError("initial checkpoint does not match the MAX architecture")
        if tuple(initial_checkpoint["classes"]) != DEFAULT_SPEC.classes:
            raise RuntimeError("initial checkpoint class map does not match")
        if int(initial_checkpoint["input_size"]) != args.input_size:
            raise RuntimeError("initial checkpoint input size does not match")
        model.load_state_dict(initial_checkpoint["model"])
        initialization = {
            "source": "checkpoint",
            "checkpoint": str(args.initial_checkpoint),
            "checkpoint_sha256": _sha256(args.initial_checkpoint),
            "checkpoint_epoch": int(initial_checkpoint["epoch"]),
        }
    else:
        initialization = (
            initialize_from_mobilenet_v3_large(model)
            if not args.random_initialization
            else {"source": "random", "copied_count": 0, "skipped_count": 45}
        )
    model.to(device)
    teacher = None
    teacher_metadata: dict[str, Any] | None = None
    if args.teacher_checkpoint is not None:
        from ultralytics import YOLO

        teacher_wrapper = YOLO(str(args.teacher_checkpoint))
        teacher_names = tuple(
            str(teacher_wrapper.names[index])
            for index in sorted(teacher_wrapper.names)
        )
        if teacher_names != DEFAULT_SPEC.classes:
            raise RuntimeError(
                f"teacher class map {teacher_names!r} does not match {DEFAULT_SPEC.classes!r}"
            )
        teacher = teacher_wrapper.model.to(device).eval()
        teacher.requires_grad_(False)
        teacher_metadata = {
            "checkpoint": str(args.teacher_checkpoint),
            "checkpoint_sha256": _sha256(args.teacher_checkpoint),
            "classes": list(teacher_names),
            "distillation_weight": args.distillation_weight,
            "box_confidence_threshold": args.teacher_box_threshold,
            "dense_anchor_contract": "stride-8, stride-16, stride-32",
        }
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs), eta_min=args.learning_rate * 0.05
    )
    args.output.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best_validation_loss = float("inf")
    best_path = args.output / "fruit-native-640-best.pt"
    last_path = args.output / "fruit-native-640-last.pt"
    training_started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_metrics = _epoch(
            model,
            train_loader,
            device=device,
            input_size=args.input_size,
            optimizer=optimizer,
            teacher=teacher,
            distillation_weight=args.distillation_weight,
            teacher_box_threshold=args.teacher_box_threshold,
            train_batch_norm=args.train_batch_norm,
        )
        validation_metrics = _epoch(
            model,
            validation_loader,
            device=device,
            input_size=args.input_size,
            optimizer=None,
            teacher=teacher,
            distillation_weight=args.distillation_weight,
            teacher_box_threshold=args.teacher_box_threshold,
            train_batch_norm=args.train_batch_norm,
        )
        record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if validation_metrics["loss"] < best_validation_loss:
            best_validation_loss = validation_metrics["loss"]
            torch.save(
                {
                    "architecture_sha256": ARCHITECTURE_SHA256,
                    "classes": DEFAULT_SPEC.classes,
                    "input_size": args.input_size,
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "initialization": initialization,
                },
                best_path,
            )
        torch.save(
            {
                "architecture_sha256": ARCHITECTURE_SHA256,
                "classes": DEFAULT_SPEC.classes,
                "input_size": args.input_size,
                "epoch": epoch,
                "model": model.state_dict(),
                "initialization": initialization,
            },
            last_path,
        )
        scheduler.step()
        (args.output / "history.json").write_text(
            json.dumps(history, indent=2) + "\n", encoding="utf-8"
        )
    result = {
        "schema_version": 1,
        "architecture_sha256": ARCHITECTURE_SHA256,
        "classes": list(DEFAULT_SPEC.classes),
        "input_size": args.input_size,
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "minimum_whole_score": args.minimum_whole_score,
        "manifest_sha256": _sha256(args.manifest),
        "ranking_sha256": _sha256(args.ranking),
        "train_images": len(train_records),
        "validation_images": len(validation_records),
        "train_boxes": _class_counts(train_records),
        "validation_boxes": _class_counts(validation_records),
        "initialization": initialization,
        "teacher": teacher_metadata,
        "batch_norm_statistics": (
            "trained" if args.train_batch_norm else "frozen"
        ),
        "best_validation_loss": best_validation_loss,
        "training_seconds": time.perf_counter() - training_started,
        "checkpoint": str(best_path),
        "checkpoint_sha256": _sha256(best_path),
        "last_checkpoint": str(last_path),
        "last_checkpoint_sha256": _sha256(last_path),
        "history": history,
        "experimental_only": True,
        "production_blocker": (
            "automated whole-fruit ranking and independent source-image rights "
            "verification are not complete"
        ),
    }
    (args.output / "training-result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(f"MAX_NATIVE_TRAINING_RESULT={json.dumps(result, sort_keys=True)}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--ranking", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--minimum-whole-score", type=float, default=0.75)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--random-initialization", action="store_true")
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--teacher-checkpoint", type=Path)
    parser.add_argument("--distillation-weight", type=float, default=4.0)
    parser.add_argument("--teacher-box-threshold", type=float, default=0.25)
    parser.add_argument("--train-batch-norm", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
