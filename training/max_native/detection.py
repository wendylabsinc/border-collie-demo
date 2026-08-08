"""Shared training and deployment box contract for the MAX-native detector."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from .model import DEFAULT_SPEC


@dataclass(frozen=True)
class GridTargets:
    boxes: Tensor
    objectness: Tensor
    classes: Tensor
    positive: Tensor


@dataclass(frozen=True)
class Detections:
    boxes: Tensor
    scores: Tensor
    class_ids: Tensor


@dataclass(frozen=True)
class DenseTeacherTargets:
    """YOLO teacher output reshaped onto one MAX-native detection grid."""

    boxes_xywh: Tensor
    class_scores: Tensor


def align_teacher_predictions(
    predictions: Tensor,
    *,
    input_size: int,
    class_count: int = len(DEFAULT_SPEC.classes),
) -> tuple[DenseTeacherTargets, ...]:
    """Split decoded YOLO predictions into the student's three spatial grids.

    Ultralytics' decoded detector output is ``[batch, 4 + classes, anchors]``.
    Its anchors are concatenated in stride-8, stride-16, stride-32 order, the
    same order used by the MAX-native detector. Boxes remain pixel-space xywh;
    this avoids forcing a YOLO box into the student's more limited raw encoding.
    """

    if predictions.ndim != 3:
        raise ValueError("teacher predictions must have shape [batch, channels, anchors]")
    if predictions.shape[1] != 4 + class_count:
        raise ValueError("teacher prediction channels do not match the class map")
    grid_sizes = tuple(input_size // stride for stride in DEFAULT_SPEC.head_strides)
    expected_anchors = sum(grid * grid for grid in grid_sizes)
    if predictions.shape[2] != expected_anchors:
        raise ValueError(
            f"teacher produced {predictions.shape[2]} anchors; expected {expected_anchors}"
        )
    aligned: list[DenseTeacherTargets] = []
    offset = 0
    for grid in grid_sizes:
        count = grid * grid
        scale = predictions[:, :, offset : offset + count]
        aligned.append(
            DenseTeacherTargets(
                boxes_xywh=scale[:, :4].reshape(-1, 4, grid, grid),
                class_scores=scale[:, 4:].reshape(-1, class_count, grid, grid),
            )
        )
        offset += count
    return tuple(aligned)


def _student_boxes_xyxy(output: Tensor, *, stride: int) -> Tensor:
    _, _, height, width = output.shape
    y, x = torch.meshgrid(
        torch.arange(height, device=output.device, dtype=output.dtype),
        torch.arange(width, device=output.device, dtype=output.dtype),
        indexing="ij",
    )
    center_x = (x + 0.5) * stride
    center_y = (y + 0.5) * stride
    distances = output[:, :4].clamp(-6.0, 6.0).exp() * stride * 2.0
    return torch.stack(
        (
            center_x - distances[:, 0],
            center_y - distances[:, 1],
            center_x + distances[:, 2],
            center_y + distances[:, 3],
        ),
        dim=1,
    )


def _student_boxes_xywh(output: Tensor, *, stride: int) -> Tensor:
    boxes = _student_boxes_xyxy(output, stride=stride)
    return torch.stack(
        (
            (boxes[:, 0] + boxes[:, 2]) / 2.0,
            (boxes[:, 1] + boxes[:, 3]) / 2.0,
            boxes[:, 2] - boxes[:, 0],
            boxes[:, 3] - boxes[:, 1],
        ),
        dim=1,
    )


def distillation_loss(
    outputs: Sequence[Tensor],
    targets: Sequence[DenseTeacherTargets],
    *,
    input_size: int,
    box_confidence_threshold: float = 0.25,
) -> tuple[Tensor, dict[str, float]]:
    """Teach the MAX-native heads to reproduce the stronger YOLO detector.

    Scores are distilled densely, including background anchors. Box geometry is
    distilled only where the teacher has a credible fruit prediction.
    """

    if len(outputs) != len(DEFAULT_SPEC.head_strides) or len(outputs) != len(targets):
        raise ValueError("student and teacher must contain the same three scales")
    score_loss = outputs[0].new_zeros(())
    box_loss = outputs[0].new_zeros(())
    teacher_positive_anchors = 0
    scale_count = 0
    for stride, output, target in zip(
        DEFAULT_SPEC.head_strides, outputs, targets, strict=True
    ):
        teacher_scores = target.class_scores.to(output.device, output.dtype).clamp(
            0.0, 1.0
        )
        if teacher_scores.shape != output[:, 5:].shape:
            raise ValueError("teacher score grid does not match the student head")
        student_scores = output[:, 4:5].sigmoid() * output[:, 5:].sigmoid()
        confidence = teacher_scores.max(dim=1, keepdim=True).values
        score_error = F.binary_cross_entropy(
            student_scores.clamp(1e-6, 1.0 - 1e-6),
            teacher_scores,
            reduction="none",
        )
        focal_weight = (student_scores - teacher_scores).abs().pow(2.0)
        score_loss = score_loss + (
            score_error * focal_weight * (1.0 + 4.0 * confidence)
        ).mean()

        box_mask = confidence[:, 0] >= box_confidence_threshold
        teacher_positive_anchors += int(box_mask.sum().detach().cpu())
        if box_mask.any():
            student_boxes = _student_boxes_xywh(output, stride=stride)
            teacher_boxes = target.boxes_xywh.to(output.device, output.dtype)
            per_coordinate = F.smooth_l1_loss(
                student_boxes / float(input_size),
                teacher_boxes / float(input_size),
                reduction="none",
                beta=0.02,
            )
            per_anchor = per_coordinate.permute(0, 2, 3, 1)[box_mask].mean(dim=1)
            weights = confidence[:, 0][box_mask]
            box_loss = box_loss + (per_anchor * weights).sum() / weights.sum().clamp_min(
                1e-6
            )
        scale_count += 1
    score_loss = score_loss / max(1, scale_count)
    box_loss = box_loss / max(1, scale_count)
    total = score_loss + 2.0 * box_loss
    return total, {
        "distillation_loss": float(total.detach().cpu()),
        "distillation_score_loss": float(score_loss.detach().cpu()),
        "distillation_box_loss": float(box_loss.detach().cpu()),
        "teacher_positive_anchors": float(teacher_positive_anchors),
    }


def _stride_for_box(width: float, height: float) -> int:
    longest = max(width, height)
    if longest <= 96.0:
        return 8
    if longest <= 224.0:
        return 16
    return 32


def encode_targets(
    boxes: Sequence[Tensor],
    class_ids: Sequence[Tensor],
    *,
    input_size: int,
    class_count: int = len(DEFAULT_SPEC.classes),
    device: torch.device | str | None = None,
) -> tuple[GridTargets, ...]:
    """Encode pixel-space xyxy boxes into the three raw-head conventions."""

    if len(boxes) != len(class_ids):
        raise ValueError("boxes and class_ids must contain the same batch size")
    batch_size = len(boxes)
    resolved_device = torch.device(device or "cpu")
    encoded: list[GridTargets] = []
    for stride in DEFAULT_SPEC.head_strides:
        grid = input_size // stride
        encoded.append(
            GridTargets(
                boxes=torch.zeros(
                    (batch_size, 4, grid, grid),
                    dtype=torch.float32,
                    device=resolved_device,
                ),
                objectness=torch.zeros(
                    (batch_size, 1, grid, grid),
                    dtype=torch.float32,
                    device=resolved_device,
                ),
                classes=torch.zeros(
                    (batch_size, class_count, grid, grid),
                    dtype=torch.float32,
                    device=resolved_device,
                ),
                positive=torch.zeros(
                    (batch_size, grid, grid),
                    dtype=torch.bool,
                    device=resolved_device,
                ),
            )
        )
    by_stride = dict(zip(DEFAULT_SPEC.head_strides, encoded, strict=True))

    for batch_index, (image_boxes, image_classes) in enumerate(
        zip(boxes, class_ids, strict=True)
    ):
        if image_boxes.ndim != 2 or image_boxes.shape[-1] != 4:
            raise ValueError("each boxes tensor must have shape [N, 4]")
        if len(image_boxes) != len(image_classes):
            raise ValueError("each image must have one class ID per box")
        for box, class_id_value in zip(image_boxes, image_classes, strict=True):
            x1, y1, x2, y2 = (float(value) for value in box.tolist())
            x1 = min(float(input_size), max(0.0, x1))
            y1 = min(float(input_size), max(0.0, y1))
            x2 = min(float(input_size), max(0.0, x2))
            y2 = min(float(input_size), max(0.0, y2))
            width = x2 - x1
            height = y2 - y1
            if width <= 1.0 or height <= 1.0:
                continue
            class_id = int(class_id_value)
            if not 0 <= class_id < class_count:
                raise ValueError(f"class ID {class_id} is outside the model class map")
            stride = _stride_for_box(width, height)
            target = by_stride[stride]
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            grid_x = min(target.positive.shape[2] - 1, int(center_x / stride))
            grid_y = min(target.positive.shape[1] - 1, int(center_y / stride))
            radius = min(
                4,
                max(1, round(min(width, height) / (2.0 * stride))),
            )
            sigma = max(0.5, radius / 2.0)
            for heat_y in range(
                max(0, grid_y - radius),
                min(target.positive.shape[1], grid_y + radius + 1),
            ):
                for heat_x in range(
                    max(0, grid_x - radius),
                    min(target.positive.shape[2], grid_x + radius + 1),
                ):
                    squared_distance = (heat_x - grid_x) ** 2 + (
                        heat_y - grid_y
                    ) ** 2
                    heat = math.exp(-squared_distance / (2.0 * sigma * sigma))
                    target.objectness[batch_index, 0, heat_y, heat_x] = max(
                        float(target.objectness[batch_index, 0, heat_y, heat_x]),
                        heat,
                    )
            positive_cells: list[tuple[int, int]] = []
            for candidate_y in range(
                max(0, grid_y - 1),
                min(target.positive.shape[1], grid_y + 2),
            ):
                for candidate_x in range(
                    max(0, grid_x - 1),
                    min(target.positive.shape[2], grid_x + 2),
                ):
                    anchor_x = (candidate_x + 0.5) * stride
                    anchor_y = (candidate_y + 0.5) * stride
                    if x1 < anchor_x < x2 and y1 < anchor_y < y2:
                        positive_cells.append((candidate_y, candidate_x))
            if not positive_cells:
                # Extremely small boxes can fall between feature-grid centers.
                # Keeping them out of regression is safer than creating an
                # impossible negative left/top/right/bottom distance target.
                continue
            for positive_y, positive_x in positive_cells:
                if target.positive[batch_index, positive_y, positive_x]:
                    continue
                anchor_x = (positive_x + 0.5) * stride
                anchor_y = (positive_y + 0.5) * stride
                target.boxes[batch_index, :, positive_y, positive_x] = torch.tensor(
                    (
                        math.log((anchor_x - x1) / (stride * 2.0)),
                        math.log((anchor_y - y1) / (stride * 2.0)),
                        math.log((x2 - anchor_x) / (stride * 2.0)),
                        math.log((y2 - anchor_y) / (stride * 2.0)),
                    ),
                    device=resolved_device,
                )
                target.classes[batch_index, class_id, positive_y, positive_x] = 1.0
                target.positive[batch_index, positive_y, positive_x] = True
    return tuple(encoded)


def _pairwise_iou(first: Tensor, others: Tensor) -> Tensor:
    left = torch.maximum(first[0], others[:, 0])
    top = torch.maximum(first[1], others[:, 1])
    right = torch.minimum(first[2], others[:, 2])
    bottom = torch.minimum(first[3], others[:, 3])
    intersection = (right - left).clamp(min=0) * (bottom - top).clamp(min=0)
    first_area = (first[2] - first[0]).clamp(min=0) * (
        first[3] - first[1]
    ).clamp(min=0)
    other_area = (others[:, 2] - others[:, 0]).clamp(min=0) * (
        others[:, 3] - others[:, 1]
    ).clamp(min=0)
    union = first_area + other_area - intersection
    return torch.where(union > 0, intersection / union, torch.zeros_like(union))


def aligned_generalized_iou_loss(
    predicted: Tensor,
    target: Tensor,
    *,
    reduction: str = "mean",
) -> Tensor:
    """Generalized-IoU loss for aligned xyxy boxes."""

    if predicted.shape != target.shape or predicted.ndim != 2 or predicted.shape[1] != 4:
        raise ValueError("predicted and target boxes must both have shape [N, 4]")
    left = torch.maximum(predicted[:, 0], target[:, 0])
    top = torch.maximum(predicted[:, 1], target[:, 1])
    right = torch.minimum(predicted[:, 2], target[:, 2])
    bottom = torch.minimum(predicted[:, 3], target[:, 3])
    intersection = (right - left).clamp(min=0) * (bottom - top).clamp(min=0)
    predicted_area = (predicted[:, 2] - predicted[:, 0]).clamp(min=0) * (
        predicted[:, 3] - predicted[:, 1]
    ).clamp(min=0)
    target_area = (target[:, 2] - target[:, 0]).clamp(min=0) * (
        target[:, 3] - target[:, 1]
    ).clamp(min=0)
    union = predicted_area + target_area - intersection
    iou = intersection / union.clamp_min(1e-7)
    enclosing_left = torch.minimum(predicted[:, 0], target[:, 0])
    enclosing_top = torch.minimum(predicted[:, 1], target[:, 1])
    enclosing_right = torch.maximum(predicted[:, 2], target[:, 2])
    enclosing_bottom = torch.maximum(predicted[:, 3], target[:, 3])
    enclosing_area = (enclosing_right - enclosing_left).clamp(min=0) * (
        enclosing_bottom - enclosing_top
    ).clamp(min=0)
    generalized_iou = iou - (enclosing_area - union) / enclosing_area.clamp_min(1e-7)
    loss = 1.0 - generalized_iou
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError("reduction must be mean, sum, or none")


def _nms(boxes: Tensor, scores: Tensor, threshold: float) -> Tensor:
    order = scores.argsort(descending=True)
    kept: list[Tensor] = []
    while order.numel():
        current = order[0]
        kept.append(current)
        if order.numel() == 1:
            break
        remaining = order[1:]
        order = remaining[
            _pairwise_iou(boxes[current], boxes[remaining]) <= threshold
        ]
    if not kept:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    return torch.stack(kept)


def decode_predictions(
    outputs: Sequence[Tensor],
    *,
    input_size: int,
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    max_detections: int = 100,
) -> list[Detections]:
    """Decode raw NCHW heads into pixel-space boxes, scores, and class IDs."""

    if len(outputs) != len(DEFAULT_SPEC.head_strides):
        raise ValueError("detector must produce exactly three detection heads")
    decoded_boxes: list[Tensor] = []
    decoded_scores: list[Tensor] = []
    decoded_classes: list[Tensor] = []
    batch_size = int(outputs[0].shape[0])
    for stride, output in zip(DEFAULT_SPEC.head_strides, outputs, strict=True):
        if output.ndim != 4 or output.shape[1] != DEFAULT_SPEC.head_channels:
            raise ValueError("raw detection heads must be NCHW with eight channels")
        _, _, height, width = output.shape
        if height != input_size // stride or width != input_size // stride:
            raise ValueError(f"stride-{stride} head has an unexpected spatial shape")
        y, x = torch.meshgrid(
            torch.arange(height, device=output.device, dtype=output.dtype),
            torch.arange(width, device=output.device, dtype=output.dtype),
            indexing="ij",
        )
        boxes = _student_boxes_xyxy(output, stride=stride).permute(0, 2, 3, 1)
        boxes = boxes.reshape(batch_size, -1, 4)
        objectness = output[:, 4].sigmoid()
        class_scores, classes = output[:, 5:].sigmoid().max(dim=1)
        scores = objectness * class_scores
        decoded_boxes.append(boxes)
        decoded_scores.append(scores.reshape(batch_size, -1))
        decoded_classes.append(classes.reshape(batch_size, -1))

    all_boxes = torch.cat(decoded_boxes, dim=1)
    all_scores = torch.cat(decoded_scores, dim=1)
    all_classes = torch.cat(decoded_classes, dim=1)
    results: list[Detections] = []
    for batch_index in range(batch_size):
        keep = all_scores[batch_index] >= confidence_threshold
        boxes = all_boxes[batch_index][keep].clamp(0.0, float(input_size))
        scores = all_scores[batch_index][keep]
        classes = all_classes[batch_index][keep]
        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes, scores, classes = boxes[valid], scores[valid], classes[valid]
        class_kept: list[Tensor] = []
        for class_id in classes.unique(sorted=True):
            positions = (classes == class_id).nonzero(as_tuple=False).flatten()
            selected = _nms(boxes[positions], scores[positions], iou_threshold)
            class_kept.append(positions[selected])
        if class_kept:
            selected = torch.cat(class_kept)
            selected = selected[scores[selected].argsort(descending=True)][
                :max_detections
            ]
        else:
            selected = torch.empty((0,), dtype=torch.long, device=boxes.device)
        results.append(
            Detections(
                boxes=boxes[selected],
                scores=scores[selected],
                class_ids=classes[selected].to(torch.long),
            )
        )
    return results


def detection_loss(
    outputs: Sequence[Tensor], targets: Sequence[GridTargets]
) -> tuple[Tensor, dict[str, float]]:
    """Focal classification plus encoded-box regression loss."""

    if len(outputs) != len(targets):
        raise ValueError("outputs and targets must contain the same detection scales")
    positive_count = sum(int(target.positive.sum()) for target in targets)
    normalizer = float(max(1, positive_count))
    classification = outputs[0].new_zeros(())
    encoded_regression = outputs[0].new_zeros(())
    overlap_regression = outputs[0].new_zeros(())
    positive_confidence = outputs[0].new_zeros(())
    positive_correct = outputs[0].new_zeros(())
    measured_positives = 0
    background_above_threshold = 0
    for stride, output, target in zip(
        DEFAULT_SPEC.head_strides, outputs, targets, strict=True
    ):
        objectness_logits = output[:, 4:5]
        class_logits = output[:, 5:]
        objectness_target = target.objectness.to(output.device, output.dtype)
        class_target = target.classes.to(output.device, output.dtype)
        objectness_probability = objectness_logits.sigmoid().clamp(
            1e-6, 1.0 - 1e-6
        )
        peaks = objectness_target.eq(1.0)
        non_peaks = ~peaks
        positive_heatmap_loss = (
            objectness_probability.log()
            * (1.0 - objectness_probability).pow(2.0)
            * peaks
        )
        negative_heatmap_loss = (
            (1.0 - objectness_probability).log()
            * objectness_probability.pow(2.0)
            * (1.0 - objectness_target).pow(4.0)
            * non_peaks
        )
        classification = classification - (
            positive_heatmap_loss.sum() + negative_heatmap_loss.sum()
        )
        positive = target.positive.to(output.device)
        if positive.any():
            cell_probabilities = class_logits.sigmoid().permute(0, 2, 3, 1)[
                positive
            ]
            cell_targets = class_target.permute(0, 2, 3, 1)[positive]
            desired_classes = cell_targets.argmax(dim=1)
            center_objectness = objectness_probability[:, 0][positive]
            desired_class_probability = cell_probabilities[
                torch.arange(len(cell_probabilities), device=output.device), desired_classes
            ]
            positive_confidence = positive_confidence + (
                center_objectness * desired_class_probability
            ).sum()
            positive_correct = positive_correct + (
                cell_probabilities.argmax(dim=1) == desired_classes
            ).sum()
            measured_positives += len(cell_probabilities)
            classification = classification + F.binary_cross_entropy_with_logits(
                class_logits.permute(0, 2, 3, 1)[positive],
                cell_targets,
                reduction="sum",
            )
            raw_boxes = output[:, :4].permute(0, 2, 3, 1)[positive]
            box_target = (
                target.boxes.to(output.device, output.dtype)
                .permute(0, 2, 3, 1)[positive]
            )
            encoded_regression = encoded_regression + F.smooth_l1_loss(
                raw_boxes, box_target, reduction="sum", beta=0.1
            )
            student_xyxy = _student_boxes_xyxy(output, stride=stride)
            target_xyxy = _student_boxes_xyxy(
                target.boxes.to(output.device, output.dtype), stride=stride
            )
            student_positive = student_xyxy.permute(0, 2, 3, 1)[positive]
            target_positive = target_xyxy.permute(0, 2, 3, 1)[positive]

            overlap_regression = overlap_regression + aligned_generalized_iou_loss(
                student_positive,
                target_positive,
                reduction="sum",
            )
        background = ~positive
        background_scores = (
            objectness_probability[:, 0]
            * class_logits.sigmoid().max(dim=1).values
        )
        background_above_threshold += int(
            (background_scores[background] >= 0.25)
            .sum()
            .detach()
            .cpu()
        )
    classification = classification / normalizer
    encoded_regression = encoded_regression / normalizer
    overlap_regression = overlap_regression / normalizer
    regression = 0.25 * encoded_regression + overlap_regression
    total = classification + 2.0 * regression
    return total, {
        "loss": float(total.detach().cpu()),
        "classification_loss": float(classification.detach().cpu()),
        "box_loss": float(regression.detach().cpu()),
        "encoded_box_loss": float(encoded_regression.detach().cpu()),
        "generalized_iou_loss": float(overlap_regression.detach().cpu()),
        "positive_cells": float(positive_count),
        "positive_confidence_mean": float(
            (positive_confidence / max(1, measured_positives)).detach().cpu()
        ),
        "positive_top1_accuracy": float(
            (positive_correct / max(1, measured_positives)).detach().cpu()
        ),
        "background_cells_above_025": float(background_above_threshold),
    }
