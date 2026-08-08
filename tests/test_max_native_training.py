from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from training.max_native.data import select_whole_fruit_records
from training.max_native.detection import (
    align_teacher_predictions,
    aligned_generalized_iou_loss,
    decode_predictions,
    detection_loss,
    distillation_loss,
    encode_targets,
)
from training.max_native.evaluate import match_detections
from training.max_native.export import fold_conv_norm
from training.max_native.model import (
    DEFAULT_SPEC,
    TrainableFruitDetector,
)


def test_trainable_twin_matches_the_benchmarked_max_contract() -> None:
    model = TrainableFruitDetector(DEFAULT_SPEC)

    assert DEFAULT_SPEC.version == 4
    assert DEFAULT_SPEC.feature_pyramid_channels == 64
    assert DEFAULT_SPEC.box_encoding == "ltrb_exp_grid_center"
    assert DEFAULT_SPEC.classes == ("pear", "apple", "banana")
    assert DEFAULT_SPEC.max_output_shapes(640) == (
        (1, 80, 80, 8),
        (1, 40, 40, 8),
        (1, 20, 20, 8),
    )

    export_layers = dict(model.named_export_layers())
    assert len(export_layers) == 63
    assert tuple(export_layers)[:4] == (
        "stem",
        "blocks.0.depthwise",
        "blocks.0.project",
        "blocks.1.expand",
    )
    assert tuple(export_layers)[-9:] == (
        "heads.stride_8.depthwise",
        "heads.stride_8.pointwise",
        "heads.stride_8.output",
        "heads.stride_16.depthwise",
        "heads.stride_16.pointwise",
        "heads.stride_16.output",
        "heads.stride_32.depthwise",
        "heads.stride_32.pointwise",
        "heads.stride_32.output",
    )
    first_head = model.heads["8"].output
    assert float(first_head.bias[4].detach()) == pytest.approx(-4.59511985013459)
    assert first_head.bias[5:].detach().tolist() == pytest.approx([0.0, 0.0, 0.0])

    outputs = model(torch.zeros((1, 3, 640, 640)))
    assert tuple(tuple(output.shape) for output in outputs) == (
        (1, 8, 80, 80),
        (1, 8, 40, 40),
        (1, 8, 20, 20),
    )


def test_max_benchmark_uses_the_exact_training_architecture() -> None:
    benchmark_manifest = json.loads(
        (
            Path(__file__).parents[1]
            / "lab/max-native-skeleton-prototype/architecture.json"
        ).read_text()
    )

    assert benchmark_manifest == json.loads(json.dumps(DEFAULT_SPEC.manifest()))


def test_training_targets_round_trip_through_the_deployment_decoder() -> None:
    boxes = [torch.tensor([[68.0, 84.0, 132.0, 148.0]])]
    class_ids = [torch.tensor([1])]
    targets = encode_targets(boxes, class_ids, input_size=640)

    outputs = [
        torch.full((1, 8, 80, 80), -20.0),
        torch.full((1, 8, 40, 40), -20.0),
        torch.full((1, 8, 20, 20), -20.0),
    ]
    positive = targets[0].positive
    assert positive.nonzero().tolist() == [
        [0, 13, 11],
        [0, 13, 12],
        [0, 13, 13],
        [0, 14, 11],
        [0, 14, 12],
        [0, 14, 13],
        [0, 15, 11],
        [0, 15, 12],
        [0, 15, 13],
    ]
    assert float(targets[0].objectness[0, 0, 14, 12]) == 1.0
    assert 0.0 < float(targets[0].objectness[0, 0, 14, 11]) < 1.0
    assert 0.0 < float(targets[0].objectness[0, 0, 14, 10]) < 1.0
    assert float(targets[0].classes[0, 1, 14, 12]) == 1.0
    outputs[0][0, :4, 14, 12] = targets[0].boxes[0, :, 14, 12]
    outputs[0][0, 4, 14, 12] = 10.0
    outputs[0][0, 5 + 1, 14, 12] = 10.0

    detections = decode_predictions(
        outputs,
        input_size=640,
        confidence_threshold=0.5,
        iou_threshold=0.5,
    )[0]

    assert detections.class_ids.tolist() == [1]
    assert detections.scores.tolist() == pytest.approx([0.9999092])
    torch.testing.assert_close(
        detections.boxes,
        torch.tensor([[68.0, 84.0, 132.0, 148.0]]),
        atol=1e-4,
        rtol=0.0,
    )


def test_yolo_teacher_predictions_align_with_the_three_max_native_grids() -> None:
    input_size = 64
    anchor_count = 8**2 + 4**2 + 2**2
    teacher = torch.zeros((1, 7, anchor_count))
    teacher[0, :4, 5] = torch.tensor([12.0, 4.0, 8.0, 6.0])
    teacher[0, 4:, 5] = torch.tensor([0.1, 0.8, 0.2])

    targets = align_teacher_predictions(teacher, input_size=input_size)

    assert tuple(tuple(target.boxes_xywh.shape) for target in targets) == (
        (1, 4, 8, 8),
        (1, 4, 4, 4),
        (1, 4, 2, 2),
    )
    assert tuple(tuple(target.class_scores.shape) for target in targets) == (
        (1, 3, 8, 8),
        (1, 3, 4, 4),
        (1, 3, 2, 2),
    )
    torch.testing.assert_close(
        targets[0].boxes_xywh[0, :, 0, 5],
        torch.tensor([12.0, 4.0, 8.0, 6.0]),
    )
    torch.testing.assert_close(
        targets[0].class_scores[0, :, 0, 5],
        torch.tensor([0.1, 0.8, 0.2]),
    )

    student = [
        torch.full((1, 8, 8, 8), -4.0),
        torch.full((1, 8, 4, 4), -4.0),
        torch.full((1, 8, 2, 2), -4.0),
    ]
    loss, metrics = distillation_loss(student, targets, input_size=input_size)
    assert torch.isfinite(loss)
    assert metrics["teacher_positive_anchors"] == 1.0


def test_generalized_iou_loss_directly_rewards_box_overlap() -> None:
    target = torch.tensor([[10.0, 10.0, 30.0, 30.0]])

    exact = aligned_generalized_iou_loss(target, target)
    shifted = aligned_generalized_iou_loss(
        torch.tensor([[20.0, 10.0, 40.0, 30.0]]), target
    )
    disjoint = aligned_generalized_iou_loss(
        torch.tensor([[40.0, 40.0, 60.0, 60.0]]), target
    )

    assert float(exact) == pytest.approx(0.0)
    assert 0.0 < float(shifted) < float(disjoint)


def test_class_loss_rejects_ambiguous_multi_class_logits() -> None:
    targets = encode_targets(
        [torch.tensor([[68.0, 84.0, 132.0, 148.0]])],
        [torch.tensor([1])],
        input_size=640,
    )
    exclusive = [
        torch.full((1, 8, 80, 80), -20.0),
        torch.full((1, 8, 40, 40), -20.0),
        torch.full((1, 8, 20, 20), -20.0),
    ]
    for position in targets[0].positive.nonzero():
        _, y, x = (int(value) for value in position)
        exclusive[0][0, :4, y, x] = targets[0].boxes[0, :, y, x]
        exclusive[0][0, 4, y, x] = 10.0
        exclusive[0][0, 5:, y, x] = torch.tensor([-10.0, 10.0, -10.0])
    ambiguous = [output.clone() for output in exclusive]
    for position in targets[0].positive.nonzero():
        _, y, x = (int(value) for value in position)
        ambiguous[0][0, 5:, y, x] = 10.0

    exclusive_loss, _ = detection_loss(exclusive, targets)
    ambiguous_loss, _ = detection_loss(ambiguous, targets)

    assert float(ambiguous_loss) > float(exclusive_loss) + 10.0


def test_dataset_selection_excludes_low_ranked_cut_fruit(tmp_path) -> None:
    manifest = {
        "records": [
            {
                "image_id": "whole-pear",
                "local_image": "images/whole-pear.jpg",
                "annotations": [
                    {
                        "class_name": "pear",
                        "bbox_xyxy_normalized": [0.1, 0.2, 0.5, 0.8],
                    }
                ],
            },
            {
                "image_id": "cut-banana",
                "local_image": "images/cut-banana.jpg",
                "annotations": [
                    {
                        "class_name": "banana",
                        "bbox_xyxy_normalized": [0.2, 0.2, 0.8, 0.8],
                    }
                ],
            },
        ]
    }
    ranking = {
        "candidates": [
            {
                "candidate_id": "whole-pear:0",
                "whole_fruit_score": 0.91,
            },
            {
                "candidate_id": "cut-banana:0",
                "whole_fruit_score": 0.22,
            },
        ]
    }
    manifest_path = tmp_path / "manifest.json"
    ranking_path = tmp_path / "ranking.json"
    manifest_path.write_text(__import__("json").dumps(manifest))
    ranking_path.write_text(__import__("json").dumps(ranking))

    records = select_whole_fruit_records(
        manifest_path,
        ranking_path,
        minimum_whole_score=0.75,
        split="all",
    )

    assert [record.image_id for record in records] == ["whole-pear"]
    assert records[0].class_ids == (0,)


def test_evaluation_matching_counts_duplicates_and_wrong_classes() -> None:
    predicted_boxes = torch.tensor(
        [
            [10.0, 10.0, 50.0, 50.0],
            [11.0, 11.0, 49.0, 49.0],
            [70.0, 70.0, 90.0, 90.0],
        ]
    )
    predicted_classes = torch.tensor([0, 0, 2])
    target_boxes = torch.tensor(
        [
            [10.0, 10.0, 50.0, 50.0],
            [70.0, 70.0, 90.0, 90.0],
        ]
    )
    target_classes = torch.tensor([0, 1])

    counts = match_detections(
        predicted_boxes,
        predicted_classes,
        target_boxes,
        target_classes,
        iou_threshold=0.5,
    )

    assert counts == {"true_positives": 1, "false_positives": 2, "false_negatives": 1}


def test_batch_norm_folding_and_hwio_export_preserve_outputs() -> None:
    from training.max_native.model import ConvNorm

    layer = ConvNorm(3, 4, 3)
    layer.eval()
    with torch.no_grad():
        layer.conv.weight.copy_(
            torch.arange(layer.conv.weight.numel()).reshape_as(layer.conv.weight)
            / 100.0
        )
        layer.norm.weight.copy_(torch.tensor([0.8, 1.2, 0.5, 1.5]))
        layer.norm.bias.copy_(torch.tensor([-0.2, 0.3, 0.1, -0.4]))
        layer.norm.running_mean.copy_(torch.tensor([0.2, -0.1, 0.4, 0.0]))
        layer.norm.running_var.copy_(torch.tensor([0.7, 1.3, 0.4, 2.0]))
    exported_weight, exported_bias = fold_conv_norm(layer)
    assert exported_weight.shape == (3, 3, 3, 4)
    fused = torch.nn.Conv2d(3, 4, 3, padding=1, bias=True)
    with torch.no_grad():
        fused.weight.copy_(torch.from_numpy(exported_weight).permute(3, 2, 0, 1))
        fused.bias.copy_(torch.from_numpy(exported_bias))
    value = torch.randn((2, 3, 8, 8))
    torch.testing.assert_close(layer(value), fused(value), atol=1e-5, rtol=1e-5)
