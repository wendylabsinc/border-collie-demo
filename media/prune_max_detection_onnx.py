"""Remove the unused YOLO segmentation branch before MAX AOT compilation."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import onnx
from onnx import shape_inference, utils


def _channel_count(model: onnx.ModelProto, value_name: str) -> int:
    values = [*model.graph.input, *model.graph.value_info, *model.graph.output]
    value = next((candidate for candidate in values if candidate.name == value_name), None)
    if value is None:
        raise ValueError(f"shape information is missing for {value_name!r}")
    dimensions = value.type.tensor_type.shape.dim
    if len(dimensions) < 2 or not dimensions[1].HasField("dim_value"):
        raise ValueError(f"channel dimension is not static for {value_name!r}")
    return dimensions[1].dim_value


def prune_segmentation_branch(source: Path, destination: Path) -> dict[str, int]:
    """Keep the box/class branches and prune mask coefficients plus mask prototypes."""
    model = shape_inference.infer_shapes(onnx.load(source))
    if len(model.graph.output) != 2:
        raise ValueError("expected the YOLO segmentation export to have two outputs")

    detection_output = model.graph.output[0]
    producer = next(
        (node for node in model.graph.node if detection_output.name in node.output), None
    )
    if producer is None or producer.op_type != "Concat" or len(producer.input) != 3:
        raise ValueError("expected detection output to be a three-input Concat")

    kept_inputs = list(producer.input[:2])
    detection_channels = sum(_channel_count(model, name) for name in kept_inputs)
    del producer.input[:]
    producer.input.extend(kept_inputs)
    del model.graph.output[1:]
    channel_dimension = detection_output.type.tensor_type.shape.dim[1]
    channel_dimension.ClearField("dim_param")
    channel_dimension.dim_value = detection_channels

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".onnx") as intermediate:
        onnx.save(model, intermediate.name)
        utils.extract_model(
            intermediate.name,
            str(destination),
            [model.graph.input[0].name],
            [detection_output.name],
            check_model=True,
        )

    pruned = onnx.load(destination)
    onnx.checker.check_model(pruned)
    return {
        "source_nodes": len(model.graph.node),
        "pruned_nodes": len(pruned.graph.node),
        "detection_channels": detection_channels,
        "source_bytes": source.stat().st_size,
        "pruned_bytes": destination.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    stats = prune_segmentation_branch(args.source, args.destination)
    for name, value in stats.items():
        print(f"{name}={value}")


if __name__ == "__main__":
    main()
