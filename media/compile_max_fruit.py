"""Inspect or compile the fixed-shape Border Collie fruit model for MAX."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from media.max_onnx import (
    build_max_graph,
    supported_operator_types,
    unsupported_operator_types,
)

FRUIT_INPUT_SHAPE = (1, 3, 640, 640)
FRUIT_OPERATOR_TYPES = frozenset(
    {
        "Add",
        "Concat",
        "Constant",
        "Conv",
        "ConvTranspose",
        "Div",
        "Gather",
        "MatMul",
        "MaxPool",
        "Mul",
        "Reshape",
        "Resize",
        "Shape",
        "Sigmoid",
        "Slice",
        "Softmax",
        "Split",
        "Sub",
        "Transpose",
    }
)


def inspect_fruit_onnx(path: Path) -> dict[str, object]:
    operators = supported_operator_types(path)
    unsupported = unsupported_operator_types(path)
    return {
        "onnx": path.name,
        "input_shape": list(FRUIT_INPUT_SHAPE),
        "operator_types": sorted(operators),
        "matches_validated_export_operator_set": operators == FRUIT_OPERATOR_TYPES,
        "unsupported_operator_types": sorted(unsupported),
        "ready_for_max_compile": not unsupported,
    }


def compile_fruit_onnx(
    *,
    onnx_path: Path,
    mef_path: Path,
    weights_path: Path,
) -> dict[str, object]:
    inspection = inspect_fruit_onnx(onnx_path)
    if inspection["unsupported_operator_types"]:
        raise RuntimeError(
            "fruit ONNX contains unsupported operators: "
            + ", ".join(inspection["unsupported_operator_types"])
        )

    import numpy as np
    from max import driver, engine

    graph, weights = build_max_graph(onnx_path, FRUIT_INPUT_SHAPE)
    session = engine.InferenceSession(devices=[driver.Accelerator(0)])
    compiled = session.compile(graph)
    mef_path.parent.mkdir(parents=True, exist_ok=True)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    compiled.export_mef(mef_path)
    np.savez(weights_path, **weights)
    return {
        **inspection,
        "mef": mef_path.name,
        "weights": weights_path.name,
        "weight_tensor_count": len(weights),
        "compiled": True,
        "execution_validated": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect or compile the fixed 640px fruit ONNX for MAX"
    )
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--mef", type=Path)
    parser.add_argument("--weights", type=Path)
    args = parser.parse_args()

    if args.check_only:
        result = inspect_fruit_onnx(args.onnx)
    else:
        if args.mef is None or args.weights is None:
            parser.error("--mef and --weights are required for compilation")
        result = compile_fruit_onnx(
            onnx_path=args.onnx,
            mef_path=args.mef,
            weights_path=args.weights,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
