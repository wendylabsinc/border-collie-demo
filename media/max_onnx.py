"""Small ONNX-to-MAX graph importer for the demo's fruit model.

Provenance: derived on 2026-08-05 from the Wendy G1 Modcon demo's pinned MAX
26.4 importer at ``Main Demo/tracker/max_onnx.py``. That importer was already
used to produce an ``sm_87`` MEF for Jetson Orin. This copy adds only the
``ConvTranspose`` operation required by the exact Border Collie YOLOE export.

MAX 26.4 no longer imports arbitrary ONNX files directly. This module
translates only the operators exercised by the pinned export into native
``max.graph`` operations. Unsupported operators fail closed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SUPPORTED_OPERATOR_TYPES = frozenset(
    {
        "Add",
        "AveragePool",
        "BatchNormalization",
        "Concat",
        "Constant",
        "Conv",
        "ConvTranspose",
        "Div",
        "Gather",
        "Gemm",
        "GlobalAveragePool",
        "MatMul",
        "MaxPool",
        "Mul",
        "Relu",
        "Reshape",
        "Resize",
        "Shape",
        "Sigmoid",
        "Slice",
        "Softmax",
        "Split",
        "Sub",
        "Transpose",
        "Unsqueeze",
    }
)

FLOAT_PRECISIONS = {
    "fp16": np.dtype(np.float16),
    "fp32": np.dtype(np.float32),
}

IDENTITY_4D_LAYOUT = (0, 1, 2, 3)
NATIVE_NHWC_LAYOUT = (0, 2, 3, 1)
SUPPORTED_LAYOUTS = frozenset({"native_nhwc"})


@dataclass(frozen=True)
class _TensorValue:
    """A MAX tensor plus the physical-axis order for its logical ONNX value.

    ONNX describes image activations as NCHW. MAX's optimized GPU kernels use
    NHWC. ``physical_to_logical=(0, 2, 3, 1)`` therefore means physical axes
    N,H,W,C while every operator in the importer can continue interpreting
    logical ONNX axes N,C,H,W.
    """

    tensor: Any
    physical_to_logical: tuple[int, ...]

    def __post_init__(self) -> None:
        if sorted(self.physical_to_logical) != list(
            range(len(self.physical_to_logical))
        ):
            raise ValueError(
                "physical_to_logical must be a permutation of the tensor axes"
            )

    @property
    def rank(self) -> int:
        return len(self.physical_to_logical)


def _normalize_axis(axis: int, rank: int) -> int:
    normalized = axis + rank if axis < 0 else axis
    if not 0 <= normalized < rank:
        raise ValueError(f"axis {axis} is out of bounds for rank {rank}")
    return normalized


def _native_layout(rank: int) -> tuple[int, ...]:
    return NATIVE_NHWC_LAYOUT if rank == 4 else tuple(range(rank))


def _physical_shape(
    logical_shape: tuple[int, ...], physical_to_logical: tuple[int, ...]
) -> tuple[int, ...]:
    if len(logical_shape) != len(physical_to_logical):
        raise ValueError("shape and layout must have the same rank")
    return tuple(logical_shape[axis] for axis in physical_to_logical)


def native_max_input_shape(logical_shape: tuple[int, ...]) -> tuple[int, ...]:
    """Return the physical MAX input shape for a logical ONNX input shape."""

    return _physical_shape(logical_shape, _native_layout(len(logical_shape)))


def _layout_permutation(source: tuple[int, ...], target: tuple[int, ...]) -> list[int]:
    """Return the physical transpose that converts ``source`` to ``target``."""

    if sorted(source) != sorted(target):
        raise ValueError("source and target layouts must contain the same axes")
    return [source.index(logical_axis) for logical_axis in target]


def _physical_axis(physical_to_logical: tuple[int, ...], logical_axis: int) -> int:
    normalized = _normalize_axis(logical_axis, len(physical_to_logical))
    return physical_to_logical.index(normalized)


def _array(value: Any) -> np.ndarray:
    """Keep ONNX scalar tensors rank-zero while making other arrays contiguous."""

    result = np.asarray(value)
    return result if result.ndim == 0 else np.ascontiguousarray(result)


def _normalize_precision(precision: str) -> str:
    normalized = precision.strip().lower()
    if normalized not in FLOAT_PRECISIONS:
        raise ValueError(
            f"unsupported precision {precision!r}; expected one of "
            + ", ".join(sorted(FLOAT_PRECISIONS))
        )
    return normalized


def _cast_floating_array(value: Any, precision: str) -> np.ndarray:
    result = _array(value)
    if np.issubdtype(result.dtype, np.floating):
        result = result.astype(FLOAT_PRECISIONS[_normalize_precision(precision)])
        return result if result.ndim == 0 else np.ascontiguousarray(result)
    return result


def _attributes(node: Any) -> dict[str, Any]:
    from onnx import helper

    return {
        attribute.name: helper.get_attribute_value(attribute)
        for attribute in node.attribute
    }


def _fixed_model(path: Path, input_shape: tuple[int, ...]) -> Any:
    import onnx

    model = onnx.load(path)
    if len(model.graph.input) != 1:
        raise ValueError(f"{path.name} must have exactly one graph input")
    dimensions = model.graph.input[0].type.tensor_type.shape.dim
    if len(dimensions) != len(input_shape):
        raise ValueError(
            f"{path.name} input rank {len(dimensions)} does not match {input_shape}"
        )
    for dimension, size in zip(dimensions, input_shape, strict=True):
        dimension.ClearField("dim_param")
        dimension.dim_value = size
    onnx.checker.check_model(model)
    return onnx.shape_inference.infer_shapes(model)


def _shape_table(model: Any) -> dict[str, tuple[int, ...]]:
    table: dict[str, tuple[int, ...]] = {}
    values = [*model.graph.input, *model.graph.value_info, *model.graph.output]
    for value in values:
        dimensions: list[int] = []
        static = True
        for dimension in value.type.tensor_type.shape.dim:
            if dimension.HasField("dim_value") and dimension.dim_value > 0:
                dimensions.append(int(dimension.dim_value))
            elif dimension.dim_param == "batch":
                dimensions.append(1)
            else:
                static = False
                break
        if static:
            table[value.name] = tuple(dimensions)
    return table


def _decode(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _constant_binary(op_type: str, lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    functions = {
        "Add": np.add,
        "Sub": np.subtract,
        "Mul": np.multiply,
        "Div": np.divide,
        "MatMul": np.matmul,
    }
    result = functions[op_type](lhs, rhs)
    if (
        op_type == "Div"
        and np.issubdtype(lhs.dtype, np.integer)
        and np.issubdtype(rhs.dtype, np.integer)
    ):
        result = np.trunc(result).astype(np.result_type(lhs.dtype, rhs.dtype))
    return _array(result)


def conv_transpose_spec(
    attributes: dict[str, Any], weight: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    """Map ONNX NCHW/CFRS ConvTranspose data to MAX NHWC/RSCF.

    ONNX stores transpose-convolution weights as input-channel,
    output-channel/group, row, column. MAX's transpose op consumes RSCF where
    the final dimension matches the NHWC input channels. The exact fruit graph
    uses group=1; grouped transpose convolution deliberately fails closed.
    """

    group = int(attributes.get("group", 1))
    if group != 1:
        raise NotImplementedError("grouped ConvTranspose is not supported")
    auto_pad = _decode(attributes.get("auto_pad", b"NOTSET"))
    if auto_pad != "NOTSET":
        raise NotImplementedError(f"ConvTranspose auto_pad {auto_pad!r}")
    if "output_shape" in attributes:
        raise NotImplementedError("ConvTranspose output_shape is not supported")
    pads = [int(item) for item in attributes.get("pads", [0, 0, 0, 0])]
    if len(pads) != 4:
        raise ValueError("ConvTranspose pads must have four values")
    options = {
        "stride": tuple(int(item) for item in attributes.get("strides", [1, 1])),
        "dilation": tuple(int(item) for item in attributes.get("dilations", [1, 1])),
        "padding": (pads[0], pads[2], pads[1], pads[3]),
        "output_paddings": tuple(
            int(item) for item in attributes.get("output_padding", [0, 0])
        ),
    }
    rscf_filter = np.ascontiguousarray(np.transpose(weight, [2, 3, 1, 0]))
    return rscf_filter, options


def _axes(value: np.ndarray) -> list[int]:
    return [int(item) for item in np.asarray(value).reshape(-1)]


def _last_axis_permutations(rank: int, axis: int) -> tuple[list[int], list[int]]:
    """Return permutations that move ``axis`` last and then restore it."""

    normalized_axis = axis + rank if axis < 0 else axis
    if not 0 <= normalized_axis < rank:
        raise ValueError(f"axis {axis} is out of bounds for rank {rank}")
    forward = [item for item in range(rank) if item != normalized_axis]
    forward.append(normalized_axis)
    inverse = [forward.index(item) for item in range(rank)]
    return forward, inverse


def _compose_permutations(first: list[int], second: list[int]) -> list[int]:
    """Compose consecutive NumPy/MAX transpose permutations."""

    if len(first) != len(second):
        raise ValueError("transpose permutations must have the same rank")
    return [first[index] for index in second]


def build_max_graph(
    path: Path,
    input_shape: tuple[int, ...],
    *,
    graph_name: str | None = None,
    precision: str = "fp32",
    layout: str = "native_nhwc",
) -> tuple[Any, dict[str, np.ndarray]]:
    """Translate ONNX while retaining MAX-native NHWC image activations."""

    from max.dtype import DType
    from max.graph import DeviceRef, Graph, TensorType, Weight, ops
    from onnx import numpy_helper

    precision = _normalize_precision(precision)
    layout = layout.strip().lower()
    if layout not in SUPPORTED_LAYOUTS:
        raise ValueError(
            f"unsupported layout {layout!r}; expected one of "
            + ", ".join(sorted(SUPPORTED_LAYOUTS))
        )
    model = _fixed_model(path, input_shape)
    shapes = _shape_table(model)
    device = DeviceRef.GPU()
    input_name = model.graph.input[0].name
    output_names = [output.name for output in model.graph.output]
    initializers = {
        initializer.name: _cast_floating_array(
            numpy_helper.to_array(initializer), precision
        )
        for initializer in model.graph.initializer
    }
    max_float_dtype = {
        "fp16": DType.float16,
        "fp32": DType.float32,
    }[precision]
    input_layout = _native_layout(len(input_shape))
    input_type = TensorType(
        max_float_dtype,
        _physical_shape(input_shape, input_layout),
        device=device,
    )
    weight_values: dict[str, np.ndarray] = {}
    weight_nodes: dict[int, Any] = {}
    with Graph(
        graph_name or path.stem.replace("-", "_"), input_types=[input_type]
    ) as graph:
        values: dict[str, Any] = {
            input_name: _TensorValue(graph.inputs[0], input_layout),
            **initializers,
        }

        def value(name: str) -> Any:
            if not name:
                return None
            if name not in values:
                raise KeyError(f"value {name!r} has not been produced")
            return values[name]

        def constant(name: str) -> np.ndarray:
            result = value(name)
            if not isinstance(result, np.ndarray):
                raise TypeError(f"{name!r} must be constant for MAX graph import")
            return result

        def tensor(item: Any) -> Any:
            if isinstance(item, _TensorValue):
                return item.tensor
            if isinstance(item, np.ndarray):
                item = _cast_floating_array(item, precision)
                if np.issubdtype(item.dtype, np.floating) and item.nbytes >= 32:
                    key = id(item)
                    if key not in weight_nodes:
                        dtypes = {
                            np.dtype(np.float16): DType.float16,
                            np.dtype(np.float32): DType.float32,
                            np.dtype(np.float64): DType.float64,
                        }
                        name = f"onnx_weight_{len(weight_values):04d}"
                        weight_nodes[key] = Weight(
                            name,
                            dtypes[item.dtype],
                            item.shape,
                            device=device,
                        )
                        weight_values[name] = item
                    return weight_nodes[key]
                return ops.constant(item, device=device)
            if isinstance(item, np.generic):
                return ops.constant(item.item(), device=device)
            if isinstance(item, (bool, int, float)):
                return ops.constant(item, device=device)
            return item

        def to_layout(item: _TensorValue, target: tuple[int, ...]) -> _TensorValue:
            if item.rank != len(target):
                raise ValueError(
                    f"cannot convert rank-{item.rank} tensor to rank-{len(target)} layout"
                )
            if item.physical_to_logical == target:
                return item
            return _TensorValue(
                ops.permute(
                    item.tensor,
                    _layout_permutation(item.physical_to_logical, target),
                ),
                target,
            )

        def identity_value(item: Any) -> _TensorValue:
            if not isinstance(item, _TensorValue):
                raise TypeError("expected a runtime tensor")
            return to_layout(item, tuple(range(item.rank)))

        def constant_for_layout(
            item: np.ndarray,
            *,
            rank: int,
            target: tuple[int, ...],
        ) -> np.ndarray:
            result = _cast_floating_array(item, precision)
            if result.ndim == 0 or target == tuple(range(rank)):
                return result
            if result.ndim > rank:
                raise ValueError(
                    f"rank-{result.ndim} constant cannot broadcast to rank {rank}"
                )
            logical_shape = (1,) * (rank - result.ndim) + result.shape
            logical = np.reshape(result, logical_shape)
            permutation = _layout_permutation(tuple(range(rank)), target)
            return np.ascontiguousarray(np.transpose(logical, permutation))

        def runtime_operand(
            item: Any,
            *,
            rank: int,
            target: tuple[int, ...],
        ) -> Any:
            if isinstance(item, _TensorValue):
                return to_layout(item, target).tensor
            if isinstance(item, np.ndarray):
                item = constant_for_layout(item, rank=rank, target=target)
            return tensor(item)

        def inferred_rank(output_name: str, inputs: list[Any]) -> int:
            if output_name in shapes:
                return len(shapes[output_name])
            ranks = [
                item.rank
                if isinstance(item, _TensorValue)
                else item.ndim
                if isinstance(item, np.ndarray)
                else 0
                for item in inputs
                if item is not None
            ]
            if not ranks:
                raise ValueError(f"cannot infer rank for {output_name!r}")
            return max(ranks)

        def binary(
            op_type: str,
            lhs: Any,
            rhs: Any,
            *,
            output_name: str,
        ) -> Any:
            if isinstance(lhs, np.ndarray) and isinstance(rhs, np.ndarray):
                return _constant_binary(op_type, lhs, rhs)
            functions = {
                "Add": ops.add,
                "Sub": ops.sub,
                "Mul": ops.mul,
                "Div": ops.div,
            }
            rank = inferred_rank(output_name, [lhs, rhs])
            target = _native_layout(rank)
            return _TensorValue(
                functions[op_type](
                    runtime_operand(lhs, rank=rank, target=target),
                    runtime_operand(rhs, rank=rank, target=target),
                ),
                target,
            )

        for node in model.graph.node:
            attributes = _attributes(node)
            inputs = [value(name) if name else None for name in node.input]
            op_type = node.op_type

            if op_type == "Constant":
                result = _cast_floating_array(
                    numpy_helper.to_array(attributes["value"]), precision
                )
            elif op_type == "Shape":
                result = np.asarray(shapes[node.input[0]], dtype=np.int64)
            elif op_type == "Gather" and all(
                isinstance(item, np.ndarray) for item in inputs
            ):
                result = _array(
                    np.take(inputs[0], inputs[1], axis=int(attributes.get("axis", 0)))
                )
            elif op_type == "Unsqueeze":
                axes = (
                    _axes(inputs[1])
                    if len(inputs) > 1 and inputs[1] is not None
                    else [int(axis) for axis in attributes["axes"]]
                )
                if isinstance(inputs[0], np.ndarray):
                    result = inputs[0]
                    for axis in sorted(axes):
                        result = np.expand_dims(result, axis)
                    result = _array(result)
                else:
                    source = identity_value(inputs[0])
                    raw_result = source.tensor
                    for axis in sorted(axes):
                        raw_result = ops.unsqueeze(raw_result, axis)
                    result = _TensorValue(
                        raw_result,
                        tuple(range(source.rank + len(axes))),
                    )
            elif op_type == "Concat":
                axis = int(attributes["axis"])
                if all(isinstance(item, np.ndarray) for item in inputs):
                    result = _array(np.concatenate(inputs, axis=axis))
                else:
                    rank = inferred_rank(node.output[0], inputs)
                    target = _native_layout(rank)
                    result = _TensorValue(
                        ops.concat(
                            [
                                runtime_operand(item, rank=rank, target=target)
                                for item in inputs
                            ],
                            axis=_physical_axis(target, axis),
                        ),
                        target,
                    )
            elif op_type in {"Add", "Sub", "Mul", "Div"}:
                result = binary(
                    op_type,
                    inputs[0],
                    inputs[1],
                    output_name=node.output[0],
                )
            elif op_type == "MatMul":
                lhs = (
                    identity_value(inputs[0]).tensor
                    if isinstance(inputs[0], _TensorValue)
                    else tensor(inputs[0])
                )
                rhs = (
                    identity_value(inputs[1]).tensor
                    if isinstance(inputs[1], _TensorValue)
                    else tensor(inputs[1])
                )
                rank = inferred_rank(node.output[0], inputs)
                result = _TensorValue(ops.matmul(lhs, rhs), tuple(range(rank)))
            elif op_type == "Relu":
                source = inputs[0]
                result = _TensorValue(
                    ops.relu(source.tensor), source.physical_to_logical
                )
            elif op_type == "Sigmoid":
                source = inputs[0]
                result = _TensorValue(
                    ops.sigmoid(source.tensor), source.physical_to_logical
                )
            elif op_type == "Softmax":
                source = inputs[0]
                axis = int(attributes.get("axis", -1))
                physical_axis = _physical_axis(source.physical_to_logical, axis)
                forward, _inverse = _last_axis_permutations(source.rank, physical_axis)
                # MAX 26.4's GPU softmax kernel requires the innermost
                # physical axis. Retain the new layout after moving that axis;
                # a following convolution can consume NHWC directly.
                if forward == list(range(source.rank)):
                    result = _TensorValue(
                        ops.softmax(source.tensor, axis=source.rank - 1),
                        source.physical_to_logical,
                    )
                else:
                    transposed = ops.permute(source.tensor, forward)
                    result = _TensorValue(
                        ops.softmax(transposed, axis=source.rank - 1),
                        tuple(source.physical_to_logical[index] for index in forward),
                    )
            elif op_type == "ConvTranspose":
                rscf_filter, options = conv_transpose_spec(
                    attributes, np.asarray(inputs[1])
                )
                source = to_layout(inputs[0], NATIVE_NHWC_LAYOUT)
                transposed = ops.conv2d_transpose(
                    source.tensor,
                    tensor(rscf_filter),
                    bias=None,
                    **options,
                )
                if len(inputs) > 2 and inputs[2] is not None:
                    # MAX 26.4 permutes the NHWC result incorrectly in the
                    # op's built-in bias path. Adding the same bias explicitly
                    # preserves ONNX semantics and the correct output layout.
                    bias = ops.reshape(tensor(inputs[2]), [1, 1, 1, -1])
                    transposed = ops.add(transposed, bias)
                result = _TensorValue(transposed, NATIVE_NHWC_LAYOUT)
            elif op_type == "Conv":
                pads = [int(item) for item in attributes.get("pads", [0, 0, 0, 0])]
                if pads[0] != pads[2] or pads[1] != pads[3]:
                    raise NotImplementedError(f"asymmetric Conv padding in {node.name}")
                strides = tuple(int(item) for item in attributes.get("strides", [1, 1]))
                dilations = tuple(
                    int(item) for item in attributes.get("dilations", [1, 1])
                )
                source = to_layout(inputs[0], NATIVE_NHWC_LAYOUT)
                rscf_filter = np.transpose(np.asarray(inputs[1]), [2, 3, 1, 0])
                convolved = ops.conv2d(
                    source.tensor,
                    tensor(np.ascontiguousarray(rscf_filter)),
                    stride=strides,
                    dilation=dilations,
                    padding=(pads[0], pads[2], pads[1], pads[3]),
                    groups=int(attributes.get("group", 1)),
                    bias=tensor(inputs[2]) if len(inputs) > 2 else None,
                )
                result = _TensorValue(convolved, NATIVE_NHWC_LAYOUT)
            elif op_type in {"MaxPool", "AveragePool"}:
                pads = [int(item) for item in attributes.get("pads", [0, 0, 0, 0])]
                if pads[0] != pads[2] or pads[1] != pads[3]:
                    raise NotImplementedError(f"asymmetric Pool padding in {node.name}")
                source = to_layout(inputs[0], NATIVE_NHWC_LAYOUT)
                options = {
                    "kernel_size": tuple(
                        int(item) for item in attributes["kernel_shape"]
                    ),
                    "stride": tuple(
                        int(item) for item in attributes.get("strides", [1, 1])
                    ),
                    "dilation": tuple(
                        int(item) for item in attributes.get("dilations", [1, 1])
                    ),
                    "padding": (pads[0], pads[1]),
                    "ceil_mode": bool(attributes.get("ceil_mode", 0)),
                }
                if op_type == "MaxPool":
                    pooled = ops.max_pool2d(source.tensor, **options)
                else:
                    pooled = ops.avg_pool2d(
                        source.tensor,
                        **options,
                        count_boundary=bool(attributes.get("count_include_pad", 0)),
                    )
                result = _TensorValue(pooled, NATIVE_NHWC_LAYOUT)
            elif op_type == "GlobalAveragePool":
                source = identity_value(inputs[0])
                result = _TensorValue(
                    ops.mean(ops.mean(source.tensor, axis=2), axis=3),
                    IDENTITY_4D_LAYOUT,
                )
            elif op_type == "BatchNormalization":
                epsilon = float(attributes.get("epsilon", 1e-5))
                source = inputs[0]
                rank = source.rank
                if rank < 2:
                    raise ValueError(
                        f"BatchNormalization input must have rank >= 2 in {node.name}"
                    )
                parameter_shape = [1] * rank
                parameter_shape[_physical_axis(source.physical_to_logical, 1)] = -1
                scale, bias, mean, variance = [
                    np.asarray(item).reshape(parameter_shape) for item in inputs[1:5]
                ]
                multiplier = scale / np.sqrt(variance + epsilon)
                offset = bias - mean * multiplier
                result = _TensorValue(
                    ops.add(
                        ops.mul(source.tensor, tensor(multiplier)),
                        tensor(offset),
                    ),
                    source.physical_to_logical,
                )
            elif op_type == "Reshape":
                source = identity_value(inputs[0])
                target_shape = _axes(constant(node.input[1]))
                result = _TensorValue(
                    ops.reshape(source.tensor, target_shape),
                    tuple(range(len(target_shape))),
                )
            elif op_type == "Transpose":
                permutation = [int(item) for item in attributes["perm"]]
                source = identity_value(inputs[0])
                result = _TensorValue(
                    ops.permute(source.tensor, permutation),
                    tuple(range(source.rank)),
                )
            elif op_type == "Split":
                axis = int(attributes.get("axis", 0))
                source = inputs[0]
                if len(inputs) > 1 and inputs[1] is not None:
                    sizes = _axes(inputs[1])
                elif "split" in attributes:
                    sizes = [int(item) for item in attributes["split"]]
                else:
                    logical_axis = _normalize_axis(axis, source.rank)
                    sizes = [shapes[name][logical_axis] for name in node.output]
                split_results = ops.split(
                    source.tensor,
                    sizes,
                    axis=_physical_axis(source.physical_to_logical, axis),
                )
                if len(split_results) != len(node.output):
                    raise ValueError(f"Split output mismatch in {node.name}")
                values.update(
                    (
                        name,
                        _TensorValue(item, source.physical_to_logical),
                    )
                    for name, item in zip(node.output, split_results, strict=True)
                )
                continue
            elif op_type == "Slice":
                starts = _axes(inputs[1])
                ends = _axes(inputs[2])
                axes = (
                    _axes(inputs[3])
                    if len(inputs) > 3 and inputs[3] is not None
                    else list(range(len(starts)))
                )
                steps = (
                    _axes(inputs[4])
                    if len(inputs) > 4 and inputs[4] is not None
                    else [1] * len(starts)
                )
                source = inputs[0]
                indices: list[slice] = [slice(None)] * source.rank
                for start, end, axis, step in zip(
                    starts, ends, axes, steps, strict=True
                ):
                    stop = None if end >= np.iinfo(np.int64).max else end
                    indices[_physical_axis(source.physical_to_logical, axis)] = slice(
                        start, stop, step
                    )
                result = _TensorValue(
                    ops.slice_tensor(source.tensor, indices),
                    source.physical_to_logical,
                )
            elif op_type == "Resize":
                output_shape = shapes[node.output[0]]
                mode = _decode(attributes.get("mode", b"nearest"))
                transform = _decode(
                    attributes.get("coordinate_transformation_mode", b"half_pixel")
                )
                transform_modes = {
                    "half_pixel": 0,
                    "align_corners": 1,
                    "asymmetric": 2,
                    "pytorch_half_pixel": 0,
                }
                if transform not in transform_modes:
                    raise NotImplementedError(f"Resize transform {transform!r}")
                if mode == "nearest":
                    nearest = _decode(
                        attributes.get("nearest_mode", b"round_prefer_floor")
                    )
                    round_modes = {
                        "round_prefer_floor": 0,
                        "round_prefer_ceil": 1,
                        "floor": 2,
                        "ceil": 3,
                    }
                    input_shape_for_resize = shapes[node.input[0]]
                    scale_h = output_shape[2] // input_shape_for_resize[2]
                    scale_w = output_shape[3] // input_shape_for_resize[3]
                    exact_integer_scale = (
                        output_shape[:2] == input_shape_for_resize[:2]
                        and output_shape[2] == input_shape_for_resize[2] * scale_h
                        and output_shape[3] == input_shape_for_resize[3] * scale_w
                        and transform == "asymmetric"
                        and nearest == "floor"
                    )
                    target = NATIVE_NHWC_LAYOUT
                    source = to_layout(inputs[0], target)
                    physical_output_shape = _physical_shape(output_shape, target)
                    if exact_integer_scale:
                        n, channels, height, width = input_shape_for_resize
                        expanded = ops.reshape(
                            source.tensor,
                            [n, height, 1, width, 1, channels],
                        )
                        repeated = ops.broadcast_to(
                            expanded,
                            [n, height, scale_h, width, scale_w, channels],
                        )
                        resized = ops.reshape(repeated, physical_output_shape)
                    else:
                        resized = ops.resize_nearest(
                            source.tensor,
                            physical_output_shape,
                            coordinate_transform_mode=transform_modes[transform],
                            round_mode=round_modes[nearest],
                        )
                    result = _TensorValue(resized, target)
                elif mode == "linear":
                    target = NATIVE_NHWC_LAYOUT
                    source = to_layout(inputs[0], target)
                    result = _TensorValue(
                        ops.resize_linear(
                            source.tensor,
                            _physical_shape(output_shape, target),
                            coordinate_transform_mode=transform_modes[transform],
                        ),
                        target,
                    )
                else:
                    raise NotImplementedError(f"Resize mode {mode!r}")
            elif op_type == "Gemm":
                lhs, rhs, bias = inputs[:3]
                if isinstance(lhs, _TensorValue):
                    lhs = identity_value(lhs).tensor
                if isinstance(rhs, _TensorValue):
                    rhs = identity_value(rhs).tensor
                if int(attributes.get("transA", 0)):
                    lhs = (
                        np.swapaxes(lhs, -1, -2)
                        if isinstance(lhs, np.ndarray)
                        else ops.transpose(lhs, -1, -2)
                    )
                if int(attributes.get("transB", 0)):
                    rhs = (
                        np.swapaxes(rhs, -1, -2)
                        if isinstance(rhs, np.ndarray)
                        else ops.transpose(rhs, -1, -2)
                    )
                result = _TensorValue(
                    ops.matmul(tensor(lhs), tensor(rhs)),
                    tuple(range(inferred_rank(node.output[0], inputs))),
                )
                alpha = float(attributes.get("alpha", 1.0))
                beta = float(attributes.get("beta", 1.0))
                if alpha != 1.0:
                    result = binary(
                        "Mul",
                        result,
                        np.asarray(alpha, dtype=np.float32),
                        output_name=node.output[0],
                    )
                if bias is not None:
                    if beta != 1.0:
                        bias = np.asarray(bias) * beta
                    result = binary(
                        "Add",
                        result,
                        bias,
                        output_name=node.output[0],
                    )
            else:
                raise NotImplementedError(
                    f"ONNX operator {op_type!r} is not supported ({node.name})"
                )

            if len(node.output) != 1:
                raise ValueError(
                    f"{op_type} unexpectedly produced {len(node.output)} values"
                )
            values[node.output[0]] = result

        outputs = [identity_value(value(name)).tensor for name in output_names]
        graph.output(*outputs)
    return graph, weight_values


def supported_operator_types(path: Path) -> set[str]:
    """Return the operator set for diagnostics and regression tests."""

    import onnx

    return {node.op_type for node in onnx.load(path).graph.node}


def unsupported_operator_types(path: Path) -> set[str]:
    """Return graph operators that this intentionally narrow importer rejects."""

    return supported_operator_types(path) - SUPPORTED_OPERATOR_TYPES
