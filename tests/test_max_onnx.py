from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")

from media.compile_max_fruit import FRUIT_OPERATOR_TYPES
from media.max_onnx import (
    NATIVE_NHWC_LAYOUT,
    SUPPORTED_OPERATOR_TYPES,
    _cast_floating_array,
    _compose_permutations,
    _constant_binary,
    _last_axis_permutations,
    _layout_permutation,
    _native_layout,
    _normalize_precision,
    _physical_axis,
    _shape_table,
    _TensorValue,
    conv_transpose_spec,
    native_max_input_shape,
)


def test_fp16_precision_casts_only_floating_tensors() -> None:
    floating = _cast_floating_array(np.asarray([1.0, 2.0], dtype=np.float32), "FP16")
    integer = _cast_floating_array(np.asarray([1, 2], dtype=np.int64), "fp16")

    assert floating.dtype == np.float16
    assert floating.flags.c_contiguous
    assert integer.dtype == np.int64
    assert _normalize_precision(" fp32 ") == "fp32"


def test_unknown_precision_fails_closed() -> None:
    with pytest.raises(ValueError, match="unsupported precision"):
        _normalize_precision("int8")


def test_fruit_conv_transpose_maps_onnx_cfrs_to_max_rscf() -> None:
    onnx_weight = np.arange(2 * 3 * 2 * 2, dtype=np.float32).reshape(2, 3, 2, 2)

    max_filter, options = conv_transpose_spec(
        {
            "dilations": [1, 1],
            "group": 1,
            "kernel_shape": [2, 2],
            "pads": [0, 0, 0, 0],
            "strides": [2, 2],
        },
        onnx_weight,
    )

    assert max_filter.shape == (2, 2, 3, 2)
    assert max_filter[1, 0, 2, 1] == onnx_weight[1, 2, 1, 0]
    assert max_filter.flags.c_contiguous
    assert options == {
        "stride": (2, 2),
        "dilation": (1, 1),
        "padding": (0, 0, 0, 0),
        "output_paddings": (0, 0),
    }


def test_exact_exported_fruit_operator_set_is_covered() -> None:
    assert FRUIT_OPERATOR_TYPES <= SUPPORTED_OPERATOR_TYPES


def test_grouped_conv_transpose_fails_closed() -> None:
    with pytest.raises(NotImplementedError, match="grouped ConvTranspose"):
        conv_transpose_spec(
            {"group": 2},
            np.zeros((2, 2, 2, 2), dtype=np.float32),
        )


def test_shape_table_skips_symbolic_intermediates_but_keeps_static_values() -> None:
    class Dimension:
        def __init__(self, *, value: int = 0, parameter: str = "") -> None:
            self.dim_value = value
            self.dim_param = parameter

        def HasField(self, name: str) -> bool:
            return name == "dim_value" and self.dim_value > 0

    def value(name: str, dimensions: list[Dimension]):
        return SimpleNamespace(
            name=name,
            type=SimpleNamespace(
                tensor_type=SimpleNamespace(shape=SimpleNamespace(dim=dimensions))
            ),
        )

    model = SimpleNamespace(
        graph=SimpleNamespace(
            input=[value("input", [Dimension(value=1), Dimension(value=3)])],
            value_info=[
                value(
                    "symbolic_slice",
                    [Dimension(parameter="unk__0"), Dimension(value=2)],
                )
            ],
            output=[value("output", [Dimension(value=1), Dimension(value=2)])],
        )
    )

    assert _shape_table(model) == {"input": (1, 3), "output": (1, 2)}


def test_integer_division_constant_folding_matches_onnx_truncation() -> None:
    result = _constant_binary(
        "Div",
        np.asarray([5, -5], dtype=np.int64),
        np.asarray([2, 2], dtype=np.int64),
    )

    assert result.tolist() == [2, -2]
    assert result.dtype == np.int64


@pytest.mark.parametrize(
    ("rank", "axis", "forward", "inverse"),
    [
        (4, 2, [0, 1, 3, 2], [0, 1, 3, 2]),
        (4, -1, [0, 1, 2, 3], [0, 1, 2, 3]),
        (4, 0, [1, 2, 3, 0], [3, 0, 1, 2]),
    ],
)
def test_softmax_axis_can_be_moved_last_and_restored(
    rank: int,
    axis: int,
    forward: list[int],
    inverse: list[int],
) -> None:
    assert _last_axis_permutations(rank, axis) == (forward, inverse)


def test_consecutive_transposes_can_be_composed() -> None:
    assert _compose_permutations(
        [0, 2, 1, 3],
        [0, 2, 3, 1],
    ) == [0, 1, 3, 2]


def test_native_max_input_is_physical_nhwc() -> None:
    assert _native_layout(4) == NATIVE_NHWC_LAYOUT
    assert _native_layout(3) == (0, 1, 2)
    assert native_max_input_shape((1, 3, 640, 640)) == (1, 640, 640, 3)


def test_layout_permutations_cross_the_onnx_max_boundary_once() -> None:
    assert _layout_permutation((0, 1, 2, 3), NATIVE_NHWC_LAYOUT) == [0, 2, 3, 1]
    assert _layout_permutation(NATIVE_NHWC_LAYOUT, (0, 1, 2, 3)) == [0, 3, 1, 2]
    assert _physical_axis(NATIVE_NHWC_LAYOUT, 1) == 3
    assert _physical_axis(NATIVE_NHWC_LAYOUT, -2) == 1


def test_tensor_layout_rejects_invalid_axis_metadata() -> None:
    value = _TensorValue(object(), NATIVE_NHWC_LAYOUT)
    assert value.rank == 4

    with pytest.raises(ValueError, match="must be a permutation"):
        _TensorValue(object(), (0, 1, 1, 3))
