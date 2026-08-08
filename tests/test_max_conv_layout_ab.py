from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROBE_DIR = Path(__file__).parents[1] / "lab" / "max-conv-layout-ab"
sys.path.insert(0, str(PROBE_DIR))
from probe import (  # noqa: E402
    INPUT_SHAPE,
    KERNEL_SIZE,
    OUTPUT_CHANNELS,
    evaluate_result,
    make_weight_variants,
)


def test_weight_variants_preserve_one_convolution_kernel() -> None:
    fcrs, rscf = make_weight_variants()

    assert fcrs.shape == (
        OUTPUT_CHANNELS,
        INPUT_SHAPE[-1],
        KERNEL_SIZE,
        KERNEL_SIZE,
    )
    assert rscf.shape == (
        KERNEL_SIZE,
        KERNEL_SIZE,
        INPUT_SHAPE[-1],
        OUTPUT_CHANNELS,
    )
    assert fcrs.flags.c_contiguous
    assert rscf.flags.c_contiguous
    np.testing.assert_array_equal(fcrs, np.transpose(rscf, (3, 2, 0, 1)))


def test_verdict_requires_parity_and_a_large_layout_speedup() -> None:
    confirmed = evaluate_result(
        rscf_median_ms=50.0,
        fcrs_median_ms=1.0,
        maximum_absolute_error=0.01,
    )
    parity_failure = evaluate_result(
        rscf_median_ms=50.0,
        fcrs_median_ms=1.0,
        maximum_absolute_error=0.5,
    )
    small_speedup = evaluate_result(
        rscf_median_ms=4.0,
        fcrs_median_ms=1.0,
        maximum_absolute_error=0.01,
    )

    assert confirmed["confirms_dispatch_hypothesis"] is True
    assert parity_failure["confirms_dispatch_hypothesis"] is False
    assert small_speedup["confirms_dispatch_hypothesis"] is False
