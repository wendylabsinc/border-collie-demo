"""Parity and fallback tests for the vectorised colour band backend.

The flag/fallback tests run everywhere, including the numpy-free project test
venv. The numeric parity tests skip unless numpy is importable, so the
vectorised path is verified wherever it can actually be built.
"""

from __future__ import annotations

import importlib.util

import pytest

from media import fruit_color_backend
from media.fruit_color import (
    classify_bgr_pixels,
    count_band_pixels,
    summarize_band_counts,
)

# Counts are integers and every derived field is a ratio of those integers, so
# the honest tolerance for this kernel is exact equality on the counts. The
# tolerance below applies only to the derived floating-point evidence fields.
ABSOLUTE_TOLERANCE = 0.0

NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None

requires_numpy = pytest.mark.skipif(not NUMPY_AVAILABLE, reason="numpy not installed")


@pytest.fixture(autouse=True)
def _clear_backend_cache():
    fruit_color_backend.reset_backend_cache()
    yield
    fruit_color_backend.reset_backend_cache()


def _parity_cases() -> list[list[list[float]]]:
    """Pixel batches chosen to sit on every branch boundary in the kernel."""
    return [
        [[40.0, 150.0, 235.0]] * 8,  # plain orange
        [[20.0, 20.0, 200.0]] * 8,  # plain red
        [[128.0, 128.0, 128.0]] * 8,  # grey, fails the saturation gate
        [[10.0, 10.0, 12.0]] * 8,  # too dark, fails the brightness gate
        [[0.0, 0.0, 40.0]],  # exactly the 40.0 brightness boundary
        [[0.0, 0.0, 39.999999999999996]],  # one ulp under it
        [[0.0, 0.0, 30.0]],  # exactly the 30.0 delta boundary, below brightness
        [[10.0, 10.0, 40.0]],  # exactly the 30.0 delta boundary, at brightness
        [[10.0, 10.0, 39.99999999999999]],  # one ulp under the delta boundary
        [[0.0, 0.0, 160.0]],  # exactly the maximum * 0.25 boundary
        [[40.0, 40.0, 160.0]],  # exactly at maximum * 0.25, delta 120
        [[41.0, 41.0, 160.0]],  # one unit under maximum * 0.25
        [[200.0, 100.0, 50.0]],  # blue dominant, counted valid then skipped
        [[100.0, 200.0, 200.0]],  # red ties the green maximum, still considered
        [[0.0, 8.0, 60.0]],  # low hue, red band
        [[8.0, 0.0, 60.0]],  # negative hue, wraps past 345 into the red band
        [[60.0, 52.0, 120.0]],  # hue exactly -8 -> 352, red band via wraparound
        [[0.0, 255.0, 255.0]],  # hue exactly 60
        [[255.0, 0.0, 255.0]],  # hue exactly 300, outside every band
        [[0.0, 60.0, 60.0]],  # hue 60, orange band
        [[0.0, 60.0, 61.0]],  # just off the 60/61 ratio
        [[120.0, 0.0, 120.0]],  # hue exactly -60 -> 300, outside every band
        [[0.0, 34.0, 255.0]],  # hue on the 8.0 red/orange split
        [[0.0, 0.0, 255.0]],  # hue exactly 0.0
        [[float("nan"), 100.0, 200.0]],  # non-finite channel is skipped
        [[float("inf"), 100.0, 200.0]],
        [[100.0, float("-inf"), 200.0]],
        [[float("nan"), float("nan"), float("nan")]],
        [[0.0, 0.0, 255.0], [255.0, 255.0, 0.0], [0.0, 255.0, 0.0]],
        [[0.0, 0.0, 255.0], [float("nan"), 1.0, 2.0], [40.0, 150.0, 235.0]],
        [],
    ]


def _random_case(seed: int, count: int = 400) -> list[list[float]]:
    import numpy as np

    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(count, 3)).astype(float).tolist()


def _assert_parity(backend, pixels) -> None:
    expected = count_band_pixels(list(pixels))
    actual = backend.count(pixels) if len(pixels) else (0, 0, 0)
    assert actual == expected, f"counts differ for {list(pixels)[:4]}"

    expected_summary = summarize_band_counts(*expected)
    actual_summary = summarize_band_counts(*actual)
    for key, expected_value in expected_summary.items():
        actual_value = actual_summary[key]
        if isinstance(expected_value, float):
            assert abs(actual_value - expected_value) <= ABSOLUTE_TOLERANCE, key
        else:
            assert actual_value == expected_value, key


# --- flag and fallback behaviour (no numpy required) -------------------------


@pytest.mark.parametrize("value", ["off", "none", "disabled", "reference", "python"])
def test_reference_aliases_disable_acceleration(monkeypatch, value):
    monkeypatch.setenv("BORDER_COLLIE_COLOR_BACKEND", value)
    fruit_color_backend.reset_backend_cache()
    assert fruit_color_backend.accelerated_band_counts([[40.0, 150.0, 235.0]]) is None
    assert fruit_color_backend.active_backend_name() == "reference"


def test_unknown_backend_name_falls_back(monkeypatch):
    monkeypatch.setenv("BORDER_COLLIE_COLOR_BACKEND", "nonsense")
    fruit_color_backend.reset_backend_cache()
    assert fruit_color_backend.accelerated_band_counts([[40.0, 150.0, 235.0]]) is None
    assert fruit_color_backend.active_backend_name() == "reference"


def test_classification_matches_reference_when_flag_is_off(monkeypatch):
    monkeypatch.setenv("BORDER_COLLIE_COLOR_BACKEND", "reference")
    fruit_color_backend.reset_backend_cache()
    pixels = [[40.0, 150.0, 235.0]] * 40
    assert classify_bgr_pixels(pixels) == summarize_band_counts(
        *count_band_pixels(pixels)
    )


def test_backend_failure_latches_off_and_degrades_to_reference(monkeypatch):
    class ExplodingBackend:
        name = "exploding"

        def count(self, pixels):
            raise RuntimeError("kernel exploded")

    monkeypatch.setattr(fruit_color_backend, "_BACKEND", ExplodingBackend())
    monkeypatch.setattr(fruit_color_backend, "_BACKEND_RESOLVED", True)
    pixels = [[40.0, 150.0, 235.0]] * 40

    assert fruit_color_backend.accelerated_band_counts(pixels) is None
    # The failure latches the backend off rather than retrying every frame.
    assert fruit_color_backend._BACKEND is None
    assert classify_bgr_pixels(pixels)["identity"] == "orange"


def test_unrepresentable_input_degrades_without_latching_off(monkeypatch):
    class PickyBackend:
        name = "picky"

        def count(self, pixels):
            raise ValueError("ragged rows")

    picky = PickyBackend()
    monkeypatch.setattr(fruit_color_backend, "_BACKEND", picky)
    monkeypatch.setattr(fruit_color_backend, "_BACKEND_RESOLVED", True)

    assert fruit_color_backend.accelerated_band_counts([[1.0, 2.0]]) is None
    # A batch this backend cannot represent is not a backend fault.
    assert fruit_color_backend._BACKEND is picky


def test_summarize_band_counts_is_shared_and_pure():
    assert summarize_band_counts(0, 0, 0)["identity"] == "unknown"
    assert summarize_band_counts(100, 5, 90)["identity"] == "orange"
    assert summarize_band_counts(100, 90, 5)["identity"] == "red_apple"


def test_module_imports_and_classifies_without_numpy(monkeypatch):
    """With numpy absent the backend never resolves and the loop still runs."""
    import builtins

    real_import = builtins.__import__

    def _no_numpy(name, *args, **kwargs):
        if name == "numpy" or name.startswith("numpy."):
            raise ImportError("numpy is not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_numpy)
    monkeypatch.delenv("BORDER_COLLIE_COLOR_BACKEND", raising=False)
    fruit_color_backend.reset_backend_cache()

    assert fruit_color_backend.active_backend_name() == "reference"
    assert fruit_color_backend.accelerated_band_counts([[40.0, 150.0, 235.0]]) is None
    pixels = [[40.0, 150.0, 235.0]] * 40
    assert classify_bgr_pixels(pixels) == summarize_band_counts(
        *count_band_pixels(pixels)
    )


def test_malformed_pixel_batches_still_match_the_reference(monkeypatch):
    """Inputs numpy cannot represent fall back rather than changing the answer."""
    monkeypatch.delenv("BORDER_COLLIE_COLOR_BACKEND", raising=False)
    fruit_color_backend.reset_backend_cache()
    batches = [
        [[40.0, 150.0, 235.0], [1.0, 2.0]],  # ragged
        [[40.0, 150.0, 235.0], "xyz"],  # non-sequence entry
        [[40.0, 150.0, 235.0], None],
        [1.0, 2.0, 3.0],  # flat, not (n, 3)
        [[40.0, 150.0, 235.0, 255.0]] * 4,  # BGRA, extra channel ignored
        [(40.0, 150.0, 235.0)] * 4,  # tuples
        [[40, 150, 235]] * 4,  # integers
        [["40", "150", "235"]] * 4,  # numeric strings
        [{"b": 1}],  # mapping entry
    ]
    for batch in batches:
        expected = summarize_band_counts(*count_band_pixels(batch))
        assert classify_bgr_pixels(batch) == expected, batch


# --- numeric parity ---------------------------------------------------------


@requires_numpy
def test_numpy_is_the_default_backend_when_available(monkeypatch):
    monkeypatch.delenv("BORDER_COLLIE_COLOR_BACKEND", raising=False)
    fruit_color_backend.reset_backend_cache()
    assert fruit_color_backend.active_backend_name() == "numpy"
    assert fruit_color_backend.accelerated_band_counts([[40.0, 150.0, 235.0]]) == (
        1,
        0,
        1,
    )


@requires_numpy
@pytest.mark.parametrize("value", ["auto", "default", "numpy", "np", "vectorised"])
def test_numpy_aliases_select_the_vectorised_backend(monkeypatch, value):
    monkeypatch.setenv("BORDER_COLLIE_COLOR_BACKEND", value)
    fruit_color_backend.reset_backend_cache()
    assert fruit_color_backend.active_backend_name() == "numpy"


@requires_numpy
@pytest.mark.parametrize("case_index", range(len(_parity_cases())))
def test_numpy_backend_matches_reference(case_index):
    pixels = _parity_cases()[case_index]
    if not pixels:
        pytest.skip("empty batch is handled before the backend is consulted")
    _assert_parity(fruit_color_backend.NumpyBandCounter(), pixels)


@requires_numpy
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_numpy_backend_matches_reference_on_random_pixels(seed):
    _assert_parity(fruit_color_backend.NumpyBandCounter(), _random_case(seed))


@requires_numpy
def test_numpy_backend_matches_reference_on_a_strided_uint8_sweep():
    """Every 3rd value of the full uint8 BGR cube, 636,056 pixels."""
    import numpy as np

    axis = np.arange(0, 256, 3, dtype=np.float64)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    region = np.ascontiguousarray(grid.reshape(-1, 3))
    expected = count_band_pixels(region.tolist())
    assert fruit_color_backend.NumpyBandCounter().count(region) == expected
    # A sweep that exercised no band would prove nothing.
    assert expected[0] > 0 and expected[1] > 0 and expected[2] > 0


@requires_numpy
def test_numpy_backend_matches_reference_on_hue_boundary_ratios():
    """Hues placed on 8, 60, 80, 300 and 345 degrees, positive and negative."""
    pixels = []
    for degrees in (0.0, 8.0, 60.0, 80.0, 300.0, 345.0, 352.0, 359.0):
        # hue = (60 * (green - blue) / delta) % 360, so pick green - blue to hit
        # the target for a red-dominant pixel with delta = 240. Degrees above
        # 180 are produced as a negative hue that must wrap in both backends.
        offset = degrees if degrees <= 180.0 else degrees - 360.0
        difference = offset / 60.0 * 240.0
        blue = max(0.0, -difference)
        green = max(0.0, difference)
        pixels.append([blue, green, max(blue, green) + 240.0])
    _assert_parity(fruit_color_backend.NumpyBandCounter(), pixels)


@requires_numpy
def test_numpy_backend_matches_reference_on_a_realistic_box():
    import numpy as np

    from media.fruit_color_benchmark import _sample_region

    region = _sample_region(4096, seed=11)
    expected = count_band_pixels(region.tolist())
    assert fruit_color_backend.NumpyBandCounter().count(region) == expected
    assert np.isfinite(region).all()


@requires_numpy
def test_classify_bbox_color_agrees_across_backends(monkeypatch):
    """The ndarray fast path and the .tolist() loop return the same evidence."""
    import numpy as np

    from media.fruit_color import classify_bbox_color

    rng = np.random.default_rng(3)
    source = rng.integers(0, 256, size=(240, 320, 3)).astype(np.uint8)
    boxes = [
        (10.0, 20.0, 200.0, 180.0),
        (0.0, 0.0, 320.0, 240.0),
        (100.5, 100.5, 104.5, 104.5),
        (300.0, 200.0, 400.0, 300.0),
    ]

    monkeypatch.setenv("BORDER_COLLIE_COLOR_BACKEND", "reference")
    fruit_color_backend.reset_backend_cache()
    reference = [classify_bbox_color(source, box) for box in boxes]

    monkeypatch.setenv("BORDER_COLLIE_COLOR_BACKEND", "numpy")
    fruit_color_backend.reset_backend_cache()
    vectorised = [classify_bbox_color(source, box) for box in boxes]

    assert vectorised == reference
