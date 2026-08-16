"""Parity and fallback tests for the optional accelerated colour backends.

The flag/fallback tests run everywhere, including the numpy-free test venv.
The numeric parity tests skip unless numpy (and MAX) are importable, so the
accelerated path is verified wherever it can actually be built.
"""

from __future__ import annotations

import importlib.util

import pytest

from media import fruit_color_max
from media.fruit_color import (
    classify_bgr_pixels,
    count_band_pixels,
    summarize_band_counts,
)

# Counts are integers and every derived field is a ratio of those integers, so
# the honest tolerance for this kernel is exact equality on the counts. The
# tolerance below applies to the derived floating-point evidence fields.
ABSOLUTE_TOLERANCE = 1e-12

NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None
MAX_AVAILABLE = importlib.util.find_spec("max") is not None

requires_numpy = pytest.mark.skipif(not NUMPY_AVAILABLE, reason="numpy not installed")
requires_max = pytest.mark.skipif(not MAX_AVAILABLE, reason="MAX not installed")


@pytest.fixture(autouse=True)
def _clear_backend_cache():
    fruit_color_max.reset_backend_cache()
    yield
    fruit_color_max.reset_backend_cache()


def _parity_cases() -> list[list[list[float]]]:
    """Pixel batches chosen to sit on every branch boundary in the kernel."""
    return [
        [[40.0, 150.0, 235.0]] * 8,  # plain orange
        [[20.0, 20.0, 200.0]] * 8,  # plain red
        [[128.0, 128.0, 128.0]] * 8,  # grey, fails the saturation gate
        [[10.0, 10.0, 12.0]] * 8,  # too dark, fails the brightness gate
        [[0.0, 0.0, 40.0]],  # exactly the 40.0 brightness boundary
        [[0.0, 0.0, 30.0]],  # exactly the 30.0 delta boundary
        [[200.0, 100.0, 50.0]],  # blue dominant, skipped after counting valid
        [[0.0, 8.0, 60.0]],  # low hue, red band
        [[8.0, 0.0, 60.0]],  # negative hue, wraps past 345 into the red band
        [[0.0, 60.0, 60.0]],  # hue 60, orange band
        [[0.0, 60.0, 61.0]],  # just off the 60/61 ratio
        [[0.0, 255.0, 255.0]],  # hue exactly 60
        [[255.0, 0.0, 255.0]],  # hue exactly 300, outside every band
        [[float("nan"), 100.0, 200.0]],  # non-finite channel is skipped
        [[float("inf"), 100.0, 200.0]],
        [[100.0, float("-inf"), 200.0]],
        [[0.0, 0.0, 255.0], [255.0, 255.0, 0.0], [0.0, 255.0, 0.0]],
        [],
    ]


def _random_case(seed: int, count: int = 400) -> list[list[float]]:
    import numpy as np

    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(count, 3)).astype(float).tolist()


def _assert_parity(backend, pixels) -> None:
    expected = count_band_pixels(list(pixels))
    actual = backend.count(pixels) if pixels else (0, 0, 0)
    assert actual == expected, f"counts differ for {pixels[:4]}"

    expected_summary = summarize_band_counts(*expected)
    actual_summary = summarize_band_counts(*actual)
    for key, expected_value in expected_summary.items():
        actual_value = actual_summary[key]
        if isinstance(expected_value, float):
            assert abs(actual_value - expected_value) <= ABSOLUTE_TOLERANCE, key
        else:
            assert actual_value == expected_value, key


# --- flag and fallback behaviour (no numpy or MAX required) ------------------


def test_backend_is_off_by_default(monkeypatch):
    monkeypatch.delenv("BORDER_COLLIE_COLOR_BACKEND", raising=False)
    fruit_color_max.reset_backend_cache()
    assert fruit_color_max.accelerated_band_counts([[40.0, 150.0, 235.0]]) is None


@pytest.mark.parametrize("value", ["", "off", "none", "disabled", "reference", "python"])
def test_reference_aliases_disable_acceleration(monkeypatch, value):
    monkeypatch.setenv("BORDER_COLLIE_COLOR_BACKEND", value)
    fruit_color_max.reset_backend_cache()
    assert fruit_color_max.accelerated_band_counts([[40.0, 150.0, 235.0]]) is None


def test_unknown_backend_name_falls_back(monkeypatch):
    monkeypatch.setenv("BORDER_COLLIE_COLOR_BACKEND", "nonsense")
    fruit_color_max.reset_backend_cache()
    assert fruit_color_max.accelerated_band_counts([[40.0, 150.0, 235.0]]) is None


def test_classification_matches_reference_when_flag_is_off(monkeypatch):
    monkeypatch.delenv("BORDER_COLLIE_COLOR_BACKEND", raising=False)
    fruit_color_max.reset_backend_cache()
    pixels = [[40.0, 150.0, 235.0]] * 40
    assert classify_bgr_pixels(pixels) == summarize_band_counts(
        *count_band_pixels(pixels)
    )


def test_backend_failure_degrades_to_reference(monkeypatch):
    class ExplodingBackend:
        name = "exploding"

        def count(self, pixels):
            raise RuntimeError("kernel exploded")

    monkeypatch.setattr(fruit_color_max, "_BACKEND", ExplodingBackend())
    monkeypatch.setattr(fruit_color_max, "_BACKEND_RESOLVED", True)
    pixels = [[40.0, 150.0, 235.0]] * 40

    assert fruit_color_max.accelerated_band_counts(pixels) is None
    # The failure latches the backend off rather than retrying every frame.
    assert fruit_color_max._BACKEND is None
    assert classify_bgr_pixels(pixels)["identity"] == "orange"


def test_unavailable_backend_does_not_raise(monkeypatch):
    monkeypatch.setenv("BORDER_COLLIE_COLOR_BACKEND", "max")
    monkeypatch.setenv("BORDER_COLLIE_COLOR_MAX_DEVICE", "nonsense-device")
    fruit_color_max.reset_backend_cache()
    assert fruit_color_max.accelerated_band_counts([[40.0, 150.0, 235.0]]) is None


def test_summarize_band_counts_is_shared_and_pure():
    assert summarize_band_counts(0, 0, 0)["identity"] == "unknown"
    assert summarize_band_counts(100, 5, 90)["identity"] == "orange"
    assert summarize_band_counts(100, 90, 5)["identity"] == "red_apple"


# --- numeric parity ---------------------------------------------------------


@requires_numpy
@pytest.mark.parametrize("case_index", range(len(_parity_cases())))
def test_numpy_backend_matches_reference(case_index):
    pixels = _parity_cases()[case_index]
    if not pixels:
        pytest.skip("empty batch is handled before the backend is consulted")
    _assert_parity(fruit_color_max.NumpyBandCounter(), pixels)


@requires_numpy
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_numpy_backend_matches_reference_on_random_pixels(seed):
    _assert_parity(fruit_color_max.NumpyBandCounter(), _random_case(seed))


@requires_max
@requires_numpy
@pytest.mark.parametrize("case_index", range(len(_parity_cases())))
def test_max_backend_matches_reference(case_index):
    pixels = _parity_cases()[case_index]
    if not pixels:
        pytest.skip("empty batch is handled before the backend is consulted")
    _assert_parity(fruit_color_max.MaxBandCounter(device="cpu"), pixels)


@requires_max
@requires_numpy
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_max_backend_matches_reference_on_random_pixels(seed):
    _assert_parity(fruit_color_max.MaxBandCounter(device="cpu"), _random_case(seed))


@requires_max
@requires_numpy
def test_max_backend_matches_reference_on_realistic_box():
    import numpy as np

    from media.fruit_color_benchmark import _sample_region

    region = _sample_region(4096, seed=11)
    backend = fruit_color_max.MaxBandCounter(device="cpu")
    expected = count_band_pixels(region.tolist())
    assert backend.count(region) == expected
    assert fruit_color_max.NumpyBandCounter().count(region) == expected
    assert np.isfinite(region).all()
