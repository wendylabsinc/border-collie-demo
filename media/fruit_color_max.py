"""Optional accelerated backends for the per-frame bbox colour band count.

`media.fruit_color` classifies every sampled pixel inside a detector box into a
red / orange / neither hue band. That inner loop is the only genuinely
CPU-bound, embarrassingly parallel step our perception pipeline runs per frame,
so it is the honest candidate for a Modular MAX kernel.

Nothing in here is required. The module imports cleanly with neither MAX nor
numpy installed, `accelerated_band_counts` then simply returns `None`, and
`media.fruit_color` runs its existing pure-Python loop. Every backend failure
latches the backend off for the process and degrades to that same loop, so a
MAX problem can never break a run.

Selection is by environment variable, default off:

    BORDER_COLLIE_COLOR_BACKEND=reference   # default - pure Python loop
    BORDER_COLLIE_COLOR_BACKEND=numpy       # vectorised numpy
    BORDER_COLLIE_COLOR_BACKEND=max         # Modular MAX graph
    BORDER_COLLIE_COLOR_MAX_DEVICE=cpu      # cpu (default) | accelerator | auto

The device default is `cpu` deliberately. Woof's Jetson Orin NX is sm_87, and
MAX has previously been observed emitting sm_80 cubins there that fail at
execution with CUDA_ERROR_NO_BINARY_FOR_GPU. See media/Dockerfile.max for the
install recipe and the current state of that limitation.
"""

from __future__ import annotations

import os
import threading
from typing import Any

# (valid, red, orange)
BandCounts = tuple[int, int, int]

_REFERENCE_ALIASES = frozenset({"", "off", "none", "disabled", "reference", "python"})
_NUMPY_ALIASES = frozenset({"numpy", "np", "vectorized", "vectorised"})
_MAX_ALIASES = frozenset({"max", "mojo", "max_mojo", "modular"})

_BACKEND_LOCK = threading.Lock()
_BACKEND_RESOLVED = False
_BACKEND: Any = None


def selected_backend_name() -> str:
    """Normalised value of the backend environment flag."""
    raw = os.environ.get("BORDER_COLLIE_COLOR_BACKEND", "")
    return raw.casefold().strip().replace("-", "_")


def selected_device_name() -> str:
    raw = os.environ.get("BORDER_COLLIE_COLOR_MAX_DEVICE", "cpu")
    return raw.casefold().strip() or "cpu"


def reset_backend_cache() -> None:
    """Drop the memoised backend so a new flag value is honoured (tests)."""
    global _BACKEND_RESOLVED, _BACKEND
    with _BACKEND_LOCK:
        _BACKEND_RESOLVED = False
        _BACKEND = None


def _resolve_backend() -> Any:
    global _BACKEND_RESOLVED, _BACKEND
    if _BACKEND_RESOLVED:
        return _BACKEND
    with _BACKEND_LOCK:
        if _BACKEND_RESOLVED:
            return _BACKEND
        name = selected_backend_name()
        backend: Any = None
        try:
            if name in _NUMPY_ALIASES:
                backend = NumpyBandCounter()
            elif name in _MAX_ALIASES:
                backend = MaxBandCounter(device=selected_device_name())
            elif name not in _REFERENCE_ALIASES:
                backend = None
        except Exception:  # noqa: BLE001 - degrading is always safer than raising
            # An unavailable or broken accelerator is never fatal; the caller
            # falls back to the pure-Python loop.
            backend = None
        _BACKEND = backend
        _BACKEND_RESOLVED = True
        return backend


def accelerated_band_counts(pixels: object) -> BandCounts | None:
    """Return (valid, red, orange) from an accelerated backend, or None.

    Returns None whenever the accelerated path is disabled, unavailable, or
    cannot handle this input, which tells the caller to use the reference loop.
    Never raises.
    """
    backend = _resolve_backend()
    if backend is None:
        return None
    try:
        return backend.count(pixels)
    except Exception:  # noqa: BLE001 - a kernel fault must never break a run
        # Latch off after a runtime failure so we degrade once, not per frame.
        reset_backend_cache()
        global _BACKEND_RESOLVED, _BACKEND
        with _BACKEND_LOCK:
            _BACKEND = None
            _BACKEND_RESOLVED = True
        return None


def _as_pixel_array(pixels: object) -> Any:
    """Coerce list-of-BGR or an ndarray region into a finite (n, 3) float64 array."""
    import numpy as np

    array = np.asarray(pixels, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] < 3:
        raise ValueError("pixel input must be (n, 3) BGR")
    return np.ascontiguousarray(array[:, :3])


class NumpyBandCounter:
    """Vectorised numpy implementation of the reference hue-band count."""

    name = "numpy"

    def __init__(self) -> None:
        import numpy  # noqa: F401  - fail fast when numpy is absent

    def count(self, pixels: object) -> BandCounts:
        import numpy as np

        array = _as_pixel_array(pixels)
        if array.shape[0] == 0:
            return (0, 0, 0)
        blue = array[:, 0]
        green = array[:, 1]
        red_channel = array[:, 2]

        finite = np.isfinite(array).all(axis=1)
        maximum = array.max(axis=1)
        minimum = array.min(axis=1)
        delta = maximum - minimum
        threshold = np.maximum(30.0, maximum * 0.25)
        valid_mask = finite & (maximum >= 40.0) & (delta >= threshold)

        consider = valid_mask & (red_channel >= np.maximum(green, blue))
        # Non-considered pixels can hold inf/NaN channels; their hue is never
        # read, so suppress the arithmetic warnings they would otherwise raise
        # on every frame.
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            safe_delta = np.where(consider, delta, 1.0)
            hue_raw = 60.0 * ((green - blue) / safe_delta)
            hue = hue_raw - 360.0 * np.floor(hue_raw / 360.0)
            red_mask = consider & ((hue <= 8.0) | (hue >= 345.0))
            orange_mask = consider & ~red_mask & (hue <= 80.0)
        return (
            int(valid_mask.sum()),
            int(red_mask.sum()),
            int(orange_mask.sum()),
        )


class MaxBandCounter:
    """Modular MAX graph implementation of the reference hue-band count.

    The graph is built entirely from add / subtract / multiply / divide / relu /
    floor / comparison / cast / sum primitives so it lowers cleanly on CPU. All
    arithmetic is float64 to match CPython float semantics exactly, which makes
    the band counts bit-identical to the reference loop rather than merely
    close. min/max are expressed via relu (max(a, b) == b + relu(a - b)) and
    boolean logic via 0.0/1.0 float masks, avoiding any op whose reduction
    versus elementwise overload could differ between MAX releases.
    """

    name = "max"

    def __init__(self, *, device: str = "cpu") -> None:
        import numpy as np
        from max.driver import CPU, Accelerator, Buffer, accelerator_count
        from max.dtype import DType
        from max.engine import InferenceSession
        from max.graph import DeviceRef, Graph, TensorType, ops

        normalized = device.casefold().strip()
        if normalized not in {"auto", "cpu", "accelerator", "gpu"}:
            raise ValueError("colour backend device must be auto, cpu, or accelerator")
        available = int(accelerator_count())
        if normalized == "cpu":
            resolved = CPU()
        elif normalized in {"accelerator", "gpu"}:
            if available == 0:
                raise RuntimeError("no MAX accelerator is available")
            resolved = Accelerator()
        else:
            resolved = Accelerator() if available else CPU()

        device_ref = DeviceRef.from_device(resolved)
        pixel_type = TensorType(DType.float64, shape=["pixels", 3], device=device_ref)

        with Graph("collie_colour_bands", input_types=[pixel_type]) as graph:
            pixels = graph.inputs[0]
            blue, green, red_channel = ops.split(pixels, [1, 1, 1], axis=1)

            def constant(value: float) -> Any:
                return ops.constant(value, DType.float64, device_ref)

            def mask(condition: Any) -> Any:
                return ops.cast(condition, DType.float64)

            def at_most(value: Any, limit: float) -> Any:
                # `less_equal` is not exported; a >= b with operands swapped.
                return mask(ops.greater_equal(constant(limit), value))

            def at_least(value: Any, limit: float) -> Any:
                return mask(ops.greater_equal(value, constant(limit)))

            # max(b, g) and min(b, g) without an elementwise max/min op.
            blue_over_green = ops.relu(blue - green)
            bg_max = green + blue_over_green
            bg_min = blue - blue_over_green
            maximum = red_channel + ops.relu(bg_max - red_channel)
            minimum = bg_min - ops.relu(bg_min - red_channel)
            delta = maximum - minimum

            # The reference skips any pixel with a non-finite channel. NaN
            # fails both comparisons and +/-inf fails one, so this reproduces
            # `all(math.isfinite(...))` without an is_nan/is_inf op.
            finite = (
                at_most(blue, 1e308)
                * at_least(blue, -1e308)
                * at_most(green, 1e308)
                * at_least(green, -1e308)
                * at_most(red_channel, 1e308)
                * at_least(red_channel, -1e308)
            )

            # threshold = max(30.0, maximum * 0.25)
            threshold = constant(30.0) + ops.relu(
                maximum * constant(0.25) - constant(30.0)
            )
            valid_mask = (
                finite
                * at_least(maximum, 40.0)
                * mask(ops.greater_equal(delta, threshold))
            )
            consider = valid_mask * mask(ops.greater_equal(red_channel, bg_max))

            # Where considered, delta >= 30 so this is exact; elsewhere it only
            # keeps the division finite before the mask discards the result.
            safe_delta = delta + (constant(1.0) - consider)
            hue_raw = constant(60.0) * ((green - blue) / safe_delta)
            hue = hue_raw - constant(360.0) * ops.floor(hue_raw / constant(360.0))

            # The two red sub-bands are disjoint, so summing the masks is OR.
            red_mask = consider * (at_most(hue, 8.0) + at_least(hue, 345.0))
            orange_mask = consider * (constant(1.0) - red_mask) * at_most(hue, 80.0)

            totals = ops.concat(
                [
                    ops.sum(valid_mask, axis=0),
                    ops.sum(red_mask, axis=0),
                    ops.sum(orange_mask, axis=0),
                ],
                axis=1,
            )
            graph.output(totals)

        self._numpy = np
        self._buffer = Buffer
        self._device = resolved
        self._cpu = CPU()
        self.device_name = str(resolved)
        self._session = InferenceSession(devices=[resolved])
        self._model = self._session.load(graph)

    def count(self, pixels: object) -> BandCounts:
        np = self._numpy
        array = _as_pixel_array(pixels)
        if array.shape[0] == 0:
            return (0, 0, 0)
        buffer = self._buffer.from_numpy(array).to(self._device)
        output = self._model.execute(buffer)[0]
        totals = np.asarray(output.to(self._cpu).to_numpy()).reshape(-1)
        if totals.shape[0] != 3:
            raise RuntimeError("MAX colour band graph returned an unexpected shape")
        # Counts are sums of 0.0/1.0 in float64, so they are exact integers.
        return (int(totals[0]), int(totals[1]), int(totals[2]))
