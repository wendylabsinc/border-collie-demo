"""Optional vectorised backend for the per-frame bbox colour band count.

`media.fruit_color` classifies every sampled pixel inside a detector box into a
red / orange / neither hue band. At our ~4096-pixel operating point that inner
loop is the most expensive pure-Python step the perception pipeline runs per
frame, and it is embarrassingly parallel, so numpy replaces it wholesale.

Nothing in here is required. The module imports cleanly with numpy absent,
`accelerated_band_counts` then simply returns `None`, and `media.fruit_color`
runs its existing pure-Python loop. Every backend failure latches the backend
off for the rest of the process and degrades to that same loop, so a numpy
problem can never break a run.

Selection is by environment variable:

    BORDER_COLLIE_COLOR_BACKEND=auto        # default - numpy when importable
    BORDER_COLLIE_COLOR_BACKEND=numpy       # require numpy, degrade if absent
    BORDER_COLLIE_COLOR_BACKEND=reference   # kill switch - pure-Python loop

`auto` is the default because the numpy path is not an approximation: the band
counts are integers and the backend reproduces all three of them exactly (see
tests/test_fruit_color_backend.py), the media container already ships numpy via
opencv/ultralytics, and an unavailable or faulting backend falls back on its
own. The `reference` kill switch exists so a suspected colour regression can be
A/B tested on the device without a rebuild.

The float64 arithmetic below mirrors CPython float semantics op for op:
`np.remainder` implements the same floored-modulo correction as CPython's
`float_rem`, so `hue % 360.0` agrees bit for bit, including for the negative
hues produced when blue exceeds green.
"""

from __future__ import annotations

import os
import threading
from typing import Any

# (valid, red, orange)
BandCounts = tuple[int, int, int]

_AUTO_ALIASES = frozenset({"", "auto", "default"})
_REFERENCE_ALIASES = frozenset({"off", "none", "disabled", "reference", "python"})
_NUMPY_ALIASES = frozenset({"numpy", "np", "vectorized", "vectorised"})

_BACKEND_LOCK = threading.Lock()
_BACKEND_RESOLVED = False
_BACKEND: Any = None

# Input a backend cannot represent, as opposed to a backend fault. The
# reference loop skips such pixels one at a time, so these hand the batch back
# without latching the backend off.
_RECOVERABLE_INPUT_ERRORS = (TypeError, ValueError, IndexError)


def selected_backend_name() -> str:
    """Normalised value of the backend environment flag."""
    raw = os.environ.get("BORDER_COLLIE_COLOR_BACKEND", "")
    return raw.casefold().strip().replace("-", "_")


def reset_backend_cache() -> None:
    """Drop the memoised backend so a new flag value is honoured (tests)."""
    global _BACKEND_RESOLVED, _BACKEND
    with _BACKEND_LOCK:
        _BACKEND_RESOLVED = False
        _BACKEND = None


def active_backend_name() -> str:
    """Which counting path is live: 'numpy' or 'reference'. Never raises."""
    backend = _resolve_backend()
    return getattr(backend, "name", "reference")


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
            if name in _AUTO_ALIASES or name in _NUMPY_ALIASES:
                backend = NumpyBandCounter()
            # `reference` aliases and any unrecognised name stay on the loop.
        except Exception:  # noqa: BLE001 - degrading is always safer than raising
            # numpy absent or broken is never fatal; the caller uses the loop.
            backend = None
        _BACKEND = backend
        _BACKEND_RESOLVED = True
        return backend


def _latch_backend_off() -> None:
    global _BACKEND_RESOLVED, _BACKEND
    with _BACKEND_LOCK:
        _BACKEND = None
        _BACKEND_RESOLVED = True


def accelerated_band_counts(pixels: object) -> BandCounts | None:
    """Return (valid, red, orange) from the vectorised backend, or None.

    Returns None whenever the vectorised path is disabled, unavailable, or
    cannot handle this input, which tells the caller to use the reference loop.
    Never raises.
    """
    backend = _resolve_backend()
    if backend is None:
        return None
    try:
        return backend.count(pixels)
    except _RECOVERABLE_INPUT_ERRORS:
        # Ragged rows, non-numeric entries, wrong rank: not a backend fault.
        return None
    except Exception:  # noqa: BLE001 - a backend fault must never break a run
        # A genuine backend fault latches off, so we degrade once, not per frame.
        _latch_backend_off()
        return None


def _as_pixel_array(pixels: object) -> Any:
    """Coerce list-of-BGR or an ndarray region into an (n, 3) float64 array."""
    import numpy as np

    array = np.asarray(pixels, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] < 3:
        raise ValueError("pixel input must be (n, 3) BGR")
    return array[:, :3]


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

        # The reference skips any pixel with a non-finite channel before it
        # computes anything else, so this gate comes first here too.
        finite = np.isfinite(array).all(axis=1)
        # Non-finite rows still flow through the arithmetic below; their masks
        # are cleared by `finite`, so suppress the warnings they would raise on
        # every frame.
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            maximum = array.max(axis=1)
            minimum = array.min(axis=1)
            delta = maximum - minimum
            # threshold = max(30.0, maximum * 0.25)
            threshold = np.maximum(30.0, maximum * 0.25)
            valid_mask = finite & (maximum >= 40.0) & (delta >= threshold)

            # `red_channel < max(green, blue)` is counted valid but banded as
            # neither, matching the reference's `continue` after `valid += 1`.
            consider = valid_mask & (red_channel >= np.maximum(green, blue))

            # Considered pixels always have delta >= 30.0, so this substitution
            # only keeps the division finite for rows already masked out.
            safe_delta = np.where(consider, delta, 1.0)
            hue = np.remainder(60.0 * ((green - blue) / safe_delta), 360.0)
            red_mask = consider & ((hue <= 8.0) | (hue >= 345.0))
            orange_mask = consider & ~red_mask & (hue <= 80.0)

        return (
            int(valid_mask.sum()),
            int(red_mask.sum()),
            int(orange_mask.sum()),
        )
