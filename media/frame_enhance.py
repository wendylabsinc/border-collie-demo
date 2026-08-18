"""Live-tunable contrast enhancement for the frame handed to the detector.

The stage this demo runs on is not the lab it was tuned in. In the lab all three
fruits read 0.88-0.95; in a dim room with windows clipped to white the same
weights on the same frames read 0.05-0.64, and auto-exposure swings the mean
luminance from 34 to 106 across a single search rotation. This module exists so
an operator can try to recover that signal from the frame itself.

Two things matter about the design, both learned the hard way:

**It must be tunable without a redeploy.** The operator gets one short window on
the robot and needs to change the setting between runs, not between builds. So
the boot default comes from the environment and everything after that is a live
HTTP control.

**It must be luminance-only.** The detector is a YOLOE visual-prompt checkpoint
whose classes are single baked exemplar embeddings, and `media.fruit_color` keys
the mango derived route off hue bands. Stretching R, G and B independently
shifts hue and would break both. Every mode here touches only the luminance
channel and leaves chroma untouched.

`off` is the default and returns the caller's own array, so the deployed path is
byte-identical and costs nothing until someone asks for more.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, replace
from typing import Any

MODES: tuple[str, ...] = ("off", "autocontrast", "clahe")

# Per-mode strength meaning and accepted range. `off` ignores strength but must
# still round-trip a value rather than erroring, so a client can set mode and
# strength in either order without tracking which is valid right now.
STRENGTH_RANGES: dict[str, tuple[float, float]] = {
    "off": (0.0, 8.0),
    "autocontrast": (0.0, 10.0),  # percent clipped from each tail
    "clahe": (1.0, 8.0),  # clip limit
}
DEFAULT_STRENGTH: dict[str, float] = {
    "off": 0.0,
    "autocontrast": 1.0,
    "clahe": 2.0,
}
# CLAHE tile grid; 8 is OpenCV's own default and is a reasonable floor/ceiling
# for a 1280x720 frame (160x90 px tiles).
TILE_GRID_RANGE: tuple[int, int] = (2, 16)
DEFAULT_TILE_GRID = 8


class FrameEnhanceError(ValueError):
    """A requested setting is not usable; the current setting is unchanged."""


@dataclass(frozen=True)
class EnhanceSettings:
    mode: str = "off"
    strength: float = 0.0
    tile_grid: int = DEFAULT_TILE_GRID

    def as_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "strength": self.strength,
            "tile_grid": self.tile_grid,
        }

    def overlay_label(self) -> str:
        """Terse banner text, or empty when off so the preview is unchanged."""
        if self.mode == "off":
            return ""
        if self.mode == "clahe":
            return f"CLAHE {self.strength:.1f}/{self.tile_grid}"
        return f"AUTOCONTRAST {self.strength:.1f}"


def resolve_mode(raw: object) -> str:
    """Normalise a mode, falling back to `off` for anything unrecognised."""
    text = str(raw or "").strip().casefold().replace("-", "_")
    if text in MODES:
        return text
    return "off"


def validate(
    *,
    mode: object = None,
    strength: object = None,
    tile_grid: object = None,
    current: EnhanceSettings,
) -> EnhanceSettings:
    """Build a new settings object from a partial request.

    Omitted fields keep their current value, so `{"mode": "clahe"}` and
    `{"strength": 3.0}` are both valid on their own. Raises
    `FrameEnhanceError` without touching `current` when anything is unusable.
    """
    if mode is None:
        resolved_mode = current.mode
    else:
        text = str(mode).strip().casefold().replace("-", "_")
        if text not in MODES:
            raise FrameEnhanceError(
                f"mode must be one of {', '.join(MODES)}; got {mode!r}"
            )
        resolved_mode = text

    if strength is None:
        # Carry the current strength when it still fits the (possibly new) mode,
        # otherwise fall back to that mode's default rather than rejecting a
        # mode-only change.
        low, high = STRENGTH_RANGES[resolved_mode]
        resolved_strength = (
            current.strength
            if low <= current.strength <= high
            else DEFAULT_STRENGTH[resolved_mode]
        )
    else:
        if isinstance(strength, bool):
            raise FrameEnhanceError("strength must be a number")
        try:
            resolved_strength = float(strength)
        except (TypeError, ValueError) as exc:
            raise FrameEnhanceError("strength must be a number") from exc
        low, high = STRENGTH_RANGES[resolved_mode]
        if not low <= resolved_strength <= high:
            raise FrameEnhanceError(
                f"strength for {resolved_mode} must stay within "
                f"{low:.1f}..{high:.1f}; got {resolved_strength}"
            )

    if tile_grid is None:
        resolved_grid = current.tile_grid
    else:
        if isinstance(tile_grid, bool) or not isinstance(tile_grid, (int, float)):
            raise FrameEnhanceError("tile_grid must be an integer")
        if float(tile_grid) != int(tile_grid):
            raise FrameEnhanceError("tile_grid must be an integer")
        resolved_grid = int(tile_grid)
        low_grid, high_grid = TILE_GRID_RANGE
        if not low_grid <= resolved_grid <= high_grid:
            raise FrameEnhanceError(
                f"tile_grid must stay within {low_grid}..{high_grid}"
            )

    return replace(
        current,
        mode=resolved_mode,
        strength=resolved_strength,
        tile_grid=resolved_grid,
    )


class FrameEnhancer:
    """Holds the live setting and applies it.

    The frame loop reads the setting on every frame while an HTTP handler may be
    writing it, so reads take an immutable snapshot under a lock and the loop
    never observes a half-applied change.
    """

    def __init__(self, settings: EnhanceSettings | None = None) -> None:
        self._settings = settings or EnhanceSettings()
        self._lock = threading.Lock()
        self._backend_error: str | None = None

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> FrameEnhancer:
        import os

        env = environ if environ is not None else dict(os.environ)
        mode = resolve_mode(env.get("BORDER_COLLIE_FRAME_ENHANCE"))
        raw_strength = env.get("BORDER_COLLIE_FRAME_ENHANCE_STRENGTH")
        strength = DEFAULT_STRENGTH[mode]
        if raw_strength:
            try:
                candidate = float(raw_strength)
            except ValueError:
                candidate = strength
            low, high = STRENGTH_RANGES[mode]
            if low <= candidate <= high:
                strength = candidate
        return cls(EnhanceSettings(mode=mode, strength=strength))

    def settings(self) -> EnhanceSettings:
        with self._lock:
            return self._settings

    def apply_request(
        self,
        *,
        mode: object = None,
        strength: object = None,
        tile_grid: object = None,
    ) -> EnhanceSettings:
        """Validate then swap atomically. Invalid input leaves the current setting."""
        with self._lock:
            updated = validate(
                mode=mode,
                strength=strength,
                tile_grid=tile_grid,
                current=self._settings,
            )
            self._settings = updated
            return updated

    def status(self) -> dict[str, object]:
        settings = self.settings()
        payload = settings.as_dict()
        payload["modes"] = list(MODES)
        payload["strength_ranges"] = {k: list(v) for k, v in STRENGTH_RANGES.items()}
        payload["tile_grid_range"] = list(TILE_GRID_RANGE)
        payload["backend_error"] = self._backend_error
        return payload

    def enhance(self, bgr: Any) -> Any:
        """Return the frame the detector should see.

        Returns the caller's own array unchanged when off, or when the backend is
        unavailable or fails — a contrast tweak must never cost us a frame.
        """
        settings = self.settings()
        if settings.mode == "off":
            return bgr
        try:
            return _apply(bgr, settings)
        except Exception as exc:  # noqa: BLE001 - never lose a frame over this
            self._backend_error = f"{type(exc).__name__}: {exc}"
            return bgr


def _apply(bgr: Any, settings: EnhanceSettings) -> Any:
    import cv2

    # YCrCb rather than LAB: the Y channel is a direct luminance stand-in and the
    # round trip is cheaper, and chroma comes back untouched so hue is preserved.
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y = ycrcb[:, :, 0]
    if settings.mode == "clahe":
        clahe = cv2.createCLAHE(
            clipLimit=float(settings.strength),
            tileGridSize=(settings.tile_grid, settings.tile_grid),
        )
        ycrcb[:, :, 0] = clahe.apply(y)
    else:
        ycrcb[:, :, 0] = _percentile_stretch(y, settings.strength)
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def _percentile_stretch(y: Any, cutoff_percent: float) -> Any:
    """Linear contrast stretch ignoring the brightest/darkest `cutoff_percent`.

    Clipping the tails is what makes this useful on this stage: a handful of
    blown-out window pixels would otherwise pin the upper bound and leave the
    floor compressed into a few levels.
    """
    import cv2
    import numpy as np

    if cutoff_percent <= 0.0:
        low, high = float(y.min()), float(y.max())
    else:
        cutoff = min(max(cutoff_percent, 0.0), 49.0)
        low = float(np.percentile(y, cutoff))
        high = float(np.percentile(y, 100.0 - cutoff))
    if high - low < 1.0:
        return y
    scaled = (y.astype(np.float32) - low) * (255.0 / (high - low))
    return cv2.convertScaleAbs(scaled, alpha=1.0, beta=0.0)
