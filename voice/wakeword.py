"""openWakeWord acoustic wake word (the real 'Hey Wendy').

A tiny custom-trained model scores each audio chunk 0-1 for the wake phrase; the
heavy ASR only runs after it fires. Exposes the same spot(frame) -> keyword|None
interface the pipeline's wake gate already uses, so it drops in where the sherpa
KWS was. Runs on ONNX Runtime (proven on-device).
"""

from __future__ import annotations

import numpy as np

# openWakeWord expects 16 kHz int16 frames; 1280 samples = 80 ms is its unit.
_CHUNK = 1280


class OpenWakeWordSpotter:
    def __init__(self, model_path: str, threshold: float = 0.5, key: str | None = None) -> None:
        import os

        from openwakeword.model import Model

        self.threshold = threshold
        # Accept either a path to a custom model (e.g. our trained
        # /models/hey_wendy.onnx) or the name of a pretrained openWakeWord model
        # (e.g. "hey_jarvis"), fetching the latter on first use. The name form
        # lets the on-device acoustic path be exercised before a custom model
        # exists, so swapping in the real one is only a file change.
        if not os.path.exists(model_path):
            import openwakeword.utils

            name = os.path.splitext(os.path.basename(model_path))[0]
            print(f"[oww] '{model_path}' not found; treating '{name}' as a pretrained model",
                  flush=True)
            openwakeword.utils.download_models([name])
            model_path = name
        model_options = {
            "wakeword_models": [model_path],
            "inference_framework": "onnx",
        }
        melspec_path = os.environ.get("OWW_MELSPEC_PATH", "").strip()
        embedding_path = os.environ.get("OWW_EMBEDDING_PATH", "").strip()
        if melspec_path:
            model_options["melspec_model_path"] = melspec_path
        if embedding_path:
            model_options["embedding_model_path"] = embedding_path
        self._model = Model(**model_options)
        # The score dict is keyed by model name (derived from the file); take the
        # first unless an explicit key is given.
        self._key = key or next(iter(self._model.models.keys()))
        self._buf = np.zeros(0, dtype=np.int16)

    @property
    def key(self) -> str:
        return self._key

    def spot(self, frame: np.ndarray, sample_rate: int = 16000) -> str | None:
        """Feed one float32 [-1,1] mono frame; return the wake key when it fires.

        Buffers to openWakeWord's 80 ms unit; on a detection it resets internal
        state so a single utterance fires once.
        """
        pcm = (np.clip(np.asarray(frame, dtype=np.float32), -1.0, 1.0) * 32767).astype(np.int16)
        self._buf = np.concatenate([self._buf, pcm])
        fired = None
        while len(self._buf) >= _CHUNK:
            chunk = self._buf[:_CHUNK]
            self._buf = self._buf[_CHUNK:]
            scores = self._model.predict(chunk)
            if float(scores.get(self._key, 0.0)) >= self.threshold:
                fired = self._key
                self._model.reset()
        return fired
