"""Selectable fruit-inference runtimes behind the sidecar model boundary."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol


class MaxRunner(Protocol):
    def run(self, tensor: object) -> Mapping[str, object]: ...


class MaxCompiledModelRunner:
    """Load one precompiled MAX artifact and retain its accelerator session."""

    def __init__(
        self,
        *,
        model_path: str | Path,
        device_index: int = 0,
        output_names: tuple[str, ...] = ("boxes", "scores", "class_ids"),
        weights_path: str | Path | None = None,
        driver_module: Any | None = None,
        engine_module: Any | None = None,
    ) -> None:
        artifact = Path(model_path)
        if not artifact.is_file():
            raise FileNotFoundError(f"MAX artifact is missing: {artifact}")
        if driver_module is None or engine_module is None:
            from max import driver, engine

            driver_module = driver
            engine_module = engine
        accelerator = driver_module.Accelerator
        buffer_type = driver_module.Buffer
        session_type = engine_module.InferenceSession
        self._device = accelerator(device_index)
        self._buffer_type = buffer_type
        self._session = session_type(devices=[self._device])
        load_options: dict[str, object] = {}
        self._weights: dict[str, object] = {}
        if weights_path is not None:
            import numpy as np

            weight_artifact = Path(weights_path)
            if not weight_artifact.is_file():
                raise FileNotFoundError(
                    f"MAX weight registry is missing: {weight_artifact}"
                )
            with np.load(weight_artifact, allow_pickle=False) as archive:
                self._weights = {
                    name: buffer_type.from_numpy(
                        np.ascontiguousarray(archive[name])
                    ).to(self._device)
                    for name in archive.files
                }
            load_options["weights_registry"] = self._weights
        self._model = self._session.load(str(artifact), **load_options)
        self._output_names = output_names

    def run(self, tensor: object) -> Mapping[str, object]:
        device_input = self._buffer_type.from_numpy(tensor).to(self._device)
        outputs = self._model.execute(device_input)
        if not isinstance(outputs, (list, tuple)) or len(outputs) != len(
            self._output_names
        ):
            raise RuntimeError(
                "MAX detector output count does not match its configured contract"
            )
        values = [output.to_numpy() for output in outputs]
        return dict(zip(self._output_names, values, strict=True))


@dataclass(frozen=True)
class PreparedInput:
    tensor: object
    scale: float
    pad_x: float
    pad_y: float
    source_width: int
    source_height: int


class _ListTensor:
    def __init__(self, values: object) -> None:
        self._values = values

    def detach(self) -> _ListTensor:
        return self

    def cpu(self) -> _ListTensor:
        return self

    def tolist(self) -> object:
        return self._values


class _NormalizedBoxes:
    def __init__(self, scores: list[float], boxes: list[list[float]]) -> None:
        self.conf = _ListTensor(scores)
        self.xyxy = [_ListTensor(box) for box in boxes]

    def __len__(self) -> int:
        return len(self.xyxy)


def _tolist(value: object) -> object:
    conversion = getattr(value, "tolist", None)
    return conversion() if callable(conversion) else value


def _iou(first: list[float], second: list[float]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(
        0.0, second[3] - second[1]
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _default_preprocessor(source: Any, input_size: int) -> PreparedInput:
    import cv2
    import numpy as np

    source_height, source_width = (int(value) for value in source.shape[:2])
    scale = min(input_size / source_width, input_size / source_height)
    resized_width = max(1, round(source_width * scale))
    resized_height = max(1, round(source_height * scale))
    resized = cv2.resize(source, (resized_width, resized_height))
    pad_x = (input_size - resized_width) // 2
    pad_y = (input_size - resized_height) // 2
    canvas = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    canvas[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    tensor = np.ascontiguousarray(
        rgb.transpose(2, 0, 1)[None],
        dtype=np.float32,
    )
    tensor /= 255.0
    return PreparedInput(
        tensor=tensor,
        scale=scale,
        pad_x=float(pad_x),
        pad_y=float(pad_y),
        source_width=source_width,
        source_height=source_height,
    )


class MaxYoloModelAdapter:
    """Expose a compiled MAX detector through the existing YOLO model seam.

    The compiled model contract is three decoded outputs in letterboxed input
    pixels: ``boxes`` (N x 4 xyxy), ``scores`` (N), and ``class_ids`` (N).
    Target filtering, NMS, and source-coordinate restoration stay here.
    """

    def __init__(
        self,
        *,
        runner: MaxRunner,
        class_names: Mapping[int, str],
        input_size: int = 640,
        nms_iou: float = 0.45,
        output_contract: str = "decoded",
        preprocessor: Callable[[Any, int], PreparedInput] = _default_preprocessor,
    ) -> None:
        if input_size < 1:
            raise ValueError("MAX input size must be positive")
        if not 0.0 <= nms_iou <= 1.0:
            raise ValueError("MAX NMS IoU must be in [0, 1]")
        if output_contract not in {"decoded", "yoloe_segment_raw"}:
            raise ValueError("unsupported MAX detector output contract")
        self._runner = runner
        self.names = {int(class_id): str(name) for class_id, name in class_names.items()}
        self._input_size = input_size
        self._nms_iou = nms_iou
        self._output_contract = output_contract
        self._preprocessor = preprocessor

    def predict(
        self,
        *,
        source: Any,
        conf: float,
        classes: list[int],
        device: int | str,
        verbose: bool,
    ) -> list[object]:
        del device, verbose
        prepared = self._preprocessor(source, self._input_size)
        outputs = self._runner.run(prepared.tensor)
        accepted_classes = {int(class_id) for class_id in classes}
        if self._output_contract == "decoded":
            proposals = self._decoded_proposals(
                outputs,
                accepted_classes=accepted_classes,
                confidence_floor=conf,
            )
        else:
            proposals = self._yoloe_segment_proposals(
                outputs,
                accepted_classes=accepted_classes,
                confidence_floor=conf,
            )
        kept: list[tuple[float, list[float]]] = []
        for proposal in proposals:
            if all(_iou(proposal[1], previous[1]) <= self._nms_iou for previous in kept):
                kept.append(proposal)

        source_boxes: list[list[float]] = []
        source_scores: list[float] = []
        for score, box in kept:
            x1 = max(0.0, min((box[0] - prepared.pad_x) / prepared.scale, prepared.source_width))
            y1 = max(0.0, min((box[1] - prepared.pad_y) / prepared.scale, prepared.source_height))
            x2 = max(0.0, min((box[2] - prepared.pad_x) / prepared.scale, prepared.source_width))
            y2 = max(0.0, min((box[3] - prepared.pad_y) / prepared.scale, prepared.source_height))
            if x1 < x2 and y1 < y2:
                source_scores.append(score)
                source_boxes.append([x1, y1, x2, y2])
        return [SimpleNamespace(boxes=_NormalizedBoxes(source_scores, source_boxes))]

    @staticmethod
    def _decoded_proposals(
        outputs: Mapping[str, object],
        *,
        accepted_classes: set[int],
        confidence_floor: float,
    ) -> list[tuple[float, list[float]]]:
        try:
            boxes = list(_tolist(outputs["boxes"]))
            scores = list(_tolist(outputs["scores"]))
            class_ids = list(_tolist(outputs["class_ids"]))
        except (KeyError, TypeError) as exc:
            raise RuntimeError(
                "MAX detector must return boxes, scores, and class_ids"
            ) from exc
        if not (len(boxes) == len(scores) == len(class_ids)):
            raise RuntimeError("MAX detector output lengths do not match")
        return sorted(
            (
                (float(score), [float(value) for value in box])
                for box, score, class_id in zip(boxes, scores, class_ids, strict=True)
                if int(class_id) in accepted_classes
                and float(score) >= confidence_floor
            ),
            reverse=True,
        )

    @staticmethod
    def _yoloe_segment_proposals(
        outputs: Mapping[str, object],
        *,
        accepted_classes: set[int],
        confidence_floor: float,
    ) -> list[tuple[float, list[float]]]:
        try:
            batch = list(_tolist(outputs["predictions"]))
            channels = list(batch[0])
        except (KeyError, TypeError, IndexError) as exc:
            raise RuntimeError(
                "MAX YOLOE detector must return a batched predictions tensor"
            ) from exc
        required_channels = 4 + max(accepted_classes, default=-1) + 1
        if len(channels) < required_channels:
            raise RuntimeError("MAX YOLOE prediction tensor has too few classes")
        anchor_count = len(channels[0])
        if any(len(channel) != anchor_count for channel in channels):
            raise RuntimeError("MAX YOLOE prediction channel lengths do not match")
        proposals: list[tuple[float, list[float]]] = []
        for class_id in accepted_classes:
            class_scores = channels[4 + class_id]
            for index, raw_score in enumerate(class_scores):
                score = float(raw_score)
                if score < confidence_floor:
                    continue
                center_x = float(channels[0][index])
                center_y = float(channels[1][index])
                width = float(channels[2][index])
                height = float(channels[3][index])
                proposals.append(
                    (
                        score,
                        [
                            center_x - width / 2.0,
                            center_y - height / 2.0,
                            center_x + width / 2.0,
                            center_y + height / 2.0,
                        ],
                    )
                )
        return sorted(proposals, reverse=True)


@dataclass(frozen=True)
class InferenceRuntimeConfig:
    requested_backend: str = "ultralytics"
    allow_ultralytics_fallback: bool = False
    max_model_path: str = "/media/apple-pear-mango.cuda-sm87.mef"
    max_weights_path: str = "/media/apple-pear-mango.cuda-sm87.weights.npz"
    max_fruit_class_ids: dict[str, int] = field(default_factory=dict)
    max_input_size: int = 640
    max_device_index: int = 0
    max_output_contract: str = "yoloe_segment_raw"

    @classmethod
    def from_env(cls) -> InferenceRuntimeConfig:
        return cls.from_mapping(os.environ)

    @classmethod
    def from_mapping(cls, values: Mapping[str, str]) -> InferenceRuntimeConfig:
        backend = values.get(
            "FRUIT_INFERENCE_BACKEND", "ultralytics"
        ).strip().casefold()
        # The original MAX branch called every control path "tensorrt". The
        # presented checkpoint now loads apple-pear-mango.pt through
        # Ultralytics, so retain the old spelling only as a compatibility alias.
        if backend == "tensorrt":
            backend = "ultralytics"
        if backend not in {"ultralytics", "max"}:
            raise ValueError("FRUIT_INFERENCE_BACKEND must be ultralytics or max")
        allow_fallback = values.get(
            "MAX_ALLOW_ULTRALYTICS_FALLBACK",
            values.get("MAX_ALLOW_TENSORRT_FALLBACK", "0"),
        ).strip().casefold() in {"1", "true", "yes", "on"}
        raw_class_ids = values.get("MAX_FRUIT_CLASS_IDS_JSON", "{}").strip()
        try:
            decoded_class_ids = json.loads(raw_class_ids)
        except json.JSONDecodeError as exc:
            raise ValueError("MAX_FRUIT_CLASS_IDS_JSON must be valid JSON") from exc
        if not isinstance(decoded_class_ids, dict) or not all(
            isinstance(name, str)
            and isinstance(class_id, int)
            and not isinstance(class_id, bool)
            and class_id >= 0
            for name, class_id in decoded_class_ids.items()
        ):
            raise ValueError(
                "MAX_FRUIT_CLASS_IDS_JSON must map fruit names to non-negative IDs"
            )
        max_input_size = int(values.get("MAX_INPUT_SIZE", "640"))
        max_device_index = int(values.get("MAX_DEVICE_INDEX", "0"))
        if max_input_size < 1:
            raise ValueError("MAX_INPUT_SIZE must be positive")
        if max_device_index < 0:
            raise ValueError("MAX_DEVICE_INDEX must be non-negative")
        output_contract = values.get(
            "MAX_OUTPUT_CONTRACT", "yoloe_segment_raw"
        ).strip().casefold()
        if output_contract not in {"decoded", "yoloe_segment_raw"}:
            raise ValueError(
                "MAX_OUTPUT_CONTRACT must be decoded or yoloe_segment_raw"
            )
        return cls(
            requested_backend=backend,
            allow_ultralytics_fallback=allow_fallback,
            max_model_path=values.get(
                "MAX_MODEL_PATH", "/media/apple-pear-mango.cuda-sm87.mef"
            ).strip(),
            max_weights_path=values.get(
                "MAX_WEIGHTS_PATH",
                "/media/apple-pear-mango.cuda-sm87.weights.npz",
            ).strip(),
            max_fruit_class_ids={
                name.casefold().strip(): class_id
                for name, class_id in decoded_class_ids.items()
            },
            max_input_size=max_input_size,
            max_device_index=max_device_index,
            max_output_contract=output_contract,
        )


@dataclass(frozen=True)
class LoadedInferenceRuntime:
    model: object
    requested_backend: str
    active_backend: str
    fallback_reason: str | None = None

    def status(self) -> dict[str, object]:
        return {
            "requested_backend": self.requested_backend,
            "active_backend": self.active_backend,
            "fallback_used": self.fallback_reason is not None,
            "fallback_reason": self.fallback_reason,
            "candidate_validated": self.active_backend == "ultralytics",
        }


def create_max_model(
    config: InferenceRuntimeConfig,
    *,
    runner_factory: Callable[..., MaxRunner] = MaxCompiledModelRunner,
) -> MaxYoloModelAdapter:
    if not config.max_fruit_class_ids:
        raise ValueError(
            "MAX_FRUIT_CLASS_IDS_JSON is required when MAX is selected"
        )
    runner = runner_factory(
        model_path=config.max_model_path,
        device_index=config.max_device_index,
        output_names=(
            ("predictions", "prototypes")
            if config.max_output_contract == "yoloe_segment_raw"
            else ("boxes", "scores", "class_ids")
        ),
        weights_path=config.max_weights_path,
    )
    return MaxYoloModelAdapter(
        runner=runner,
        class_names={
            class_id: fruit
            for fruit, class_id in config.max_fruit_class_ids.items()
        },
        input_size=config.max_input_size,
        output_contract=config.max_output_contract,
    )


def load_general_model(
    config: InferenceRuntimeConfig,
    *,
    ultralytics_factory: Callable[[], object],
    max_factory: Callable[[InferenceRuntimeConfig], object],
) -> LoadedInferenceRuntime:
    if config.requested_backend == "ultralytics":
        return LoadedInferenceRuntime(
            model=ultralytics_factory(),
            requested_backend="ultralytics",
            active_backend="ultralytics",
        )
    try:
        model = max_factory(config)
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        if config.allow_ultralytics_fallback:
            return LoadedInferenceRuntime(
                model=ultralytics_factory(),
                requested_backend="max",
                active_backend="ultralytics",
                fallback_reason=reason,
            )
        raise RuntimeError(
            f"MAX inference startup failed: {reason}"
        ) from exc
    return LoadedInferenceRuntime(
        model=model,
        requested_backend="max",
        active_backend="max",
    )
