"""Guards for loading serialized model artifacts (.engine / .onnx).

A serialized artifact loses its task metadata. Ultralytics then decodes the
apple-pear-mango segmentation head's 39 output channels as 4 box + 35 classes
instead of 4 box + 3 classes + 32 mask coefficients, which surfaces at predict
time as `KeyError: 23`. Every YOLO load in the media service must therefore
name its task explicitly, so an engine can replace a checkpoint by path alone.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).parents[1]
MEDIA = ROOT / "media"


def _yolo_calls(tree: ast.AST) -> list[ast.Call]:
    """Direct `YOLO(...)` calls and `partial(YOLO, ...)` deferred loads."""
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if isinstance(function, ast.Name) and function.id == "YOLO":
            calls.append(node)
        elif (
            isinstance(function, ast.Name)
            and function.id == "partial"
            and node.args
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "YOLO"
        ):
            calls.append(node)
    return calls


def test_every_media_yolo_load_names_its_task() -> None:
    found = 0
    for source_path in sorted(MEDIA.glob("*.py")):
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        for call in _yolo_calls(tree):
            found += 1
            keywords = {keyword.arg for keyword in call.keywords}
            assert "task" in keywords, (
                f"{source_path.name}:{call.lineno} loads YOLO without task=;"
                " a serialized engine would decode 39 channels as 35 classes"
            )
    assert found >= 4, "expected to find the media service's YOLO load sites"


def test_the_general_fruit_model_is_loaded_as_a_segmentation_task() -> None:
    tree = ast.parse((MEDIA / "perception_sidecar.py").read_text(encoding="utf-8"))
    segment_loads = [
        call
        for call in _yolo_calls(tree)
        for keyword in call.keywords
        if keyword.arg == "task"
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value == "segment"
    ]
    assert segment_loads, "PEAR_MODEL_PATH must be loaded with task='segment'"


def test_pear_model_path_stays_on_the_checkpoint_until_an_engine_is_proven() -> None:
    descriptor = json.loads((ROOT / "wendy.json").read_text(encoding="utf-8"))
    path = descriptor["services"]["media"]["env"]["PEAR_MODEL_PATH"]
    assert path.endswith(".pt"), (
        "switching PEAR_MODEL_PATH to an engine is deliberate; the engine must"
        " first pass lab/tensorrt-export against the reference frame"
    )


def test_the_tensorrt_export_app_builds_an_fp16_engine_on_device() -> None:
    export_dir = ROOT / "lab" / "tensorrt-export"
    source = (export_dir / "export.py").read_text(encoding="utf-8")
    assert "half=True" in source, "the engine must be exported FP16"
    assert 'format="engine"' in source
    assert 'task="segment"' in source, "the engine must be reloaded as a segmenter"
    descriptor = json.loads((export_dir / "wendy.json").read_text(encoding="utf-8"))
    entitlements = descriptor["services"]["export"]["entitlements"]
    assert {"type": "gpu"} in entitlements, "TensorRT export needs the GPU"


def test_the_media_image_ships_the_colour_backend_module() -> None:
    stagefile = yaml.safe_load((MEDIA / "build.stagefile.yaml").read_text())
    media = stagefile["stages"][-1]
    application_copy = [
        entry for entry in media["copy"] if entry["dest"] == "/app/media/"
    ]
    assert len(application_copy) == 1
    assert "fruit_color_backend.py" in application_copy[0]["paths"]
