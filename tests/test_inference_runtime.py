from types import SimpleNamespace

import pytest

from media.inference_runtime import (
    InferenceRuntimeConfig,
    MaxCompiledModelRunner,
    MaxYoloModelAdapter,
    PreparedInput,
    load_general_model,
)
from media.model_router import FruitModelRouter


def test_tensorrt_remains_the_default_validated_runtime() -> None:
    sentinel = object()

    loaded = load_general_model(
        InferenceRuntimeConfig.from_mapping({}),
        tensorrt_factory=lambda: sentinel,
        max_factory=lambda _config: (_ for _ in ()).throw(
            AssertionError("MAX must not load by default")
        ),
    )

    assert loaded.model is sentinel
    assert loaded.status() == {
        "requested_backend": "tensorrt",
        "active_backend": "tensorrt",
        "fallback_used": False,
        "fallback_reason": None,
        "candidate_validated": True,
    }


def test_max_startup_failure_is_fail_closed_without_explicit_fallback() -> None:
    config = InferenceRuntimeConfig.from_mapping(
        {"FRUIT_INFERENCE_BACKEND": "max"}
    )

    with pytest.raises(RuntimeError, match="MAX inference startup failed"):
        load_general_model(
            config,
            tensorrt_factory=lambda: object(),
            max_factory=lambda _config: (_ for _ in ()).throw(
                FileNotFoundError("candidate.mef")
            ),
        )


def test_explicit_max_fallback_reports_that_tensorrt_is_active() -> None:
    sentinel = object()
    config = InferenceRuntimeConfig.from_mapping(
        {
            "FRUIT_INFERENCE_BACKEND": "max",
            "MAX_ALLOW_TENSORRT_FALLBACK": "1",
        }
    )

    loaded = load_general_model(
        config,
        tensorrt_factory=lambda: sentinel,
        max_factory=lambda _config: (_ for _ in ()).throw(
            RuntimeError("accelerator unavailable")
        ),
    )

    assert loaded.model is sentinel
    assert loaded.status() == {
        "requested_backend": "max",
        "active_backend": "tensorrt",
        "fallback_used": True,
        "fallback_reason": "RuntimeError: accelerator unavailable",
        "candidate_validated": True,
    }


def test_max_adapter_preserves_the_existing_candidate_contract() -> None:
    class Runner:
        def __init__(self) -> None:
            self.inputs = []

        def run(self, tensor):
            self.inputs.append(tensor)
            return {
                "boxes": [
                    [50, 240, 150, 340],
                    [52, 242, 152, 342],
                    [400, 200, 500, 300],
                ],
                "scores": [0.84, 0.70, 0.99],
                "class_ids": [2, 2, 1],
            }

    runner = Runner()
    prepared = PreparedInput(
        tensor=SimpleNamespace(shape=(1, 3, 640, 640), dtype="float32"),
        scale=0.5,
        pad_x=0.0,
        pad_y=140.0,
        source_width=1280,
        source_height=720,
    )
    adapter = MaxYoloModelAdapter(
        runner=runner,
        class_names={1: "apple", 2: "pear"},
        input_size=640,
        preprocessor=lambda _source, _input_size: prepared,
    )
    router = FruitModelRouter(
        general_model=adapter,
        general_class_ids={"apple": 1, "pear": 2},
    )

    result = router.predict(
        source=object(),
        target_fruit="pear",
        device=0,
    )

    assert runner.inputs[0].shape == (1, 3, 640, 640)
    assert runner.inputs[0].dtype == "float32"
    assert result.candidate is not None
    assert result.candidate.confidence == pytest.approx(0.84)
    assert result.candidate.bbox_xyxy == (100, 200, 300, 400)


def test_max_adapter_decodes_the_exported_yoloe_segmentation_head() -> None:
    class Runner:
        def run(self, _tensor):
            return {
                "predictions": [
                    [
                        [100.0, 102.0],
                        [300.0, 302.0],
                        [40.0, 40.0],
                        [60.0, 60.0],
                        [0.02, 0.03],
                        [0.10, 0.08],
                        [0.82, 0.70],
                    ]
                ],
                "prototypes": [],
            }

    prepared = PreparedInput(
        tensor=object(),
        scale=1.0,
        pad_x=0.0,
        pad_y=0.0,
        source_width=640,
        source_height=640,
    )
    adapter = MaxYoloModelAdapter(
        runner=Runner(),
        class_names={0: "apple", 1: "banana", 2: "pear"},
        input_size=640,
        output_contract="yoloe_segment_raw",
        preprocessor=lambda _source, _input_size: prepared,
    )
    router = FruitModelRouter(
        general_model=adapter,
        general_class_ids={"apple": 0, "banana": 1, "pear": 2},
    )

    result = router.predict(
        source=object(),
        target_fruit="pear",
        device=0,
    )

    assert result.candidate is not None
    assert result.candidate.confidence == pytest.approx(0.82)
    assert result.candidate.bbox_xyxy == (80, 270, 120, 330)


def test_max_runner_loads_a_compiled_mef_once_and_executes_on_accelerator(
    tmp_path,
) -> None:
    calls = []
    artifact = tmp_path / "fruit.cuda-sm87.mef"
    artifact.write_bytes(b"compiled-test-artifact")

    class Buffer:
        @classmethod
        def from_numpy(cls, value):
            calls.append(("from_numpy", value))
            return cls()

        def to(self, device):
            calls.append(("buffer_to", device))
            return self

    class Output:
        def __init__(self, value):
            self.value = value

        def to_numpy(self):
            return self.value

    class Model:
        def execute(self, value):
            calls.append(("execute", value))
            return [Output([[1, 2, 3, 4]]), Output([0.9]), Output([2])]

    device = object()

    class Driver:
        @staticmethod
        def Accelerator(index):
            calls.append(("accelerator", index))
            return device

    Driver.Buffer = Buffer

    class Session:
        def __init__(self, *, devices):
            calls.append(("session", devices))

        def load(self, path):
            calls.append(("load", path))
            return Model()

    class Engine:
        InferenceSession = Session

    runner = MaxCompiledModelRunner(
        model_path=artifact,
        driver_module=Driver,
        engine_module=Engine,
    )

    outputs = runner.run("host-tensor")

    assert outputs == {
        "boxes": [[1, 2, 3, 4]],
        "scores": [0.9],
        "class_ids": [2],
    }
    assert calls[:3] == [
        ("accelerator", 0),
        ("session", [device]),
        ("load", str(artifact)),
    ]
    assert [name for name, _value in calls].count("load") == 1


def test_max_runtime_configuration_names_the_candidate_artifact_and_classes() -> None:
    config = InferenceRuntimeConfig.from_mapping(
        {
            "FRUIT_INFERENCE_BACKEND": "max",
            "MAX_MODEL_PATH": "/media/fruit.cuda-sm87.mef",
            "MAX_WEIGHTS_PATH": "/media/fruit.cuda-sm87.weights.npz",
            "MAX_FRUIT_CLASS_IDS_JSON": '{"apple": 1, "banana": 2, "pear": 3}',
            "MAX_INPUT_SIZE": "640",
            "MAX_DEVICE_INDEX": "0",
        }
    )

    assert config.max_model_path == "/media/fruit.cuda-sm87.mef"
    assert config.max_weights_path == "/media/fruit.cuda-sm87.weights.npz"
    assert config.max_fruit_class_ids == {"apple": 1, "banana": 2, "pear": 3}
    assert config.max_input_size == 640
    assert config.max_device_index == 0
    assert config.max_output_contract == "yoloe_segment_raw"
