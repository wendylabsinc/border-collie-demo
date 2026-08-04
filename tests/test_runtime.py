import time

from fastapi.testclient import TestClient

from border_collie_demo.main import build_app_from_env
from border_collie_demo.orchestrator import SimulatedStageExecutor
from border_collie_demo.simulation import SimulatedHardware, simulated_camera_perception


def test_explicit_simulation_runtime_completes_activate_demo(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("BORDER_COLLIE_RUNTIME_MODE", "simulation")
    monkeypatch.setenv("BORDER_COLLIE_RUNS_DIR", str(tmp_path))

    with TestClient(build_app_from_env()) as client:
        status = client.get("/api/status").json()
        assert status["runtime_mode"] == "simulation"
        assert status["activation"]["ready"] is True

        started = client.post("/api/run", json={"target_fruit": "pear"}).json()["run"]
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            run = client.get(f"/api/results/{started['run_id']}").json()["run"]
            if run["outcome"] is not None:
                break
            time.sleep(0.01)

        assert run["outcome"] == "COMPLETED"
        assert run["reason"] == "SUCCESS"
        assert all(
            evidence["motion_commands_sent"] is False
            for evidence in run["stage_results"].values()
        )


def test_production_runtime_wires_the_real_stage_executor(
    tmp_path, monkeypatch
) -> None:
    import border_collie_demo.main as main_module

    created = []

    class Perception:
        def __init__(self, _config) -> None:
            pass

        status = staticmethod(simulated_camera_perception)

        @staticmethod
        def select_target(target_fruit: str):
            return {"target_fruit": target_fruit}

        @staticmethod
        def camera_frame():
            return b"\xff\xd8preview\xff\xd9"

    def stages(hardware, perception, bark):
        created.append((hardware, perception, bark))
        return SimulatedStageExecutor()

    class Bark:
        def __init__(self, _config) -> None:
            pass

        @staticmethod
        def status():
            return {"ready": True, "detail": "bark ready"}

    class Evidence:
        @classmethod
        def from_env(cls):
            return cls()

        @staticmethod
        def capture():
            return []

    hardware = SimulatedHardware()
    monkeypatch.setenv("BORDER_COLLIE_RUNTIME_MODE", "production")
    monkeypatch.setenv("BORDER_COLLIE_RUNS_DIR", str(tmp_path))
    monkeypatch.setattr(main_module, "HardwareManager", lambda _config: hardware)
    monkeypatch.setattr(main_module, "PerceptionStatusClient", Perception)
    monkeypatch.setattr(main_module, "BarkClient", Bark)
    monkeypatch.setattr(main_module, "TerminalEvidenceClient", Evidence)
    monkeypatch.setattr(main_module, "ProductionStageExecutor", stages)

    with TestClient(build_app_from_env()) as client:
        started = client.post("/api/run", json={"target_fruit": "pear"}).json()["run"]
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            run = client.get(f"/api/results/{started['run_id']}").json()["run"]
            if run["outcome"] is not None:
                break
            time.sleep(0.01)

    assert run["outcome"] == "COMPLETED"
    assert created and created[0][0] is hardware
