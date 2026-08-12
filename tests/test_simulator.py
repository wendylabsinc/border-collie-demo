from datetime import timedelta

from robotkit.perception.simulator import interpret_world
from tests.conftest import NOW


def test_simulator_is_deterministic():
    assert interpret_world(NOW) == interpret_world(NOW)


def test_simulator_changes_with_time_without_retaining_state():
    before = interpret_world(NOW)
    after = interpret_world(NOW + timedelta(seconds=1))
    assert before != after
    assert {item.stream for item in before} == {
        "localization.pose",
        "vision.obstacles",
        "terrain.clearance",
    }
