from time import monotonic

import pytest

from border_collie_demo.mission import MissionMachine, RestartRequired
from border_collie_demo.models import MissionPhase, RemoteInput


def test_happy_path_has_one_explicit_order() -> None:
    mission = MissionMachine()

    observed = []
    while mission.phase != MissionPhase.COMPLETE:
        observed.append(mission.advance("test progression"))

    assert observed == [
        MissionPhase.PREFLIGHT,
        MissionPhase.CAPTURE_HOME,
        MissionPhase.WAIT_FOR_COMMAND,
        MissionPhase.TURN_TO_FRUIT,
        MissionPhase.FIND_FRUIT,
        MissionPhase.APPROACH_FRUIT,
        MissionPhase.ARRIVED,
        MissionPhase.SIT_AND_BARK,
        MissionPhase.STAND,
        MissionPhase.TURN_TOWARD_HOME,
        MissionPhase.RETURN_HOME,
        MissionPhase.RESTORE_HEADING,
        MissionPhase.COMPLETE,
    ]


def test_remote_takeover_is_latched_until_process_restart() -> None:
    mission = MissionMachine()
    mission.advance("begin preflight")

    phase = mission.remote_takeover(
        RemoteInput("unitree_remote", "left_stick", monotonic())
    )

    assert phase == MissionPhase.REMOTE_TAKEOVER
    assert mission.status()["restart_required"] is True
    with pytest.raises(RestartRequired, match="restart"):
        mission.advance("must not resume")
    with pytest.raises(RestartRequired, match="restart"):
        mission.stop()

    restarted_process = MissionMachine()
    assert restarted_process.phase == MissionPhase.IDLE
    assert restarted_process.takeover_latched is False


def test_repeated_remote_input_cannot_clear_or_replace_the_latch() -> None:
    mission = MissionMachine()
    first = RemoteInput("unitree_remote", "button_a", monotonic())
    second = RemoteInput("unitree_remote", "button_b", monotonic())

    mission.remote_takeover(first)
    mission.remote_takeover(second)

    assert mission.phase == MissionPhase.REMOTE_TAKEOVER
    assert mission.reason == "unitree_remote: button_a"
    assert len(mission.history) == 2


def test_releasing_the_remote_hands_control_back_without_resuming_anything() -> None:
    mission = MissionMachine()
    mission.advance("begin preflight")
    mission.remote_takeover(RemoteInput("go2_controller", "left_stick_y", monotonic()))

    phase = mission.release_remote_takeover("physical remote released")

    # The hazard the latch guards -- a human driving the robot by hand -- ends
    # when the remote is released, so the latch ends with it.
    assert phase == MissionPhase.STOPPED
    assert mission.takeover_latched is False
    assert mission.status()["restart_required"] is False
    # Nothing resumed: the takeover stays in the history and the mission is
    # stopped, waiting for a fresh deliberate human activation.
    assert [event.phase for event in mission.history][-2:] == [
        MissionPhase.REMOTE_TAKEOVER,
        MissionPhase.STOPPED,
    ]
    assert mission.begin_run("operator started a new Demo Run") == (
        MissionPhase.PREFLIGHT
    )


def test_releasing_an_unlatched_mission_changes_nothing() -> None:
    mission = MissionMachine()
    mission.advance("begin preflight")

    assert mission.release_remote_takeover() == MissionPhase.PREFLIGHT
    assert len(mission.history) == 2
