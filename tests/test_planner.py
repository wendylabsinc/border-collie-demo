from datetime import timedelta

from robotkit.contracts import Effect, EffectRecord, EffectStatus, Goal, GoalRecord
from robotkit.planner.logic import plan
from tests.conftest import NOW


def _goal(decision, *, key="mission-goal") -> GoalRecord:
    goal = Goal(
        idempotency_key=key,
        planner_id="mission-planner",
        instance_id="green",
        based_on_revision=10,
        goal_type=decision.goal_type,
        priority=decision.priority,
        created_at=NOW,
        valid_until=NOW + timedelta(minutes=1),
        parameters=decision.parameters,
        rationale=decision.rationale,
    )
    return GoalRecord(**goal.model_dump(), revision=11, status="active")


def _applied(goal, *, stage_complete=False) -> EffectRecord:
    effect = Effect(
        idempotency_key=f"effect:{goal.goal_id}",
        controller_id="mission-controller",
        instance_id="green",
        goal_id=goal.goal_id,
        based_on_revision=10,
        effect_type="cmd_vel",
        created_at=NOW,
        valid_until=NOW + timedelta(minutes=1),
        parameters={"stage_complete": stage_complete},
    )
    return EffectRecord(
        **effect.model_dump(), revision=12, status=EffectStatus.APPLIED
    )


def _voice(observation_factory, *, key="voice-1", revision=1):
    return observation_factory(
        stream="voice.intent",
        key=key,
        revision=revision,
        payload={"intent": "find", "slots": {"target": "apple"}},
        ttl_seconds=30,
    )


def _apple(observation_factory, *, revision=2):
    return observation_factory(
        stream="vision.fruits",
        key=f"fruit-{revision}",
        revision=revision,
        payload={
            "detections": [
                {
                    "class_name": "apple",
                    "confidence": 0.91,
                    "bbox_xyxy_normalized": [0.4, 0.2, 0.6, 0.8],
                }
            ]
        },
    )


def test_no_trigger_is_idle_at_home(snapshot_factory):
    decision = plan(snapshot_factory(), NOW)

    assert decision.goal_type == "idle_at_home"
    assert decision.parameters["mission_type"] == "idle"


def test_find_apple_voice_starts_correlated_mission(
    observation_factory, snapshot_factory
):
    voice = _voice(observation_factory)
    decision = plan(snapshot_factory([voice]), NOW)

    assert decision.goal_type == "search_apple"
    assert decision.parameters == {
        "mission_schema_version": "1",
        "mission_type": "apple",
        "mission_stage": "search_apple",
        "stage_index": 0,
        "trigger_event_id": str(voice.event_id),
        "trigger_stream": "voice.intent",
    }


def test_find_apple_website_starts_correlated_mission(
    observation_factory, snapshot_factory
):
    website = observation_factory(
        stream="website.intent",
        payload={
            "intent": "find",
            "slots": {"target": "apple"},
            "source_text": "find apple",
        },
    )

    decision = plan(snapshot_factory([website]), NOW)

    assert decision.goal_type == "search_apple"
    assert decision.parameters["trigger_event_id"] == str(website.event_id)
    assert decision.parameters["trigger_stream"] == "website.intent"


def test_unsupported_pear_is_an_explicit_idle_rejection(
    observation_factory, snapshot_factory
):
    website = observation_factory(
        stream="website.intent",
        payload={"intent": "find", "slots": {"target": "pear"}},
        ttl_seconds=30,
    )

    decision = plan(snapshot_factory([website]), NOW)

    assert decision.goal_type == "idle_at_home"
    assert decision.parameters["rejected_target"] == "pear"
    assert decision.parameters["trigger_event_id"] == str(website.event_id)
    assert "unsupported target 'pear'" in decision.rationale


def test_unsupported_command_stops_an_active_apple_search(
    observation_factory, snapshot_factory
):
    apple_command = _voice(observation_factory)
    active = _goal(plan(snapshot_factory([apple_command]), NOW))
    pear_command = observation_factory(
        stream="website.intent",
        key="pear-command",
        revision=2,
        observed_at=NOW + timedelta(milliseconds=1),
        payload={"intent": "find", "slots": {"target": "pear"}},
        ttl_seconds=30,
    )

    decision = plan(
        snapshot_factory([apple_command, pear_command]),
        NOW + timedelta(seconds=1),
        current_goal=active,
    )

    assert decision.goal_type == "idle_at_home"
    assert decision.parameters["rejected_target"] == "pear"


def test_newest_command_channel_wins(observation_factory, snapshot_factory):
    voice = _voice(observation_factory, revision=1)
    website = observation_factory(
        stream="website.intent",
        key="website-newer",
        revision=2,
        observed_at=NOW + timedelta(milliseconds=1),
        payload={"intent": "unknown", "slots": {}, "source_text": "never mind"},
    )

    assert plan(snapshot_factory([voice, website]), NOW + timedelta(seconds=1)).goal_type == (
        "idle_at_home"
    )


def test_move_to_apple_transcript_is_accepted(observation_factory, snapshot_factory):
    voice = observation_factory(
        stream="voice.intent",
        payload={
            "intent": "move",
            "slots": {"target": "apple"},
            "source_transcript": "Go to the apple",
        },
    )
    assert plan(snapshot_factory([voice]), NOW).goal_type == "search_apple"


def test_search_advances_as_soon_as_fresh_apple_is_detected(
    observation_factory, snapshot_factory
):
    voice = _voice(observation_factory)
    started = _goal(plan(snapshot_factory([voice]), NOW))
    apple = _apple(observation_factory)

    decision = plan(
        snapshot_factory([voice, apple]), NOW, current_goal=started
    )

    assert decision.goal_type == "approach_apple"
    assert decision.parameters["stage_index"] == 1
    assert decision.parameters["trigger_event_id"] == str(voice.event_id)


def test_approach_advances_only_after_applied_completed_effect(
    observation_factory, snapshot_factory
):
    voice = _voice(observation_factory)
    apple = _apple(observation_factory)
    search = _goal(plan(snapshot_factory([voice]), NOW))
    approach = _goal(
        plan(snapshot_factory([voice, apple]), NOW, current_goal=search),
        key="approach",
    )

    not_complete = plan(
        snapshot_factory([voice, apple]),
        NOW,
        current_goal=approach,
        latest_effect=_applied(approach, stage_complete=False),
    )
    complete = plan(
        snapshot_factory([voice, apple]),
        NOW,
        current_goal=approach,
        latest_effect=_applied(approach, stage_complete=True),
    )

    assert not_complete.goal_type == "approach_apple"
    assert complete.goal_type == "bark"


def test_bark_then_home_then_lie_down_are_applied_transitions(
    observation_factory, snapshot_factory
):
    voice = _voice(observation_factory)
    apple = _apple(observation_factory)
    snapshot = snapshot_factory([voice, apple])
    bark_decision = plan(snapshot, NOW)
    bark_decision = bark_decision.__class__(
        "bark",
        bark_decision.priority,
        {**bark_decision.parameters, "mission_stage": "bark", "stage_index": 2},
        bark_decision.rationale,
    )
    bark_goal = _goal(bark_decision, key="bark")

    home_decision = plan(
        snapshot,
        NOW,
        current_goal=bark_goal,
        latest_effect=_applied(bark_goal),
    )
    home_goal = _goal(home_decision, key="home")
    lie_decision = plan(
        snapshot,
        NOW,
        current_goal=home_goal,
        latest_effect=_applied(home_goal, stage_complete=True),
    )

    assert home_decision.goal_type == "go_home"
    assert lie_decision.goal_type == "lie_down"


def test_same_voice_event_cannot_retrigger_terminal_mission(
    observation_factory, snapshot_factory
):
    voice = _voice(observation_factory)
    lie_goal = _goal(plan(snapshot_factory([voice]), NOW), key="terminal")
    lie_goal = lie_goal.model_copy(
        update={
            "goal_type": "lie_down",
            "parameters": {
                **lie_goal.parameters,
                "mission_stage": "lie_down",
                "stage_index": 4,
            },
        }
    )

    decision = plan(snapshot_factory([voice]), NOW, current_goal=lie_goal)

    assert decision.goal_type == "lie_down"
    assert decision.parameters["trigger_event_id"] == str(voice.event_id)


def test_distinct_voice_event_starts_a_new_mission(
    observation_factory, snapshot_factory
):
    old_voice = _voice(observation_factory, key="old")
    terminal = _goal(plan(snapshot_factory([old_voice]), NOW), key="terminal")
    terminal = terminal.model_copy(
        update={
            "goal_type": "lie_down",
            "parameters": {
                **terminal.parameters,
                "mission_stage": "lie_down",
                "stage_index": 4,
            },
        }
    )
    new_voice = _voice(observation_factory, key="new", revision=2)

    decision = plan(snapshot_factory([new_voice]), NOW, current_goal=terminal)

    assert decision.goal_type == "search_apple"
    assert decision.parameters["trigger_event_id"] == str(new_voice.event_id)


def test_stale_voice_does_not_start_a_mission(observation_factory, snapshot_factory):
    voice = observation_factory(
        stream="voice.intent",
        payload={"intent": "find", "slots": {"target": "apple"}},
        observed_at=NOW - timedelta(seconds=31),
        ttl_seconds=30,
    )
    assert plan(snapshot_factory([voice]), NOW).goal_type == "idle_at_home"


def test_temperature_high_preempts_apple_and_correlates_health_event(
    observation_factory, snapshot_factory
):
    voice = _voice(observation_factory)
    apple_goal = _goal(plan(snapshot_factory([voice]), NOW))
    temperature = observation_factory(
        stream="health.temperature",
        payload={"band": "high", "temperature_c": 65.0},
        revision=3,
    )

    decision = plan(
        snapshot_factory([voice, temperature]), NOW, current_goal=apple_goal
    )

    assert decision.goal_type == "go_home"
    assert decision.priority == 100
    assert decision.parameters["mission_type"] == "health_return"
    assert decision.parameters["trigger_event_id"] == str(temperature.event_id)


def test_critical_low_battery_preempts_but_plain_low_does_not(
    observation_factory, snapshot_factory
):
    low = observation_factory(
        stream="health.battery", payload={"band": "low"}, key="low"
    )
    critical = observation_factory(
        stream="health.battery",
        payload={"band": "critical_low"},
        key="critical",
        revision=2,
    )

    assert plan(snapshot_factory([low]), NOW).goal_type == "idle_at_home"
    assert plan(snapshot_factory([critical]), NOW).goal_type == "go_home"


def test_new_health_samples_do_not_reset_an_active_health_sequence(
    observation_factory, snapshot_factory
):
    first = observation_factory(
        stream="health.temperature", payload={"band": "critical_high"}, key="first"
    )
    home_goal = _goal(plan(snapshot_factory([first]), NOW), key="health-home")
    newer = observation_factory(
        stream="health.temperature",
        payload={"band": "critical_high"},
        key="newer",
        revision=2,
    )

    decision = plan(
        snapshot_factory([newer]),
        NOW,
        current_goal=home_goal,
        latest_effect=_applied(home_goal, stage_complete=True),
    )

    assert decision.goal_type == "lie_down"
    assert decision.parameters["trigger_event_id"] == str(first.event_id)
