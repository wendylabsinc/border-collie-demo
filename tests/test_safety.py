from datetime import timedelta

from robotkit.executor.safety import validate_effect
from tests.conftest import NOW


def test_safe_effect_is_allowed(effect_factory, snapshot_factory):
    verdict = validate_effect(effect_factory(based_on_revision=0), snapshot_factory(), NOW)
    assert verdict.allowed


def test_velocity_over_limit_is_rejected(effect_factory, snapshot_factory):
    verdict = validate_effect(effect_factory(linear=2.0), snapshot_factory(), NOW)
    assert not verdict.allowed
    assert "linear velocity" in verdict.reason


def test_missing_required_stream_is_rejected(effect_factory, snapshot_factory):
    verdict = validate_effect(
        effect_factory(required=["vision.obstacles"]), snapshot_factory(), NOW
    )
    assert not verdict.allowed
    assert "missing" in verdict.reason


def test_stale_required_stream_is_rejected(
    effect_factory, observation_factory, snapshot_factory
):
    stale = observation_factory(
        stream="vision.obstacles",
        observed_at=NOW - timedelta(seconds=10),
        ttl_seconds=1,
        revision=2,
    )
    verdict = validate_effect(
        effect_factory(required=["vision.obstacles"]), snapshot_factory([stale]), NOW
    )
    assert not verdict.allowed
    assert "stale" in verdict.reason


def test_excessive_state_drift_is_rejected(effect_factory, snapshot_factory):
    verdict = validate_effect(
        effect_factory(based_on_revision=1), snapshot_factory(state_revision=10), NOW
    )
    assert not verdict.allowed
    assert "drift" in verdict.reason


def test_effect_for_superseded_goal_is_rejected(effect_factory, snapshot_factory):
    effect = effect_factory()
    from uuid import uuid4

    verdict = validate_effect(
        effect, snapshot_factory(), NOW, active_goal_id=uuid4()
    )
    assert not verdict.allowed
    assert "active goal" in verdict.reason


def test_parameterless_lie_down_is_allowed(effect_factory, snapshot_factory):
    effect = effect_factory().model_copy(
        update={"effect_type": "unitree_lie_down", "parameters": {}}
    )
    assert validate_effect(effect, snapshot_factory(), NOW).allowed


def test_lie_down_rejects_injected_parameters(effect_factory, snapshot_factory):
    effect = effect_factory().model_copy(
        update={"effect_type": "unitree_lie_down", "parameters": {"command": "other"}}
    )
    assert not validate_effect(effect, snapshot_factory(), NOW).allowed


def test_configured_bark_effect_is_allowed(effect_factory, snapshot_factory):
    effect = effect_factory().model_copy(
        update={"effect_type": "unitree_bark", "parameters": {"sound": "bark"}}
    )
    verdict = validate_effect(effect, snapshot_factory(), NOW)
    assert verdict.allowed


def test_audio_effect_cannot_select_an_arbitrary_file(effect_factory, snapshot_factory):
    effect = effect_factory().model_copy(
        update={
            "effect_type": "unitree_bark",
            "parameters": {"sound": "bark", "path": "/tmp/untrusted.wav"},
        }
    )
    verdict = validate_effect(effect, snapshot_factory(), NOW)
    assert not verdict.allowed
