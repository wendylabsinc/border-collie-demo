from __future__ import annotations

import math
import struct

import pytest

from border_collie_demo.go2_remote import (
    Go2RemoteInput,
    RemoteInputConfig,
    decode_wireless_remote,
)


def remote_payload(
    *,
    lx: float = 0.0,
    ly: float = 0.0,
    rx: float = 0.0,
    ry: float = 0.0,
    buttons_1: int = 0,
    buttons_2: int = 0,
) -> bytes:
    payload = bytearray(40)
    payload[2] = buttons_1
    payload[3] = buttons_2
    struct.pack_into("<f", payload, 4, lx)
    struct.pack_into("<f", payload, 8, rx)
    struct.pack_into("<f", payload, 12, ry)
    struct.pack_into("<f", payload, 20, ly)
    return bytes(payload)


def test_decoder_ignores_a_neutral_controller_and_stick_deadband() -> None:
    assert decode_wireless_remote(remote_payload(), axis_deadzone=0.15) is None
    assert (
        decode_wireless_remote(
            remote_payload(lx=0.149, ry=-0.149),
            axis_deadzone=0.15,
        )
        is None
    )


def test_decoder_identifies_sticks_and_named_buttons() -> None:
    assert (
        decode_wireless_remote(remote_payload(ly=0.40), axis_deadzone=0.15)
        == "left_stick"
    )
    assert (
        decode_wireless_remote(remote_payload(rx=-0.40), axis_deadzone=0.15)
        == "right_stick"
    )
    assert (
        decode_wireless_remote(
            remote_payload(buttons_1=0b00000001, buttons_2=0b00000001),
            axis_deadzone=0.15,
        )
        == "buttons:R1+A"
    )
    assert (
        decode_wireless_remote(
            remote_payload(lx=0.5, buttons_2=0b00000010),
            axis_deadzone=0.15,
        )
        == "buttons:B+left_stick"
    )


@pytest.mark.parametrize(
    "payload",
    [
        b"too short",
        remote_payload(lx=math.nan),
        remote_payload(ry=math.inf),
        remote_payload(rx=2.0),
    ],
)
def test_decoder_rejects_malformed_or_impossible_samples(payload: bytes) -> None:
    with pytest.raises(ValueError):
        decode_wireless_remote(payload, axis_deadzone=0.15)


def test_watcher_requires_two_active_samples_and_latches_once() -> None:
    watcher = Go2RemoteInput(
        RemoteInputConfig(axis_deadzone=0.15, active_confirmations=2)
    )
    active = remote_payload(ly=0.60)

    assert watcher.observe(active, received_monotonic_s=10.0) is None
    remote_input = watcher.observe(active, received_monotonic_s=10.002)

    assert remote_input is not None
    assert remote_input.source == "unitree_remote"
    assert remote_input.control == "left_stick"
    assert remote_input.received_monotonic_s == 10.002
    assert watcher.observe(active, received_monotonic_s=10.004) is None
    assert watcher.status(now=10.005)["takeover_emitted"] is True


def test_neutral_sample_resets_confirmation_and_proves_monitor_freshness() -> None:
    watcher = Go2RemoteInput(
        RemoteInputConfig(
            axis_deadzone=0.15,
            active_confirmations=2,
            maximum_sample_age_s=0.50,
        )
    )
    active = remote_payload(lx=0.50)

    assert watcher.observe(active, received_monotonic_s=5.0) is None
    assert watcher.observe(remote_payload(), received_monotonic_s=5.01) is None
    assert watcher.observe(active, received_monotonic_s=5.02) is None
    assert watcher.status(now=5.03)["ready"] is True
    assert watcher.status(now=5.60)["ready"] is False

