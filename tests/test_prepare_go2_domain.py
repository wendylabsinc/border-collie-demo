from __future__ import annotations

from training.prepare_go2_domain import _split, _yolo_line


def test_temporal_split_holds_out_every_fifth_frame() -> None:
    assert [_split(value) for value in range(1, 7)] == [
        "train",
        "train",
        "train",
        "train",
        "val",
        "train",
    ]


def test_yolo_line_uses_fixed_pear_apple_banana_ids() -> None:
    assert _yolo_line(
        "apple", [10, 20, 30, 40], width=100, height=100
    ) == "1 0.20000000 0.30000000 0.20000000 0.20000000\n"
