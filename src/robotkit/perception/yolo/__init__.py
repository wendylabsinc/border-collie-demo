"""Fruit detection B component.

The public interpretation API has no ROS2, NumPy, or Ultralytics dependency.
Runtime integrations live in :mod:`robotkit.perception.yolo.adapters`.
"""

from robotkit.perception.yolo.core import (
    COCO_FRUIT_CLASSES,
    Detection,
    FRUIT_CLASSES,
    FruitInterpretation,
    interpret_detections,
)

__all__ = [
    "COCO_FRUIT_CLASSES",
    "FRUIT_CLASSES",
    "Detection",
    "FruitInterpretation",
    "interpret_detections",
]
