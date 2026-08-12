"""ROS2 adapter for temperature and battery high/low observations."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

from robotkit.client import WorldStateClient
from robotkit.contracts import Observation
from robotkit.perception.high_low.logic import (
    BatteryThresholds,
    Evaluation,
    TemperatureThresholds,
    evaluate_battery,
    evaluate_temperature,
)
from robotkit.runtime import (
    configure_logging,
    deployment_generation,
    instance_id,
    world_state_url,
)


def _source_time(message: Any) -> datetime:
    stamp = getattr(getattr(message, "header", None), "stamp", None)
    seconds = int(getattr(stamp, "sec", 0))
    nanoseconds = int(getattr(stamp, "nanosec", 0))
    if seconds == 0 and nanoseconds == 0:
        return datetime.now(timezone.utc)
    return datetime.fromtimestamp(seconds + nanoseconds / 1_000_000_000, timezone.utc)


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


class HighLowRosProducer:
    def __init__(self) -> None:
        try:
            import rclpy
            from sensor_msgs.msg import BatteryState, Temperature
        except ImportError as exc:
            raise RuntimeError(
                "high-low ROS2 mode requires rclpy and sensor_msgs"
            ) from exc

        self._rclpy = rclpy
        self._client = WorldStateClient(world_state_url())
        self._producer_id = os.getenv("ROBOTKIT_PRODUCER_ID", "go2-health-high-low")
        self._instance_id = instance_id()
        self._generation = deployment_generation()
        self._temperature_thresholds = TemperatureThresholds(
            critical_low_c=_float("TEMPERATURE_CRITICAL_LOW_C", -10),
            low_c=_float("TEMPERATURE_LOW_C", 0),
            high_c=_float("TEMPERATURE_HIGH_C", 60),
            critical_high_c=_float("TEMPERATURE_CRITICAL_HIGH_C", 75),
        )
        self._battery_thresholds = BatteryThresholds(
            critical_low_fraction=_float("BATTERY_CRITICAL_LOW_FRACTION", 0.10),
            low_fraction=_float("BATTERY_LOW_FRACTION", 0.20),
            high_fraction=_float("BATTERY_HIGH_FRACTION", 0.95),
            critical_high_fraction=_float("BATTERY_CRITICAL_HIGH_FRACTION", 1.01),
        )
        rclpy.init(args=None)
        self._node = rclpy.create_node("robotkit_high_low_perception")
        self._node.create_subscription(
            Temperature,
            os.getenv("ROS2_TEMPERATURE_TOPIC", "/temperature"),
            self._on_temperature,
            10,
        )
        self._node.create_subscription(
            BatteryState,
            os.getenv("ROS2_BATTERY_TOPIC", "/battery_state"),
            self._on_battery,
            10,
        )

    def _publish(
        self, stream: str, observation_type: str, at: datetime, evaluation: Evaluation
    ) -> None:
        source_tick = int(at.timestamp() * 1_000_000_000)
        self._client.publish_observation(
            Observation(
                idempotency_key=f"{self._instance_id}:{stream}:{source_tick}",
                producer_id=self._producer_id,
                instance_id=self._instance_id,
                deployment_generation=self._generation,
                stream=stream,
                observation_type=observation_type,
                observed_at=at,
                confidence=1.0 if evaluation.usable else 0.0,
                ttl_seconds=_float("HEALTH_TTL_SECONDS", 5),
                payload=evaluation.payload,
            )
        )

    def _on_temperature(self, message: Any) -> None:
        at = _source_time(message)
        evaluation = evaluate_temperature(
            float(message.temperature),
            float(message.variance),
            thresholds=self._temperature_thresholds,
        )
        self._publish("health.temperature", "temperature.band.v1", at, evaluation)

    def _on_battery(self, message: Any) -> None:
        at = _source_time(message)
        charging = None
        status = int(getattr(message, "power_supply_status", 0))
        # sensor_msgs/BatteryState constants: CHARGING=1, DISCHARGING=2.
        if status in (1, 2):
            charging = status == 1
        evaluation = evaluate_battery(
            float(message.percentage),
            voltage_v=float(message.voltage),
            current_a=float(message.current),
            charging=charging,
            thresholds=self._battery_thresholds,
        )
        self._publish("health.battery", "battery.band.v1", at, evaluation)

    def run(self) -> None:
        try:
            self._rclpy.spin(self._node)
        finally:
            self._node.destroy_node()
            self._rclpy.shutdown()
            self._client.close()


def main() -> None:
    configure_logging()
    HighLowRosProducer().run()


if __name__ == "__main__":
    main()

