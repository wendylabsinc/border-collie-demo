"""Actuator adapters. ROS imports are isolated so tests need no ROS install."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import wave
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Protocol

from robotkit.contracts import EffectRecord


class EffectAdapter(Protocol):
    def apply(self, effect: EffectRecord) -> dict[str, object]: ...

    def close(self) -> None: ...


class LogAdapter:
    """Safe default for simulation, CI, and deployment smoke tests."""

    def apply(self, effect: EffectRecord) -> dict[str, object]:
        logging.getLogger("robotkit.executor").info(
            "simulated effect %s", json.dumps(effect.parameters, sort_keys=True)
        )
        return {"adapter": "log", "simulated": True}

    def close(self) -> None:
        pass


class RoutingAdapter:
    """Route independently deployable effect types to hardware adapters."""

    def __init__(self, routes: Mapping[str, EffectAdapter]) -> None:
        self._routes = dict(routes)

    def apply(self, effect: EffectRecord) -> dict[str, object]:
        adapter = self._routes.get(effect.effect_type)
        if adapter is None:
            raise ValueError(f"no adapter configured for effect type: {effect.effect_type}")
        return adapter.apply(effect)

    def close(self) -> None:
        closed: set[int] = set()
        for adapter in self._routes.values():
            if id(adapter) not in closed:
                adapter.close()
                closed.add(id(adapter))


class Ros2TwistAdapter:
    """Publish geometry_msgs/Twist to a configured Go2 ROS2 velocity topic."""

    def __init__(self, topic: str = "/cmd_vel") -> None:
        try:
            import rclpy
            from geometry_msgs.msg import Twist
        except ImportError as exc:
            raise RuntimeError(
                "ROS2 mode requires rclpy and geometry_msgs in the executor image"
            ) from exc
        self._rclpy = rclpy
        self._twist_type = Twist
        if not rclpy.ok():
            rclpy.init(args=None)
        self._node = rclpy.create_node("robotkit_effect_executor")
        self._publisher = self._node.create_publisher(Twist, topic, 10)

    def apply(self, effect: EffectRecord) -> dict[str, object]:
        message = self._twist_type()
        message.linear.x = float(effect.parameters["linear_x_mps"])
        message.angular.z = float(effect.parameters["angular_z_rps"])
        self._publisher.publish(message)
        self._rclpy.spin_once(self._node, timeout_sec=0.05)
        return {"adapter": "ros2_twist", "topic": self._publisher.topic_name}

    def close(self) -> None:
        self._node.destroy_node()
        self._rclpy.shutdown()


class Ros2StringCommandAdapter:
    """Publish a deployment-mapped semantic command on a ROS2 String topic.

    This is the hardware boundary for commands such as ``lie_down``. A Go2
    deployment may bridge this narrow topic to its audited Unitree Sport API
    implementation without coupling the planner to a vendor SDK version.
    """

    def __init__(self, topic: str = "/robotkit/posture_command") -> None:
        try:
            import rclpy
            from std_msgs.msg import String
        except ImportError as exc:
            raise RuntimeError("ROS2 command mode requires rclpy and std_msgs") from exc
        self._rclpy = rclpy
        self._message_type = String
        if not rclpy.ok():
            rclpy.init(args=None)
        self._node = rclpy.create_node("robotkit_posture_executor")
        self._publisher = self._node.create_publisher(String, topic, 10)

    def apply(self, effect: EffectRecord) -> dict[str, object]:
        if effect.effect_type != "unitree_lie_down":
            raise ValueError("posture adapter only accepts unitree_lie_down")
        message = self._message_type()
        message.data = "lie_down"
        self._publisher.publish(message)
        self._rclpy.spin_once(self._node, timeout_sec=0.05)
        return {
            "adapter": "ros2_string_command",
            "topic": self._publisher.topic_name,
            "command": "lie_down",
        }

    def close(self) -> None:
        self._node.destroy_node()


class WavPlaybackAdapter:
    """Play one deployment-configured WAV through an available system player.

    The asset and playback binary are resolved lazily on the first bark.  This
    keeps imports, unit tests, and non-audio deployments hardware-independent.
    The effect cannot provide a path, which prevents world-state input from
    selecting arbitrary local files.
    """

    def __init__(
        self,
        wav_path: str,
        *,
        player: str | None = None,
        max_duration_seconds: float = 10.0,
        runner: Callable[..., object] = subprocess.run,
        which: Callable[[str], str | None] = shutil.which,
    ) -> None:
        self._wav_path = Path(wav_path)
        self._configured_player = player
        self._max_duration_seconds = max_duration_seconds
        self._runner = runner
        self._which = which

    def apply(self, effect: EffectRecord) -> dict[str, object]:
        if effect.effect_type not in {"unitree_bark", "audio"}:
            raise ValueError("WavPlaybackAdapter only accepts bark/audio effects")
        path = self._wav_path
        if not path.is_file():
            raise RuntimeError(f"configured bark WAV does not exist: {path}")
        duration = self._validated_duration(path)
        player = self._resolve_player()
        self._runner([player, str(path)], check=True, timeout=self._max_duration_seconds + 2.0)
        return {
            "adapter": "wav_playback",
            "sound": "bark",
            "duration_seconds": duration,
            "player": Path(player).name,
        }

    def _validated_duration(self, path: Path) -> float:
        try:
            with wave.open(str(path), "rb") as wav:
                frame_rate = wav.getframerate()
                duration = wav.getnframes() / frame_rate if frame_rate > 0 else 0.0
                channels = wav.getnchannels()
        except (wave.Error, EOFError) as exc:
            raise RuntimeError(f"configured bark asset is not a valid WAV: {path}") from exc
        if duration <= 0 or duration > self._max_duration_seconds:
            raise RuntimeError(
                f"configured bark duration {duration:.3f}s is outside the safe limit"
            )
        if channels < 1 or channels > 2:
            raise RuntimeError("configured bark WAV must be mono or stereo")
        return round(duration, 6)

    def _resolve_player(self) -> str:
        candidates: Sequence[str] = (
            (self._configured_player,) if self._configured_player else ("aplay", "paplay", "afplay")
        )
        for candidate in candidates:
            resolved = self._which(candidate)
            if resolved:
                return resolved
        if self._configured_player:
            raise RuntimeError(f"configured audio player not found: {self._configured_player}")
        raise RuntimeError("no WAV player found; install aplay, paplay, or afplay")

    def close(self) -> None:
        pass
