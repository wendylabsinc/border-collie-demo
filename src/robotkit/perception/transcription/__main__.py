"""Run the Whisper voice perception producer from files, a microphone, or ROS2."""

from __future__ import annotations

import argparse
import importlib
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from robotkit.client import WorldStateClient
from robotkit.perception.transcription.audio import AudioFormat, PCMChunker, read_wav
from robotkit.perception.transcription.backend import FasterWhisperBackend
from robotkit.perception.transcription.producer import VoicePerceptionProducer
from robotkit.runtime import (
    configure_logging,
    deployment_generation,
    instance_id,
    world_state_url,
)

LOGGER = logging.getLogger(__name__)


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--mode", choices=("file", "microphone", "ros2"), default="file")
    command.add_argument("--audio-file", type=Path)
    command.add_argument("--raw-pcm", action="store_true", help="audio file is headerless PCM")
    command.add_argument("--sample-rate", type=int, default=16_000)
    command.add_argument("--channels", type=int, default=1)
    command.add_argument("--sample-width", type=int, default=2)
    command.add_argument("--chunk-seconds", type=float, default=4.0)
    command.add_argument("--model", default=os.getenv("WHISPER_MODEL", "small.en"))
    command.add_argument("--device", default=os.getenv("WHISPER_DEVICE", "auto"))
    command.add_argument("--compute-type", default=os.getenv("WHISPER_COMPUTE_TYPE", "default"))
    command.add_argument("--language", default=os.getenv("WHISPER_LANGUAGE") or None)
    command.add_argument("--ros-topic", default=os.getenv("ROS_AUDIO_TOPIC", "/audio/audio"))
    command.add_argument(
        "--ros-message-type",
        default=os.getenv("ROS_AUDIO_MESSAGE_TYPE", "audio_common_msgs.msg:AudioData"),
        help="Python module and class, e.g. audio_common_msgs.msg:AudioData",
    )
    command.add_argument("--ros-data-field", default=os.getenv("ROS_AUDIO_DATA_FIELD", "data"))
    return command


def make_producer(args: argparse.Namespace, client: WorldStateClient) -> VoicePerceptionProducer:
    backend = FasterWhisperBackend(
        args.model,
        device=args.device,
        compute_type=args.compute_type,
        language=args.language,
    )
    return VoicePerceptionProducer(
        backend,
        client,
        producer_id=os.getenv("ROBOTKIT_PRODUCER_ID", "go2-voice-transcription"),
        instance_id=instance_id(),
        deployment_generation=deployment_generation(),
    )


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    args = parser().parse_args(argv)
    audio_format = AudioFormat(args.sample_rate, args.channels, args.sample_width)
    client = WorldStateClient(world_state_url())
    producer = make_producer(args, client)
    try:
        if args.mode == "file":
            _run_file(args, producer, audio_format)
        elif args.mode == "microphone":
            _run_microphone(args, producer, audio_format)
        else:
            _run_ros2(args, producer, audio_format)
    finally:
        client.close()


def _run_file(
    args: argparse.Namespace,
    producer: VoicePerceptionProducer,
    audio_format: AudioFormat,
) -> None:
    if args.audio_file is None:
        raise SystemExit("--audio-file is required in file mode")
    data = args.audio_file.read_bytes()
    if not args.raw_pcm:
        data, audio_format = read_wav(data)
    producer.transcribe_pcm(data, audio_format)


def _run_microphone(
    args: argparse.Namespace,
    producer: VoicePerceptionProducer,
    audio_format: AudioFormat,
) -> None:
    if audio_format.sample_width_bytes != 2:
        raise SystemExit("microphone mode currently requires 16-bit PCM")
    try:
        import sounddevice
    except ImportError as exc:
        raise SystemExit("microphone mode requires the sounddevice package") from exc
    chunker = PCMChunker(audio_format, args.chunk_seconds)
    frames = max(1, round(audio_format.sample_rate_hz * 0.1))
    LOGGER.info("listening to microphone in %.1f second chunks", args.chunk_seconds)
    with sounddevice.RawInputStream(
        samplerate=audio_format.sample_rate_hz,
        channels=audio_format.channels,
        dtype="int16",
        blocksize=frames,
    ) as stream:
        while True:
            data, overflowed = stream.read(frames)
            if overflowed:
                LOGGER.warning("microphone input overflow")
            for chunk in chunker.feed(bytes(data)):
                producer.transcribe_pcm(chunk, audio_format)


def _run_ros2(
    args: argparse.Namespace,
    producer: VoicePerceptionProducer,
    audio_format: AudioFormat,
) -> None:
    try:
        import rclpy
        from rclpy.node import Node
    except ImportError as exc:
        raise SystemExit("ROS2 mode requires rclpy") from exc
    message_type = _import_type(args.ros_message_type)
    chunker = PCMChunker(audio_format, args.chunk_seconds)

    class AudioNode(Node):
        def __init__(self) -> None:
            super().__init__("robotkit_voice_transcription")
            self.create_subscription(message_type, args.ros_topic, self.receive, 10)

        def receive(self, message: Any) -> None:
            data = getattr(message, args.ros_data_field)
            for chunk in chunker.feed(bytes(data)):
                try:
                    producer.transcribe_pcm(chunk, audio_format)
                except Exception:
                    self.get_logger().exception("failed to transcribe audio chunk")

    rclpy.init()
    node = AudioNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def _import_type(reference: str) -> type[Any]:
    try:
        module_name, class_name = reference.split(":", 1)
        return getattr(importlib.import_module(module_name), class_name)
    except (ValueError, ImportError, AttributeError) as exc:
        raise SystemExit(f"cannot import ROS message type {reference!r}") from exc


if __name__ == "__main__":
    main()
