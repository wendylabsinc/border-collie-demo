# Voice transcription perception producer

This independently runnable B component converts 16-bit PCM into two world-state
observations: a durable `voice.transcript` and a short-lived `voice.intent`. It uses
`faster-whisper`; model, microphone, and ROS2 imports are lazy, so pure tests do not
need them.

Install optional runtime dependencies:

```sh
pip install -r src/robotkit/perception/transcription/requirements.txt
```

Transcribe a WAV file:

```sh
python -m robotkit.perception.transcription --mode file --audio-file command.wav
```

Capture four-second microphone chunks:

```sh
python -m robotkit.perception.transcription --mode microphone --chunk-seconds 4
```

Subscribe to ROS2 `audio_common_msgs/msg/AudioData` (raw PCM payload):

```sh
python -m robotkit.perception.transcription --mode ros2 \
  --ros-topic /audio/audio \
  --ros-message-type audio_common_msgs.msg:AudioData
```

For a custom ROS message, use `--ros-message-type package.msg:Class` and set
`--ros-data-field` to its byte-array field. Sample rate, channel count, sample width,
chunk duration, Whisper model/device, producer identity, deployment generation, and
world-state URL are all configurable through flags or the standard RobotKit
environment variables.
