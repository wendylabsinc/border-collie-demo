# Hey Wendy voice service

This service is part of the `border-collie-demo` deployment. The bundled custom
`hey_wendy.onnx` acoustic model continuously gates local Parakeet ASR. After a
wake, a phrase containing exactly one of `apple`, `mango`, or `pear` creates a
Demo Run through the main app's existing `/api/run` safety boundary. `stop`,
`stop demo`, and `stop the demo` call the existing `/api/stop` boundary.

The service auto-arms only when microphone capture is live and the main dog app
reports the expected stage-default build label, production runtime, no active
run, no restart latch, and a ready activation preflight. It deliberately does
not claim the unrelated v33 release or schema identifiers from the source PR.

Fruit activation requires both one supported fruit and one allowlisted motion
verb (`find`, `follow`, `locate`, `search`, `seek`, or `go`). Ordinary stage
conversation that merely mentions fruit is ignored. Every accepted voice
command sends a durable `activation_id`, so the main app can replay the same
Run Result rather than creating a second Demo Run after an ambiguous response.

Wake inference and Parakeet transcription are local. Raw audio is not sent to
the dog API or any cloud service. `/models` stores only the checksum-verified
Parakeet cache; the custom `hey_wendy.onnx` model and pinned openWakeWord feature
models remain part of the immutable image.

The live dashboard is served on port 8092 and preserves the latest wake,
canonical command, microphone blocker, dog readiness/build blocker, and dog
acceptance result across browser reconnects.

Microphone discovery, stream-open failures, and runtime disconnects all return
to a bounded reconnect loop. `MICROPHONE_RETRY_INTERVAL_S` is measured in
seconds, defaults to `10.0`, and accepts `0.25..30.0`. Voice actions remain
disarmed whenever capture is unavailable. `AUDIO_DEVICE` may be an exact input
index, a case-insensitive device-name substring, or `auto`; retries cannot make
a USB receiver appear if it does not expose a PortAudio-compatible input.

`MICROPHONE_FRAME_TIMEOUT_S` is the maximum silence from the capture callback,
in seconds (default `3.0`). If a USB receiver remains enumerated but stops
producing frames, the session is closed and returned to the same discovery and
capture retry loop. This timer observes frame delivery, not acoustic silence.

Deploy with `scripts/deploy-stage-default --device woof.local`. The wrapper
uses Wendy's persisted `unless-stopped` policy, so the app group returns after
an ordinary process failure and after Go2/agent boot. A deliberate
`wendy device apps stop` remains authoritative and is not undone.
