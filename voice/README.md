# Hey Wendy voice service

This service is part of the `border-collie-demo` deployment. The bundled custom
`hey_wendy.onnx` acoustic model continuously gates local Parakeet ASR. After a
wake, a phrase containing exactly one of `apple`, `banana`, or `pear` creates a
Demo Run through the main app's existing `/api/run` safety boundary. `stop`,
`stop demo`, and `stop the demo` call the existing `/api/stop` boundary.

The service auto-arms only when microphone capture is live and the main dog app
reports the expected release identity, configuration schema, production runtime,
no active run, no restart latch, and a ready activation preflight.

The live dashboard is served on port 8092 and preserves the latest wake,
canonical command, and dog acceptance result across browser reconnects.
