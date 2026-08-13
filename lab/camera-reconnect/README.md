# Camera reconnect probe

This isolated probe measures a controlled, client-initiated WebRTC camera
reconnect while Woof remains stationary. It imports no motion client and
exposes no robot-control endpoint.

The probe establishes a baseline connection generation and qualifies fresh
pear evidence. It then deliberately closes that camera connection, waits two
seconds, creates a new connection with a new opaque generation, and repeats the
full preflight. It never treats the replacement connection as a continuation
of the first generation.

Place the TensorRT model at
`model/collie-fruit-yoloe11m.engine`. With the pear visible:

```bash
wendy run --prefix lab/camera-reconnect --device woof.local -y
```

A pass requires, independently in each generation:

- present, strictly advancing PTS and a stable time base;
- at least 30 inference samples;
- pear matches at or above 0.35 confidence in at least 80% of samples; and
- a saved annotated snapshot from the replacement generation.

After `CAMERA_RECONNECT_RESULT=` appears, save the snapshot and explicitly stop
the probe:

```bash
curl --fail --output camera-reconnect.jpg \
  http://woof.local:8124/snapshot.jpg
wendy device apps stop border-collie-camera-reconnect-probe --device woof.local
```

This validates clean client disconnect/reconnect behavior. It does not simulate
packet loss, a robot-side camera crash, or an application restart during an
active Demo Run.
