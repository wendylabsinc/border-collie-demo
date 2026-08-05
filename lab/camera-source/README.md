# Camera source identity probe

This is a read-only lab probe for the Go2 WebRTC video track. It measures
decoded-frame PTS/time-base progression, callback cadence, and repeated image
content. It imports no motion client, exposes no control endpoint, and sends no
robot command.

Run it only while Woof is powered on and reachable from the WendyOS host:

```bash
wendy run --prefix lab/camera-source --device woof.local -y
```

The probe prints one line beginning with `CAMERA_SOURCE_RESULT=` after the
measurement, then serves the captured scene at
`http://woof.local:8122/snapshot.jpg`. Download the snapshot and stop the app
group:

```bash
curl --fail --output camera-source.jpg \
  http://woof.local:8122/snapshot.jpg
wendy device apps stop border-collie-camera-source-probe --device woof.local
```

A successful measurement proves that source identity was observable during
that run; it does not by itself qualify production freshness thresholds.

Checked-in results live in `results/`. Treat PTS as a source-owned ordering
marker, not a wall-clock freshness measurement: on `CAMERA-SOURCE-001`, PTS
advanced uniquely on every frame but its calculated presentation delta did not
match callback cadence.
