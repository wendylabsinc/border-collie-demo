# Pear evidence probe

This is an isolated, read-only camera and GPU-inference test. It reuses the
TensorRT pear model validated in the previous prototype but imports no motion
client and exposes no robot-control endpoint.

Before running, place the model at
`model/collie-fruit-yoloe11m.engine`. The model directory is intentionally
gitignored because this local TensorRT artifact is hardware-specific.

With Woof stationary and a pear clearly in view:

```bash
wendy run --prefix lab/pear-evidence --device woof.local -y
```

After `PEAR_EVIDENCE_RESULT=` appears, download the annotated snapshot and stop
the probe:

```bash
curl --fail --output pear-evidence.jpg \
  http://woof.local:8123/snapshot.jpg
wendy device apps stop border-collie-pear-evidence-probe --device woof.local
```

The initial acceptance rule requires at least 30 eligible inference samples and
pear detections at or above 0.35 confidence in at least 80% of them. These are
test criteria, not yet production thresholds; WDY-2294 owns that decision.
