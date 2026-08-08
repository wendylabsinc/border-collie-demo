# TensorRT fixed-corpus control

This isolated job grades the production TensorRT fruit pipeline on the exact
SHA-checked corpus used by the MAX importer tournament. It does not connect to
the Go2 camera and contains no motion client.

The Mac serves the repository read-only so Woof can fetch the manifest and
fixture images. Update `CORPUS_MANIFEST_URL` and `CORPUS_ROOT_URL` in
`wendy.json` if the Mac's LAN address changes.

```bash
python3 -m http.server 8765 --bind 0.0.0.0 --directory ../..
wendy run --device woof.local --prefix lab/tensorrt-corpus-control \
  --dockerfile Dockerfile.tensorrt-control --chunking off
```

The terminal record is printed on a line prefixed with
`TENSORRT_CONTROL_RESULT=`. A candidate is judged against the per-fruit demo
acquisition thresholds, while every target record retains confidence, box,
route, crop-confirm details, and timing for later analysis.
