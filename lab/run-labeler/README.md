# Run labeler

Opens a recorded `evidence.zip` directly as a small, local labeling UI. It
previews the model's fruit box, stores human corrections in the browser, and
exports normalized YOLO-ready box coordinates as JSON.

```bash
python3 lab/run-labeler/server.py \
  --archive /path/to/evidence.zip \
  --class-name banana
```

Open `http://127.0.0.1:8120/`. The server only exposes frame filenames listed
in the archive manifest. Use a separate archive for each fruit and keep entire
recording sessions together when assigning train, validation, and test splits.
