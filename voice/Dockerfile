FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libportaudio2 \
        libasound2 \
        alsa-utils \
        libsndfile1 \
        libgomp1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# openWakeWord does not package these shared feature-extractor models. Cache
# them in the image and fail the build if upstream did not provide both files.
RUN python -c "import openwakeword.utils as u; u.download_models()"
RUN python - <<'PYCHECK'
import os, openwakeword
resources = os.path.join(
    os.path.dirname(openwakeword.__file__), "resources", "models"
)
available = os.listdir(resources) if os.path.isdir(resources) else []
missing = [
    name
    for name in ("melspectrogram.onnx", "embedding_model.onnx")
    if name not in available
]
if missing:
    raise SystemExit(f"missing wake-word feature models {missing} in {resources}")
PYCHECK

# The custom acoustic model is stable and deliberately precedes volatile source.
COPY hey_wendy.onnx /app/hey_wendy.onnx
COPY *.py ./

ENV PYTHONUNBUFFERED=1 \
    MODEL_DIR=/models \
    WAKE_WORD=/app/hey_wendy.onnx

CMD ["python", "app.py"]
