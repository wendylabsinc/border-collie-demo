FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libportaudio2 \
        libasound2 \
        alsa-utils \
        libsndfile1 \
        libgomp1 \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Keep the two shared openWakeWord feature models checksum-pinned outside
# /models. The persistent /models mount is reserved for the Parakeet cache and
# would otherwise hide image-baked files.
RUN mkdir -p /opt/openwakeword-models \
    && curl -fL --retry 3 \
      https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.onnx \
      -o /opt/openwakeword-models/melspectrogram.onnx \
    && echo "ba2b0e0f8b7b875369a2c89cb13360ff53bac436f2895cced9f479fa65eb176f  /opt/openwakeword-models/melspectrogram.onnx" | sha256sum -c - \
    && curl -fL --retry 3 \
      https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.onnx \
      -o /opt/openwakeword-models/embedding_model.onnx \
    && echo "70d164290c1d095d1d4ee149bc5e00543250a7316b59f31d056cff7bd3075c1f  /opt/openwakeword-models/embedding_model.onnx" | sha256sum -c -

# The custom acoustic model is stable and deliberately precedes volatile source.
COPY hey_wendy.onnx /app/hey_wendy.onnx
COPY *.py ./

ENV PYTHONUNBUFFERED=1 \
    MODEL_DIR=/models \
    WAKE_WORD=/app/hey_wendy.onnx \
    OWW_MELSPEC_PATH=/opt/openwakeword-models/melspectrogram.onnx \
    OWW_EMBEDDING_PATH=/opt/openwakeword-models/embedding_model.onnx

CMD ["python", "app.py"]
