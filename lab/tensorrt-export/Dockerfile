# syntax=docker/dockerfile:1.6

# Same base as the media service, so the engine is built against the TensorRT
# and CUDA that the media container will load it with. An engine built against
# a different TensorRT will refuse to deserialize.
FROM dustynv/pytorch:2.7-r36.4.0-cu128-24.04

ARG DEBIAN_FRONTEND=noninteractive

ENV PIP_INDEX_URL=https://pypi.org/simple \
    PIP_EXTRA_INDEX_URL="" \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential ca-certificates libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /export
# Staged by `make stage` in the README; both files are versioned under
# media/model/ but live outside this build context.
COPY model/apple-pear-mango.pt ./apple-pear-mango.pt
COPY model/apple-pear-mango-reference.jpg ./apple-pear-mango-reference.jpg
COPY export.py ./

ENV EXPORT_WORK_DIR=/export \
    EXPORT_IMAGE_SIZE=640 \
    EXPORT_DEVICE=0 \
    EXPORT_FORCE=0 \
    EXPORT_TIMING_PASSES=30 \
    EXPORT_SERVE_PORT=8124

CMD ["python3", "export.py"]
