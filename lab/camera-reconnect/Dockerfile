# syntax=docker/dockerfile:1.6

FROM dustynv/pytorch:2.7-r36.4.0-cu128-24.04

ARG DEBIAN_FRONTEND=noninteractive

ENV PIP_INDEX_URL=https://pypi.org/simple \
    PIP_EXTRA_INDEX_URL="" \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential ca-certificates libffi8 libgl1 libglib2.0-0 \
      libportaudio2 libsrtp2-1 libssl3 portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /probe
COPY model/collie-fruit-yoloe11m.engine ./model.engine
COPY probe.py ./

ENV UNITREE_ROBOT_IP=192.168.123.161 \
    PEAR_CONFIDENCE_THRESHOLD=0.35 \
    RECONNECT_MINIMUM_SAMPLES=30 \
    RECONNECT_REQUIRED_MATCH_RATIO=0.80 \
    RECONNECT_GENERATION_TIMEOUT_S=30 \
    RECONNECT_OFFLINE_HOLD_S=2 \
    RECONNECT_SNAPSHOT_PORT=8124

CMD ["python3", "probe.py"]
