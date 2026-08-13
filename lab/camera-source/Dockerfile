# syntax=docker/dockerfile:1.6

FROM python:3.11-slim-bookworm AS builder

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --no-cache-dir -r requirements.txt


FROM python:3.11-slim-bookworm

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates libffi8 libportaudio2 libsrtp2-1 libssl3 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.11/site-packages \
  /usr/local/lib/python3.11/site-packages

WORKDIR /probe
COPY probe.py ./

ENV PYTHONUNBUFFERED=1 \
    UNITREE_ROBOT_IP=192.168.123.161 \
    CAMERA_PROBE_DURATION_S=30 \
    CAMERA_PROBE_STARTUP_TIMEOUT_S=15

CMD ["python", "probe.py"]
