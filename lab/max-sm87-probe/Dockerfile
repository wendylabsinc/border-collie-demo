# syntax=docker/dockerfile:1.7
FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates wget python3 python3-pip \
    && wget -q \
        https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/cuda-keyring_1.1-1_all.deb \
        -O /tmp/cuda-keyring.deb \
    && dpkg -i /tmp/cuda-keyring.deb \
    && rm /tmp/cuda-keyring.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        cuda-cudart-12-6 cuda-cuobjdump-12-6 cuda-nvcc-12-6 \
    && ln -sfn /usr/local/cuda-12.6 /usr/local/cuda \
    && rm -rf /var/lib/apt/lists/*

ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu/nvidia:/usr/lib/aarch64-linux-gnu/tegra \
    MODULAR_NVPTX_COMPILER_PATH=/usr/local/cuda/bin/ptxas \
    MAX_SM87_CUOBJDUMP=/usr/local/cuda/bin/cuobjdump \
    MAX_SM87_ARTIFACT_DIR=/artifacts \
    MAX_SM87_PROBE_PORT=8124 \
    PYTHONUNBUFFERED=1

RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install \
        --extra-index-url https://modular.gateway.scarf.sh/simple/ \
        'max[serve]==26.4.0' numpy==1.26.4 \
        'fastapi>=0.115,<1' 'uvicorn[standard]>=0.30,<1' \
    && max --version \
    && mojo --version \
    && /usr/local/cuda/bin/ptxas --version \
    && /usr/local/cuda/bin/cuobjdump --version

WORKDIR /app
COPY probe.py /app/probe.py

EXPOSE 8124
CMD ["python3", "/app/probe.py"]
