# syntax=docker/dockerfile:1.7
FROM ubuntu:22.04

ADD --checksum=sha256:8845834d2b6481195b6fe5236ee35200fb86a01d916765a6072dacc495288a37 \
    https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/cuda-nvcc-12-6_12.6.85-1_arm64.deb \
    /tmp/cuda-nvcc-12-6.deb

RUN dpkg-deb -x /tmp/cuda-nvcc-12-6.deb / \
    && test -x /usr/local/cuda-12.6/bin/ptxas \
    && rm /tmp/cuda-nvcc-12-6.deb

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
        cuda-cuobjdump-12-6 cuda-libraries-12-6 \
        libcudnn9-cuda-12=9.3.0.75-1 \
    && ln -sfn /usr/local/cuda-12.6 /usr/local/cuda \
    && /usr/local/cuda/bin/ptxas --version \
    && /usr/local/cuda/bin/cuobjdump --version \
    && rm -rf /var/lib/apt/lists/*

ENV LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu/nvidia:/usr/lib/aarch64-linux-gnu/tegra \
    MODULAR_NVPTX_COMPILER_PATH=/usr/local/cuda/bin/ptxas \
    CUDA_DISABLE_PTX_JIT=1 \
    PYTHONUNBUFFERED=1 \
    MAX_SKELETON_PORT=8125

RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install \
        --extra-index-url https://modular.gateway.scarf.sh/simple/ \
        max==26.4.0 onnx==1.19.1 numpy==1.26.4 \
        'fastapi>=0.115,<1' 'uvicorn[standard]>=0.30,<1'

WORKDIR /app
COPY probe.py /app/probe.py

EXPOSE 8125
CMD ["python3", "/app/probe.py"]
