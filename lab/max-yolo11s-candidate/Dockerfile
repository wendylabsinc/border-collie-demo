# syntax=docker/dockerfile:1.7
FROM ubuntu:22.04

ADD --checksum=sha256:8845834d2b6481195b6fe5236ee35200fb86a01d916765a6072dacc495288a37 \
    https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/cuda-nvcc-12-6_12.6.85-1_arm64.deb \
    /tmp/cuda-nvcc-12-6.deb

ADD --checksum=sha256:404b1d921366d94f60a027298523e295c5484ee6f6e3b8a27da304ad4fd92bad \
    https://repo.download.nvidia.com/jetson/common/pool/main/n/nsight-systems-2024.5.4/nsight-systems-2024.5.4_2024.5.4.34-245434855735v0_arm64.deb \
    /tmp/nsight-systems.deb

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
        /tmp/nsight-systems.deb \
    && ln -sfn /usr/local/cuda-12.6 /usr/local/cuda \
    && /usr/local/cuda/bin/ptxas --version \
    && /opt/nvidia/nsight-systems/2024.5.4/bin/nsys --version \
    && rm /tmp/nsight-systems.deb \
    && rm -rf /var/lib/apt/lists/*

ENV LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu/nvidia:/usr/lib/aarch64-linux-gnu/tegra \
    MODULAR_NVPTX_COMPILER_PATH=/usr/local/cuda/bin/ptxas \
    PYTHONUNBUFFERED=1 \
    MAX_PROBE_PORT=8126

RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install \
        --extra-index-url https://modular.gateway.scarf.sh/simple/ \
        max==26.4.0 onnx==1.19.1 numpy==1.26.4 \
        'fastapi>=0.115,<1' 'uvicorn[standard]>=0.30,<1'

RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install 'Pillow>=10,<13'

WORKDIR /app
COPY max_onnx.py /app/max_onnx.py
COPY snapshot_inference.py /app/snapshot_inference.py
COPY probe.py /app/probe.py
COPY profile_once.py /app/profile_once.py
COPY profile_once.sh /app/profile_once.sh
ADD --checksum=sha256:3d8d177005ffd5dbedc66f38d7d8a911b7cc9b35f61ecc11d808152166d50e1c \
    http://192.168.0.250:8765/runs/detect/artifacts/fruit-model/experiments/MAX-NATIVE-007/yolo11n-640-go2-head/weights/best.onnx \
    /models/fruit-yolo11n-go2-640.onnx

EXPOSE 8126
CMD ["python3", "/app/probe.py"]
