#!/bin/sh
set -eu

NSYS=/opt/nvidia/nsight-systems/2024.5.4/bin/nsys

exec "$NSYS" profile \
    --session-new=max-yolo11n-warm-20260807-2 \
    --trace=cuda,osrt,nvtx \
    --cuda-memory-usage=true \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --output=/artifacts/max-yolo11n-one-warm \
    --force-overwrite=true \
    python3 /app/profile_once.py
