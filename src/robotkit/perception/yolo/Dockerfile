FROM ros:humble-ros-base

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN apt-get update \
    && apt-get install --yes --no-install-recommends python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --no-cache-dir . \
    && python3 -m pip install --no-cache-dir -r src/robotkit/perception/yolo/requirements.txt

ENV YOLO_INPUT_MODE=ros2 \
    YOLO_IMAGE_TOPIC=/camera/image_raw \
    YOLO_MODEL=yolo11n.pt \
    YOLO_CLASSES=apple,banana,orange

CMD ["python3", "-m", "robotkit.perception.yolo"]
