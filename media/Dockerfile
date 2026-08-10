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
    python3 -m pip install -r /tmp/requirements.txt

WORKDIR /app
COPY model/model.engine /media/model.engine
COPY model/banana-specialist.pt /media/banana-specialist.pt
COPY __init__.py model_router.py perception_sidecar.py service_supervision.py ./media/

ENV UNITREE_ROBOT_IP=192.168.123.161 \
    PEAR_MODEL_PATH=/media/model.engine \
    BANANA_SPECIALIST_MODEL_PATH=/media/banana-specialist.pt \
    BANANA_SPECIALIST_MIN_CONFIDENCE=0.55 \
    BANANA_SPECIALIST_MIN_IOU=0.10 \
    BORDER_COLLIE_MEDIA_PORT=8111 \
    BORDER_COLLIE_BARK_UUID=161387de-21ab-4f0b-b4e9-97124b000d06

EXPOSE 8111

CMD ["uvicorn", "media.perception_sidecar:app", "--host", "0.0.0.0", "--port", "8111"]
