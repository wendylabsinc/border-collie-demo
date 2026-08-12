FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

RUN useradd --create-home --uid 10001 robotkit \
    && mkdir -p /data \
    && chown robotkit:robotkit /data
USER robotkit

CMD ["uvicorn", "robotkit.world_state.app:app", "--host", "0.0.0.0", "--port", "8080"]

