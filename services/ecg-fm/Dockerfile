# ECG-FM service
ARG BASE=python:3.10-slim
FROM ${BASE}

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DOWNLOAD_TIMEOUT=60

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY packages/cardio-echo-core /packages/cardio-echo-core
COPY services/ecg-fm/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir /packages/cardio-echo-core && \
    pip install --no-cache-dir -r /app/requirements.txt

COPY services/ecg-fm/app /app/app

EXPOSE 8004
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8004"]
