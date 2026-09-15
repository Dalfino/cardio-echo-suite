# MedSAM2 service
ARG BASE=python:3.10-slim
FROM ${BASE}

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY packages/cardio-echo-core /packages/cardio-echo-core
COPY services/medsam2/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir /packages/cardio-echo-core && \
    pip install --no-cache-dir -r /app/requirements.txt

COPY services/medsam2/app /app/app

EXPOSE 8005
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8005"]
