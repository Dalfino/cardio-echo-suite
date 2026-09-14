# EchoPrime service (HF Hub wrapper + LoRA fine-tune scaffold)
ARG BASE=python:3.10-slim
FROM ${BASE}

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DOWNLOAD_TIMEOUT=60

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app /app/app

EXPOSE 8002
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8002"]
