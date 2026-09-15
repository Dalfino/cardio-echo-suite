# EchoNet-Dynamic service
# CPU + GPU compatible (select base via build arg)
ARG BASE=python:3.10-slim
FROM ${BASE}

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System deps for OpenCV/pydicom
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy shared cardio-echo-core package (built as part of monorepo)
COPY packages/cardio-echo-core /packages/cardio-echo-core

# Python deps
COPY services/echonet-dynamic/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir /packages/cardio-echo-core && \
    pip install --no-cache-dir -r /app/requirements.txt

# Install the upstream echonet package (preserved as a subtree)
COPY services/echonet-dynamic/upstream /app/upstream
RUN pip install --no-cache-dir -e /app/upstream

# Our wrapper code
COPY services/echonet-dynamic/app /app/app

EXPOSE 8001
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8001"]
