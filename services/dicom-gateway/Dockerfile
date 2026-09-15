# DICOM C-STORE gateway
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY packages/cardio-echo-core /packages/cardio-echo-core
COPY services/dicom-gateway/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir /packages/cardio-echo-core && \
    pip install --no-cache-dir -r /app/requirements.txt

COPY services/dicom-gateway/app /app/app

# Create data directories
RUN mkdir -p /data/incoming /data/results

EXPOSE 11112
CMD ["python", "-m", "app.main", "--port", "11112"]
