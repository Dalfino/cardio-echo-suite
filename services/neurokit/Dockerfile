# NeuroKit2 utility service (no model — pure signal processing)
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY packages/cardio-echo-core /packages/cardio-echo-core
COPY services/neurokit/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir /packages/cardio-echo-core && \
    pip install --no-cache-dir -r /app/requirements.txt

COPY services/neurokit/app /app/app

EXPOSE 8007
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8007"]
