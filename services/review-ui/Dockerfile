# Review UI service
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY packages/cardio-echo-core /packages/cardio-echo-core
COPY services/review-ui/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir /packages/cardio-echo-core && \
    pip install --no-cache-dir -r /app/requirements.txt

COPY services/review-ui/app /app/app

EXPOSE 8090
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8090"]
