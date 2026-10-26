# Gauge: one image that serves the API and the built website from a single address.
# Stage 1 builds the React frontend; stage 2 is the Python server (with ffmpeg for video and audio).

FROM node:22-slim AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.13-slim
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 GAUGE_DATA_DIR=/data
WORKDIR /app
COPY backend/requirements.txt backend/requirements.txt
RUN pip install -r backend/requirements.txt
COPY backend/app backend/app
COPY --from=web /web/dist frontend/dist
RUN mkdir -p /data
WORKDIR /app/backend
EXPOSE 8000
# One process on purpose: jobs run as background tasks inside it and share one SQLite file.
CMD ["sh", "-c", "exec python -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
