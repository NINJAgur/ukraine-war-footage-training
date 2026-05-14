FROM python:3.11-slim

# ffmpeg for yt-dlp video merging
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Build context is repo root — shared/ must sit alongside scraper-engine/ at /app
WORKDIR /app
COPY shared/ ./shared/

WORKDIR /app/scraper-engine
COPY scraper-engine/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY scraper-engine/ .

ENV PYTHONUNBUFFERED=1

CMD ["celery", "-A", "celery_app", "worker", "-Q", "default", "--loglevel=info", "--concurrency=4"]
