FROM python:3.11-slim

# Playwright Chromium deps + ffmpeg for yt-dlp video merging
RUN apt-get update && apt-get install -y --no-install-recommends \
    libnss3 libnspr4 libdbus-1-3 libatk1.0-0 libatk-bridge2.0-0 \
    libcups2 libxkbcommon0 libatspi2.0-0 libx11-6 libxcomposite1 \
    libxdamage1 libxext6 libxfixes3 libxrandr2 libgbm1 libpango-1.0-0 \
    libcairo2 libasound2 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Build context is repo root — shared/ must sit alongside scraper-engine/ at /app
WORKDIR /app
COPY shared/ ./shared/

WORKDIR /app/scraper-engine
COPY scraper-engine/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    playwright install chromium --with-deps

COPY scraper-engine/ .

ENV PYTHONUNBUFFERED=1

CMD ["celery", "-A", "celery_app", "worker", "-Q", "default", "--loglevel=info", "--concurrency=4"]
