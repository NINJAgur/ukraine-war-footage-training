# scraper-engine/Dockerfile
# Used in Phase 4 for production deployment.
# During Phase 1-3, run the scraper natively on Windows.

FROM python:3.11-slim

# Install system dependencies for Playwright Chromium
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Chromium dependencies
    libnss3 libnspr4 libdbus-1-3 libatk1.0-0 libatk-bridge2.0-0 \
    libcups2 libxkbcommon0 libatspi2.0-0 libx11-6 libxcomposite1 \
    libxdamage1 libxext6 libxfixes3 libxrandr2 libgbm1 libpango-1.0-0 \
    libcairo2 libasound2 \
    # ffmpeg for yt-dlp video merging
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright Chromium browser
RUN playwright install chromium --with-deps

# Copy application code
COPY . .

# Default command: start Celery worker
# Override with Beat for the scheduler container
CMD ["celery", "-A", "celery_app", "worker", "-Q", "default", "--loglevel=info", "--concurrency=4"]
