FROM nvcr.io/nvidia/pytorch:24.01-py3

# OpenCV + ffmpeg for video annotation
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Build context is repo root — shared/ must sit alongside ml-engine/ at /app
WORKDIR /app
COPY shared/ ./shared/

WORKDIR /app/ml-engine
COPY ml-engine/requirements.txt .
# Install groundingdino-py from requirements, then upgrade to IDEA-Research source for full CUDA support
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --force-reinstall \
        git+https://github.com/IDEA-Research/GroundingDINO.git

COPY ml-engine/ .

# Entrypoint downloads GDINO + YOLO weights into ml_checkpoints volume on cold start
COPY ml-engine/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["/entrypoint.sh"]
CMD ["celery", "-A", "celery_app", "worker", "-Q", "gpu", "--loglevel=info", "--concurrency=1"]
