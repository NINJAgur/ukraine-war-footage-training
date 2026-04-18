FROM nvcr.io/nvidia/pytorch:24.01-py3

WORKDIR /app

# System deps: ffmpeg for H.264 re-encoding in render_annotated
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir \
       git+https://github.com/IDEA-Research/GroundingDINO.git

COPY . .

# Start GPU Celery worker (concurrency=1 — single GPU)
CMD ["celery", "-A", "celery_app", "worker", "-Q", "gpu", \
     "--loglevel=info", "--concurrency=1"]
