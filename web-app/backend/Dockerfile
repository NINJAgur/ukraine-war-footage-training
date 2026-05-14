FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Build context is repo root — shared/ must be at /app/shared,
# and backend at /app/web-app/backend so that parent.parent.parent == /app (repo root)
WORKDIR /app
COPY shared/ ./shared/

WORKDIR /app/web-app/backend
COPY web-app/backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY web-app/backend/ .

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
