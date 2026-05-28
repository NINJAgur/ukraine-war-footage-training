#!/usr/bin/env bash
# Start Celery workers for local native development.
# Scraper (Q=default) and inference-engine (Q=pipeline) workers + beat schedulers.
# In production these are replaced by Docker Compose (scraper) and systemd (inference-engine).
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"

start_service() {
    local name="$1"
    local dir="$2"
    local cmd="$3"
    echo "[start_workers] $name"
    cd "$dir"
    eval "$cmd" &
    cd "$ROOT"
}

start_service "scraper-engine worker" "$ROOT/scraper-engine" \
    "celery -A celery_app worker -Q default --loglevel=info --concurrency=4"

start_service "scraper-engine beat" "$ROOT/scraper-engine" \
    "celery -A celery_app beat --loglevel=info"

start_service "inference-engine worker" "$ROOT/inference-engine" \
    "celery -A celery_app worker -Q pipeline --pool=solo --concurrency=1 --loglevel=info"

start_service "inference-engine beat" "$ROOT/inference-engine" \
    "celery -A celery_app beat --loglevel=info"

echo ""
echo "Beat schedule:"
echo "  00:00 UTC  scrape_funker530    (scraper-engine beat)"
echo "  00:15 UTC  scrape_geoconfirmed (scraper-engine beat)"
echo "  03:05 UTC  auto_label_batch    (inference-engine beat)  -- GDINO datasets"
echo "  03:35 UTC  annotate_clips      (inference-engine beat)  -- YOLO annotation"
echo ""
echo "All workers running. Press Ctrl+C to stop."
echo "To run pipeline steps manually: python run_local.py status"
wait
