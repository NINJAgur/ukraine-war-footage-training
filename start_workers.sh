#!/usr/bin/env bash
# Start all Celery workers and Beat schedulers.
# Phase 4 (Docker) will replace this with supervisord/entrypoints.
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

start_service "ml-engine worker" "$ROOT/ml-engine" \
    "celery -A celery_app worker -Q gpu --pool=solo --concurrency=1 --loglevel=info"

start_service "ml-engine beat" "$ROOT/ml-engine" \
    "celery -A celery_app beat --loglevel=info"

echo ""
echo "Beat schedule:"
echo "  00:00 UTC  scrape_funker530    (scraper-engine beat)"
echo "  00:15 UTC  scrape_geoconfirmed (scraper-engine beat)"
echo "  02:00 UTC  auto_label_batch    (ml-engine beat)  -- GDINO datasets, clips stay DOWNLOADED"
echo "  04:00 UTC  annotate_clips      (ml-engine beat)  -- YOLO annotation + finetune trigger"
echo ""
echo "All workers running. Press Ctrl+C to stop."
wait
