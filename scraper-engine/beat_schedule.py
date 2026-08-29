"""
scraper-engine/beat_schedule.py
Celery Beat periodic task definitions.

Schedule overview:
  - Funker530 scraper:     daily at 00:00 UTC
  - GeoConfirmed scraper:  daily at 00:15 UTC
  inference-engine runs auto_label_batch at 03:05 and annotate_clips at 03:35 UTC
  (see inference-engine/celery_app.py) — both after downloads finish.
"""
from celery.schedules import crontab

BEAT_SCHEDULE = {
    # ── Funker530: scrape last 24h posts, dispatch yt-dlp downloads ───
    "scrape-funker530-daily": {
        "task": "tasks.scrape_funker530.scrape_funker530",
        "schedule": crontab(minute=0, hour=0),
        "options": {"queue": "default"},
    },

    # ── GeoConfirmed: same, offset by 15 min to avoid DB contention ──
    "scrape-geoconfirmed-daily": {
        "task": "tasks.scrape_geoconfirmed.scrape_geoconfirmed",
        "schedule": crontab(minute=15, hour=0),
        "options": {"queue": "default"},
    },
}
