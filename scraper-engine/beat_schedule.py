"""
scraper-engine/beat_schedule.py
Celery Beat periodic task definitions.

Schedule overview:
  - Funker530 scraper:     daily at 00:00 UTC
  - GeoConfirmed scraper:  daily at 00:15 UTC
  ml-engine annotate_clips fires at 04:00 UTC (after downloads finish).
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
