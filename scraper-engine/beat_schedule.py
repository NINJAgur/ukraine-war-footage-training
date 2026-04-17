"""
scraper-engine/beat_schedule.py
Celery Beat periodic task definitions.

Schedule overview:
  - Funker530 scraper:     every hour
  - GeoConfirmed scraper:  every 2 hours
  - Kaggle datasets:       every night at 02:00 UTC
"""
from celery.schedules import crontab

BEAT_SCHEDULE = {
    # ── Funker530: scrape latest posts every hour ─────────────────────
    "scrape-funker530-hourly": {
        "task": "tasks.scrape_funker530.scrape_funker530",
        "schedule": crontab(minute=0),          # top of every hour
        "options": {"queue": "default"},
    },

    # ── GeoConfirmed: fetch latest incidents every 2 hours ───────────
    "scrape-geoconfirmed-bihourly": {
        "task": "tasks.scrape_geoconfirmed.scrape_geoconfirmed",
        "schedule": crontab(minute=15, hour="*/2"),   # :15 every 2nd hour
        "options": {"queue": "default"},
    },

    # ── Kaggle: download/refresh baseline datasets nightly at 02:00 UTC
    "download-kaggle-nightly": {
        "task": "tasks.download_kaggle.download_kaggle_datasets",
        "schedule": crontab(minute=0, hour=2),  # 02:00 UTC daily
        "options": {"queue": "default"},
    },
}
