"""
scraper-engine/celery_app.py
Celery application instance and Beat schedule for the scraper engine.

Start workers:
    celery -A celery_app worker -Q default --loglevel=info --concurrency=4

Start Beat scheduler:
    celery -A celery_app beat --loglevel=info
"""
from celery import Celery
from kombu import Queue
from config import settings

celery_app = Celery(
    "scraper_engine",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=[
        "tasks.scrape_funker530",
        "tasks.scrape_geoconfirmed",
        "tasks.download_kaggle",
    ],
)

celery_app.conf.update(
    # Serialization
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],

    # Reliability
    task_track_started=True,
    task_acks_late=True,           # re-queue task if worker crashes mid-execution
    worker_prefetch_multiplier=1,  # don't prefetch — prevents one worker hoarding tasks

    # Queues
    task_queues=[Queue("default")],
    task_default_queue="default",

    # Result expiry (24 hours)
    result_expires=86400,

    # Beat schedule (loaded from beat_schedule.py)
    beat_schedule_filename="celerybeat-schedule",
)

# Import and register Beat schedule
from beat_schedule import BEAT_SCHEDULE  # noqa: E402
celery_app.conf.beat_schedule = BEAT_SCHEDULE
