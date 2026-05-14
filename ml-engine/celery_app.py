"""
ml-engine/celery_app.py
Celery application instance for the ML engine (GPU worker).

Start GPU worker (concurrency=1 — single GPU):
    celery -A celery_app worker -Q gpu --loglevel=info --concurrency=1

GPU tasks must NOT run concurrently; a single worker with concurrency=1
ensures only one training/inference job runs at a time.
"""
from celery import Celery
from celery.schedules import crontab
from kombu import Queue
from config import settings

celery_app = Celery(
    "ml_engine",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=[
        "tasks.auto_label",
        "tasks.package_dataset",
        "tasks.annotate_clips",
        "tasks.train_baseline",
        "tasks.train_finetune",
    ],
)

from celery.signals import worker_ready


@worker_ready.connect
def on_worker_ready(**kwargs):
    from db.session import init_db
    init_db()


celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],

    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,   # never prefetch GPU tasks

    task_queues=[Queue("gpu")],
    task_default_queue="gpu",

    result_expires=86400,

    beat_schedule={
        # 02:00 UTC — GDINO auto-label all DOWNLOADED clips (2h after scrape)
        "auto-label-batch-daily": {
            "task": "tasks.auto_label.auto_label_batch",
            "schedule": crontab(minute=0, hour=2),
            "options": {"queue": "gpu"},
        },
        # 04:00 UTC — YOLO annotation on LABELED clips (2h after GDINO batch)
        "annotate-clips-daily": {
            "task": "tasks.annotate_clips.annotate_clips",
            "schedule": crontab(minute=0, hour=4),
            "options": {"queue": "gpu"},
        },
    },
)
