"""
ml-engine/celery_app.py
Celery application instance for the ML engine (GPU worker).

Start GPU worker (concurrency=1 — single GPU):
    celery -A celery_app worker -Q gpu --loglevel=info --concurrency=1

GPU tasks must NOT run concurrently; a single worker with concurrency=1
ensures only one training/inference job runs at a time.
"""
from celery import Celery
from kombu import Queue
from config import settings

celery_app = Celery(
    "ml_engine",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=[
        "tasks.auto_label",
        "tasks.package_dataset",
        "tasks.render_annotated",
        "tasks.train_baseline",
        "tasks.train_finetune",
    ],
)

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
)
