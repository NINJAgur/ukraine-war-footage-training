from celery import Celery
from celery.schedules import crontab
from kombu import Queue
from config import settings

celery_app = Celery(
    "inference_engine",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=[
        "tasks.auto_label",
        "tasks.package_dataset",
        "tasks.annotate_clips",
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
    worker_prefetch_multiplier=1,
    task_queues=[Queue("pipeline")],
    task_default_queue="pipeline",
    result_expires=86400,
    beat_schedule={
        # VM starts at 03:00 UTC — fire 5min after boot
        "auto-label-batch-daily": {
            "task": "tasks.auto_label.auto_label_batch",
            "schedule": crontab(minute=5, hour=3),
            "options": {"queue": "pipeline"},
        },
        # Fire 30min after GDINO batch; waits behind auto_label tasks in Q=pipeline
        "annotate-clips-daily": {
            "task": "tasks.annotate_clips.annotate_clips",
            "schedule": crontab(minute=35, hour=3),
            "options": {"queue": "pipeline"},
        },
    },
)
