from celery import Celery
from kombu import Queue
from config import settings

celery_app = Celery(
    "training_engine",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=[
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
    worker_prefetch_multiplier=1,
    task_queues=[Queue("training")],
    task_default_queue="training",
    result_expires=86400,
)
