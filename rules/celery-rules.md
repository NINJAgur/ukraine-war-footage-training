# Celery Rules — Ukraine Combat Footage Project

These rules are enforced on all Celery task code in `scraper-engine/` and `ml-engine/`.
Violations must be corrected before merging.

---

## MANDATORY RULES

### 1. All Tasks Must Be Idempotent
Running a task twice with the same input must produce the same result.
No duplicate DB records, no duplicate files.

```python
# CORRECT — idempotent: insert-or-ignore + file existence check
@celery_app.task(bind=True)
def download_clip(self, url: str):
    url_hash = sha256(url.encode()).hexdigest()
    # Skip if already downloaded
    if db.query(Clip).filter_by(url_hash=url_hash).first():
        return {"status": "skipped", "reason": "already_exists"}
    # ... proceed with download

# WRONG — creates duplicates on retry
@celery_app.task
def download_clip(url: str):
    clip = Clip(url=url)
    db.add(clip)
    db.commit()  # duplicate on retry
```

### 2. Always Use `bind=True` with Standard Retry Config
```python
# CORRECT — standard task signature for this project
@celery_app.task(
    bind=True,
    autoretry_for=(Exception,),
    max_retries=3,
    default_retry_delay=60,  # seconds before retry
    queue='default'          # or 'gpu' for ML tasks
)
def my_task(self, arg1: str, arg2: int):
    logger.info(f"[{self.request.id}] Starting my_task({arg1}, {arg2})")
    ...
```

### 3. GPU Tasks Go to the `gpu` Queue
```python
# CORRECT — GPU tasks isolated to their own queue
@celery_app.task(bind=True, queue='gpu', max_retries=2)
def train_baseline(self, dataset_ids: list[int]):
    ...

# WRONG — GPU task on default queue may run concurrently with other GPU tasks
@celery_app.task(bind=True)
def train_baseline(self, dataset_ids: list[int]):
    ...
```

GPU worker must be started with `concurrency=1`:
```bash
celery -A ml_engine.celery_app worker -Q gpu --concurrency=1
```

### 4. Log Task ID at Start and End
```python
import logging
logger = logging.getLogger(__name__)

@celery_app.task(bind=True)
def scrape_funker530(self):
    logger.info(f"[{self.request.id}] scrape_funker530 started")
    try:
        result = _do_scrape()
        logger.info(f"[{self.request.id}] scrape_funker530 completed: {result}")
        return result
    except Exception as exc:
        logger.error(f"[{self.request.id}] scrape_funker530 failed: {exc}")
        raise
```

### 5. Emit Progress for Long-Running Tasks
```python
# CORRECT — training tasks update state for WebSocket consumers
@celery_app.task(bind=True, queue='gpu')
def train_baseline(self, dataset_ids: list[int]):
    for epoch in range(epochs):
        # ... train one epoch
        self.update_state(
            state='PROGRESS',
            meta={
                'epoch': epoch + 1,
                'total_epochs': epochs,
                'loss': float(loss),
                'mAP50': float(map50)
            }
        )
```

### 6. Use Redis Locks to Prevent Overlapping Beat Tasks
```python
from redis import Redis
from celery.utils.log import get_task_logger

REDIS_LOCK_TTL = 3600  # 1 hour

@celery_app.task(bind=True)
def scrape_funker530(self):
    redis = Redis.from_url(settings.REDIS_URL)
    lock_key = "lock:scrape_funker530"
    if not redis.set(lock_key, self.request.id, ex=REDIS_LOCK_TTL, nx=True):
        logger.info(f"[{self.request.id}] Task already running, skipping")
        return {"status": "skipped", "reason": "lock_held"}
    try:
        return _do_scrape()
    finally:
        redis.delete(lock_key)
```

### 7. Never Call `time.sleep()` — Use Celery Retry with `countdown`
```python
# CORRECT — exponential backoff via Celery retry
except RateLimitError as exc:
    raise self.retry(exc=exc, countdown=2 ** self.request.retries * 30)

# WRONG — blocks the worker thread
import time
time.sleep(60)
```

### 8. Database Sessions Managed as Context Managers
```python
# CORRECT
from scraper_engine.db.session import get_sync_session

@celery_app.task(bind=True)
def my_task(self):
    with get_sync_session() as session:
        clip = session.query(Clip).filter_by(id=1).first()
        # session auto-committed and closed

# WRONG — session may not be closed on exception
session = Session()
clip = session.query(Clip).first()
# if exception here, session leaks
```

---

## Celery App Configuration Template

```python
# celery_app.py
from celery import Celery
from kombu import Queue

celery_app = Celery(
    'app_name',
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=['tasks.module_name']
)

celery_app.conf.update(
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
    task_track_started=True,
    task_acks_late=True,          # ack only after task completes (safer)
    worker_prefetch_multiplier=1, # don't prefetch GPU tasks
    task_queues=[
        Queue('default'),
        Queue('gpu'),
    ],
    task_default_queue='default',
)
```
