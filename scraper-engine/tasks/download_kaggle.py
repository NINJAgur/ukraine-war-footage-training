"""
scraper-engine/tasks/download_kaggle.py

Celery task: download Kaggle military datasets for Stage 1 baseline training.

Flow:
  1. For each dataset slug in settings.KAGGLE_BASELINE_DATASETS:
     a. Download via kagglehub to DATASETS_DIR
     b. Verify the download succeeded
  2. Idempotent: skips datasets that are already on disk.

Datasets used for Stage 1 (Baseline) training:
  - sudipchakrabarty/kiit-mita  — military images/annotations
  Add more to KAGGLE_BASELINE_DATASETS in .env as needed.
"""
import logging
import os
from pathlib import Path

import kagglehub

from celery_app import celery_app
from config import settings

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────

def _set_kaggle_env() -> None:
    """Ensure Kaggle credentials are set in environment for kagglehub."""
    if settings.KAGGLE_USERNAME:
        os.environ["KAGGLE_USERNAME"] = settings.KAGGLE_USERNAME
    if settings.KAGGLE_KEY:
        os.environ["KAGGLE_KEY"] = settings.KAGGLE_KEY


def _dataset_local_path(dataset_slug: str) -> Path:
    """
    Expected local path for a downloaded dataset.
    kagglehub downloads to a cache dir; we also maintain a symlink/copy in DATASETS_DIR.
    """
    # Normalize slug: "owner/dataset-name" → "owner__dataset-name"
    safe_name = dataset_slug.replace("/", "__")
    return settings.DATASETS_DIR / "kaggle" / safe_name


def _download_dataset(dataset_slug: str) -> dict:
    """
    Download a Kaggle dataset using kagglehub.
    Returns dict with local_path and status.
    """
    local_path = _dataset_local_path(dataset_slug)

    # Idempotency: skip if already downloaded
    if local_path.exists() and any(local_path.iterdir()):
        logger.info(f"Dataset already present: {dataset_slug} → {local_path}")
        return {
            "dataset": dataset_slug,
            "status": "already_exists",
            "local_path": str(local_path),
        }

    logger.info(f"Downloading Kaggle dataset: {dataset_slug}")
    local_path.mkdir(parents=True, exist_ok=True)

    # kagglehub downloads to its own cache, returns the cache path
    cache_path = kagglehub.dataset_download(dataset_slug)
    logger.info(f"kagglehub cache path: {cache_path}")

    # Count files in the downloaded dataset
    file_count = sum(1 for _ in Path(cache_path).rglob("*") if Path(_).is_file())
    logger.info(f"Dataset {dataset_slug}: {file_count} files downloaded")

    # Create a marker file so we know the download completed
    (local_path / ".kaggle_source").write_text(dataset_slug)
    (local_path / ".cache_path").write_text(str(cache_path))

    return {
        "dataset": dataset_slug,
        "status": "downloaded",
        "cache_path": str(cache_path),
        "local_path": str(local_path),
        "file_count": file_count,
    }


# ── Celery Task ───────────────────────────────────────────────────────

@celery_app.task(
    bind=True,
    name="tasks.download_kaggle.download_kaggle_datasets",
    queue="default",
    autoretry_for=(Exception,),
    max_retries=2,
    default_retry_delay=600,  # 10 min retry delay (large downloads)
)
def download_kaggle_datasets(self) -> dict:
    """
    Download all configured Kaggle baseline datasets.
    Idempotent: skips datasets already on disk.
    Uses a Redis lock to prevent overlapping Beat executions.
    """
    import redis as redis_lib

    r = redis_lib.from_url(settings.REDIS_URL)
    lock_key = "lock:download_kaggle_datasets"
    lock_ttl = 7200  # 2 hours (large datasets take time)

    if not r.set(lock_key, self.request.id, ex=lock_ttl, nx=True):
        logger.info(f"[{self.request.id}] download_kaggle_datasets already running — skipping")
        return {"status": "skipped", "reason": "lock_held"}

    logger.info(f"[{self.request.id}] download_kaggle_datasets started")
    _set_kaggle_env()

    datasets = settings.kaggle_dataset_list
    if not datasets:
        logger.warning(f"[{self.request.id}] No Kaggle datasets configured")
        return {"status": "skipped", "reason": "no_datasets_configured"}

    results = []
    errors = []

    try:
        for dataset_slug in datasets:
            try:
                result = _download_dataset(dataset_slug)
                results.append(result)
                logger.info(f"[{self.request.id}] ✓ {dataset_slug}: {result['status']}")
            except Exception as exc:
                logger.error(f"[{self.request.id}] ✗ {dataset_slug}: {exc}", exc_info=True)
                errors.append({"dataset": dataset_slug, "error": str(exc)})

        summary = {
            "total": len(datasets),
            "succeeded": len(results),
            "failed": len(errors),
            "results": results,
            "errors": errors,
        }
        logger.info(f"[{self.request.id}] download_kaggle_datasets completed: {summary}")
        return summary

    finally:
        r.delete(lock_key)


@celery_app.task(
    bind=True,
    name="tasks.download_kaggle.download_single_dataset",
    queue="default",
    autoretry_for=(Exception,),
    max_retries=2,
    default_retry_delay=300,
)
def download_single_dataset(self, dataset_slug: str) -> dict:
    """
    Download a single Kaggle dataset by slug.
    Used for on-demand pulls triggered by the Admin UI.
    Example slug: "sudipchakrabarty/kiit-mita"
    """
    logger.info(f"[{self.request.id}] download_single_dataset: {dataset_slug}")
    _set_kaggle_env()

    try:
        result = _download_dataset(dataset_slug)
        logger.info(f"[{self.request.id}] Completed: {result}")
        return result
    except Exception as exc:
        logger.error(f"[{self.request.id}] Failed: {exc}", exc_info=True)
        raise
