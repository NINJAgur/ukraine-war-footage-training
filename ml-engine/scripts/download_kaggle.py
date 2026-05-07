"""
ml-engine/scripts/download_kaggle.py

CLI: download Kaggle datasets into KAGGLE_CACHE_DIR.
Idempotent: skips datasets already present with files on disk.

Usage:
    cd ml-engine && python scripts/download_kaggle.py owner/dataset-name [owner/dataset-name ...]
"""
import logging
import sys
from pathlib import Path

import kagglehub

ML_ENGINE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ML_ENGINE_DIR))

from config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _dataset_path(dataset_slug: str) -> Path:
    safe_name = dataset_slug.replace("/", "__")
    return settings.KAGGLE_CACHE_DIR / safe_name


def download(dataset_slug: str) -> dict:
    local_path = _dataset_path(dataset_slug)

    if local_path.exists() and any(local_path.iterdir()):
        logger.info(f"Already present: {dataset_slug} → {local_path}")
        return {"dataset": dataset_slug, "status": "already_exists", "local_path": str(local_path)}

    logger.info(f"Downloading: {dataset_slug}")
    local_path.mkdir(parents=True, exist_ok=True)

    cache_path = kagglehub.dataset_download(dataset_slug)
    logger.info(f"kagglehub cache: {cache_path}")

    file_count = sum(1 for f in Path(cache_path).rglob("*") if f.is_file())
    (local_path / ".kaggle_source").write_text(dataset_slug)
    (local_path / ".cache_path").write_text(str(cache_path))

    return {
        "dataset": dataset_slug,
        "status": "downloaded",
        "cache_path": str(cache_path),
        "local_path": str(local_path),
        "file_count": file_count,
    }


if __name__ == "__main__":
    if not sys.argv[1:]:
        print("Usage: python scripts/download_kaggle.py owner/dataset-name [owner/dataset-name ...]")
        sys.exit(1)

    for slug in sys.argv[1:]:
        try:
            result = download(slug)
            logger.info(f"  {result['status']}: {result['local_path']}")
        except Exception as exc:
            logger.error(f"  FAILED {slug}: {exc}", exc_info=True)
