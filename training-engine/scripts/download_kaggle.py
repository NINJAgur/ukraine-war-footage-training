"""
training-engine/scripts/download_kaggle.py

CLI: download Kaggle datasets into KAGGLE_CACHE_DIR.
Idempotent: skips datasets already present with files on disk.

Usage:
    cd training-engine && python scripts/download_kaggle.py owner/dataset-name [owner/dataset-name ...]
"""
import logging
import sys
from pathlib import Path

import kagglehub

TRAINING_ENGINE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TRAINING_ENGINE_DIR))

from config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _dataset_path(dataset_slug: str) -> Path:
    owner, name = dataset_slug.split("/", 1)
    return settings.KAGGLE_CACHE_DIR / "imported" / owner / name


def download(dataset_slug: str) -> dict:
    owner, name = dataset_slug.split("/", 1)
    versions_dir = _dataset_path(dataset_slug) / "versions"

    if versions_dir.exists() and any(versions_dir.iterdir()):
        logger.info(f"Already present: {dataset_slug} → {versions_dir.parent}")
        return {"dataset": dataset_slug, "status": "already_exists", "local_path": str(versions_dir.parent)}

    logger.info(f"Downloading: {dataset_slug}")
    cache_path = kagglehub.dataset_download(dataset_slug)
    logger.info(f"kagglehub cache: {cache_path}")

    # Mirror kagglehub structure under imported/
    import shutil as _shutil
    src = Path(cache_path)
    # cache_path is .../owner/name/versions/N — copy that version into our structure
    version_name = src.name
    dst = versions_dir / version_name
    dst.mkdir(parents=True, exist_ok=True)
    _shutil.copytree(str(src), str(dst), dirs_exist_ok=True)

    file_count = sum(1 for f in dst.rglob("*") if f.is_file())
    return {
        "dataset": dataset_slug,
        "status": "downloaded",
        "cache_path": str(cache_path),
        "local_path": str(versions_dir.parent),
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
