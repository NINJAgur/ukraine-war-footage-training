"""
scraper-engine/utils/gcs.py

Upload raw downloaded videos to GCS when STORAGE_MODE=remote.
Returns gs:// URL stored as clip.file_path for the ML worker to pick up.
"""
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def upload_raw(local_path: Path, source: str, bucket_name: str) -> str:
    """Upload raw video to GCS, delete local copy, return gs:// URL."""
    from google.cloud import storage as gcs
    client = gcs.Client()
    bucket = client.bucket(bucket_name)
    blob_name = f"raw/{source}/{local_path.parent.name}/{local_path.name}"
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(str(local_path), content_type="video/mp4")
    local_path.unlink()
    url = f"gs://{bucket_name}/{blob_name}"
    logger.info(f"Uploaded raw to GCS: {url}")
    return url
