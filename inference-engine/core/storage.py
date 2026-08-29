"""
inference-engine/core/storage.py

Finalizes an annotated clip: renames temp file and either keeps it local or
uploads it to GCS (when STORAGE_MODE=remote).

All annotation paths (annotate_clips.py Celery task + manual pipeline scripts)
call finalize_clip() — single source of truth.
"""
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def finalize_clip(clip, temp_path: Path, model_name: str, base_name: str = None) -> str:
    """
    Rename temp annotated file to its permanent name and optionally upload to GCS.

    base_name overrides the name derived from temp_path. In remote mode the raw
    source is a random temp download, so without it the output inherits a name
    that cannot be traced back to its clip.

    Deleting the raw source is the CALLER's job, after it has committed — doing it
    here deletes the source before the DB knows the clip was annotated.

    Returns the final mp4_path (local path or GCS URL) to store on the Clip.
    """
    stem = base_name or temp_path.stem.removeprefix("temp_")
    perm_path = temp_path.parent / f"{stem}_annotated.mp4"
    shutil.move(str(temp_path), str(perm_path))

    from config import settings
    if settings.STORAGE_MODE == "remote":
        file_size = perm_path.stat().st_size
        url = _upload_gcs(perm_path, model_name, settings.REMOTE_STORAGE_BUCKET)
        clip.file_size_bytes = file_size
        return url

    clip.file_size_bytes = perm_path.stat().st_size
    return str(perm_path)


def _upload_gcs(local_path: Path, model_name: str, bucket_name: str) -> str:
    from google.cloud import storage as gcs
    client = gcs.Client()
    bucket = client.bucket(bucket_name)
    blob_name = f"annotated/{model_name.lower()}/{local_path.parent.name}/{local_path.name}"
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(str(local_path), content_type="video/mp4")
    local_path.unlink()
    logger.info(f"Uploaded to GCS: gs://{bucket_name}/{blob_name}")
    return f"https://storage.googleapis.com/{bucket_name}/{blob_name}"
