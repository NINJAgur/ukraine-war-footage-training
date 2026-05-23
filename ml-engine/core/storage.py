"""
ml-engine/core/storage.py

Finalizes an annotated clip: renames temp file, deletes raw source, and
either keeps the file local or uploads it to GCS (when STORAGE_MODE=remote).

All annotation paths (annotate_clips.py Celery task + manual pipeline scripts)
call finalize_clip() — single source of truth.
"""
import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def _delete_gcs_object(gs_url: str, bucket_name: str) -> None:
    try:
        from google.cloud import storage as gcs
        without_scheme = gs_url[len("gs://"):]
        _, _, blob_name = without_scheme.partition("/")
        gcs.Client().bucket(bucket_name).blob(blob_name).delete()
        logger.info(f"Deleted GCS raw: {gs_url}")
    except Exception as exc:
        logger.warning(f"Failed to delete GCS object {gs_url}: {exc}")


def finalize_clip(clip, temp_path: Path, model_name: str) -> str:
    """
    Rename temp annotated file to its permanent name, delete the raw source,
    and optionally upload to GCS.

    Returns the final mp4_path (local path or GCS URL) to store on the Clip.
    """
    clean_name = temp_path.stem.removeprefix("temp_") + "_annotated.mp4"
    perm_path = temp_path.parent / clean_name
    shutil.move(str(temp_path), str(perm_path))

    if clip.file_path:
        if clip.file_path.startswith("gs://"):
            from config import settings as _s
            if _s.STORAGE_MODE == "remote":
                _delete_gcs_object(clip.file_path, _s.REMOTE_STORAGE_BUCKET)
        elif os.path.exists(clip.file_path):
            try:
                os.remove(clip.file_path)
            except PermissionError:
                logger.warning(f"Could not delete raw file (file lock): {clip.file_path}")
    clip.file_path = None

    from config import settings
    if settings.STORAGE_MODE == "remote":
        return _upload_gcs(perm_path, model_name, settings.REMOTE_STORAGE_BUCKET)

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
