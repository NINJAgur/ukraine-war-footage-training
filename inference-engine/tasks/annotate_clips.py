"""
inference-engine/tasks/annotate_clips.py

Celery task: run specialist YOLO models on DB-scored clips.
Sequential: AIRCRAFT → VEHICLE → PERSONNEL.
Each specialist processes up to BATCH_SIZE candidates, validates detection rate,
saves annotated MP4 to ANNOTATED_VIDEO_DIR, deletes raw file, updates DB.
"""
import logging
from datetime import datetime, timezone
from pathlib import Path

from celery_app import celery_app
from config import settings
from db.models import Clip, ClipStatus, TrainingRun, TrainingStatus
from db.session import get_session
from tasks.weights import _latest_weights

logger = logging.getLogger(__name__)

INFERENCE_ENGINE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = INFERENCE_ENGINE_DIR.parent

CONF_THRESH = 0.25
MIN_RATE = 0.10
BATCH_SIZE = 10


def _delete_gcs_object(gs_url: str) -> None:
    """Delete a gs:// object from GCS."""
    try:
        from google.cloud import storage as gcs
        without_scheme = gs_url[len("gs://"):]
        bucket_name, _, blob_name = without_scheme.partition("/")
        client = gcs.Client()
        client.bucket(bucket_name).blob(blob_name).delete()
        logger.info(f"Deleted GCS object: {gs_url}")
    except Exception as exc:
        logger.warning(f"Failed to delete GCS object {gs_url}: {exc}")


def _download_from_gcs(gs_url: str) -> Path:
    """Download a gs:// object to a temp file and return its path."""
    import tempfile
    from google.cloud import storage as gcs
    without_scheme = gs_url[len("gs://"):]
    bucket_name, _, blob_name = without_scheme.partition("/")
    suffix = Path(blob_name).suffix or ".mp4"
    tmp = Path(tempfile.mktemp(suffix=suffix, dir="/tmp"))
    client = gcs.Client()
    client.bucket(bucket_name).blob(blob_name).download_to_filename(str(tmp))
    logger.info(f"Downloaded from GCS: {gs_url} → {tmp}")
    return tmp


def _resolve_clip_path(raw: str) -> Path:
    if raw.startswith("gs://"):
        return _download_from_gcs(raw)
    p = Path(raw)
    if p.exists():
        return p
    # Docker container writes /app/scraper-engine/media/... → map to repo root
    normalized = raw.replace("\\", "/")
    marker = "scraper-engine/media/"
    if marker in normalized:
        rel = normalized[normalized.index(marker):]
        return PROJECT_DIR / rel
    return p


from core.storage import finalize_clip


def _cleanup_raw(raw_path: Path, original_file_path: str) -> None:
    """Delete local (temp) file and, when original was a GCS URL, the GCS object."""
    if raw_path.exists():
        raw_path.unlink()
    if original_file_path and original_file_path.startswith("gs://"):
        _delete_gcs_object(original_file_path)


def _run_specialist(
    model_name: str,
    score_col: str,
    tie_cols: list,
) -> dict:
    from ultralytics import YOLO
    from core.inference import validate_clip, infer_video_multi_model

    try:
        weights = _latest_weights(model_name)
        model = YOLO(str(weights))
    except FileNotFoundError as exc:
        logger.warning(f"[{model_name}] No weights — skipping: {exc}")
        return {"skipped": True}

    color = settings.MODEL_COLORS[model_name]
    accepted = rejected = errors = 0

    with get_session() as session:
        score_attr = getattr(Clip, score_col)
        q = (
            session.query(Clip)
            .filter(Clip.status == ClipStatus.DOWNLOADED)
            .filter(Clip.file_path.isnot(None))
            .filter(score_attr > 0)
        )
        for col in tie_cols:
            q = q.filter(score_attr >= getattr(Clip, col))
        candidates = q.limit(BATCH_SIZE).all()
        total = len(candidates)
        logger.info(
            f"[{model_name}] {total} candidates  weights={weights.name}"
        )

        for clip in candidates:
            raw_path = _resolve_clip_path(clip.file_path)
            title = clip.title or f"clip_{clip.id}"
            logger.info(
                f"[{model_name}] clip_id={clip.id}  "
                f"aircraft={clip.score_aircraft:.2f}  vehicle={clip.score_vehicle:.2f}  personnel={clip.score_personnel:.2f}\n"
                f"    title: {title}"
            )

            if not raw_path.exists():
                logger.warning(f"[{model_name}]   -> ERROR: file missing: {raw_path}")
                clip.status = ClipStatus.ERROR
                errors += 1
                continue

            passed, rate = validate_clip(model, raw_path, conf_thresh=CONF_THRESH, min_rate=MIN_RATE)
            if not passed:
                logger.info(f"[{model_name}]   -> REJECT: validate rate={rate:.0%} < {MIN_RATE:.0%}")
                _cleanup_raw(raw_path, clip.file_path)
                clip.file_path = None
                clip.status = ClipStatus.PENDING
                rejected += 1
                continue

            date_str = (clip.published_at or datetime.now(timezone.utc)).strftime("%Y-%m-%d")
            out_dir = settings.ANNOTATED_VIDEO_DIR / model_name.lower() / date_str
            out_dir.mkdir(parents=True, exist_ok=True)
            temp_out = out_dir / f"temp_{raw_path.name}"
            _, det_counts = infer_video_multi_model(
                [(model, model_name, color)], str(raw_path),
                save_path=str(temp_out), no_display=True, conf_thresh=CONF_THRESH,
            )
            clip_dets = sum(det_counts.values())

            if clip_dets == 0:
                logger.info(f"[{model_name}]   -> REJECT: zero detections in full inference pass")
                if temp_out.exists():
                    temp_out.unlink()
                _cleanup_raw(raw_path, clip.file_path)
                clip.file_path = None
                clip.status = ClipStatus.PENDING
                rejected += 1
                continue

            clip.mp4_path = finalize_clip(clip, temp_out, model_name)
            clip.det_class = model_name
            clip.status = ClipStatus.ANNOTATED
            clip.updated_at = datetime.now(timezone.utc)
            if raw_path.exists():
                raw_path.unlink()
            clip.file_path = None
            accepted += 1
            logger.info(
                f"[{model_name}]   -> ANNOTATED: dets={clip_dets}  "
                f"file={Path(clip.mp4_path).name}"
            )

        session.commit()

    return {"accepted": accepted, "rejected": rejected, "errors": errors, "total": total}


def _run_general() -> dict:
    """Pick up remaining DOWNLOADED clips that no specialist consumed."""
    from sqlalchemy import or_
    from ultralytics import YOLO
    from core.inference import validate_clip, infer_video_multi_model

    try:
        weights = _latest_weights("GENERAL")
        model = YOLO(str(weights))
    except FileNotFoundError as exc:
        logger.warning(f"[GENERAL] No weights — skipping: {exc}")
        return {"skipped": True}

    color = settings.MODEL_COLORS["GENERAL"]
    accepted = rejected = errors = 0

    with get_session() as session:
        candidates = (
            session.query(Clip)
            .filter(Clip.status == ClipStatus.DOWNLOADED)
            .filter(Clip.file_path.isnot(None))
            .filter(or_(
                Clip.score_aircraft > 0,
                Clip.score_vehicle > 0,
                Clip.score_personnel > 0,
                Clip.score_uas > 0,
            ))
            .limit(BATCH_SIZE)
            .all()
        )
        total = len(candidates)
        logger.info(
            f"[GENERAL] {total} candidates (leftovers from specialists)  weights={weights.name}"
        )

        for clip in candidates:
            raw_path = _resolve_clip_path(clip.file_path)
            title = clip.title or f"clip_{clip.id}"
            logger.info(
                f"[GENERAL] clip_id={clip.id}  "
                f"aircraft={clip.score_aircraft:.2f}  vehicle={clip.score_vehicle:.2f}  personnel={clip.score_personnel:.2f}\n"
                f"    title: {title}"
            )

            if not raw_path.exists():
                logger.warning(f"[GENERAL]   -> ERROR: file missing: {raw_path}")
                clip.status = ClipStatus.ERROR
                errors += 1
                continue

            passed, rate = validate_clip(model, raw_path, conf_thresh=CONF_THRESH, min_rate=MIN_RATE)
            if not passed:
                logger.info(f"[GENERAL]   -> REJECT: validate rate={rate:.0%} < {MIN_RATE:.0%}")
                _cleanup_raw(raw_path, clip.file_path)
                clip.file_path = None
                clip.status = ClipStatus.PENDING
                rejected += 1
                continue

            date_str = (clip.published_at or datetime.now(timezone.utc)).strftime("%Y-%m-%d")
            out_dir = settings.ANNOTATED_VIDEO_DIR / "general" / date_str
            out_dir.mkdir(parents=True, exist_ok=True)
            temp_out = out_dir / f"temp_{raw_path.name}"
            _, det_counts = infer_video_multi_model(
                [(model, "GENERAL", color)], str(raw_path),
                save_path=str(temp_out), no_display=True, conf_thresh=CONF_THRESH,
            )
            clip_dets = sum(det_counts.values())

            if clip_dets == 0:
                logger.info(f"[GENERAL]   -> REJECT: zero detections in full inference pass")
                if temp_out.exists():
                    temp_out.unlink()
                _cleanup_raw(raw_path, clip.file_path)
                clip.file_path = None
                clip.status = ClipStatus.PENDING
                rejected += 1
                continue

            clip.mp4_path = finalize_clip(clip, temp_out, "GENERAL")
            clip.det_class = "GENERAL"
            clip.status = ClipStatus.ANNOTATED
            clip.updated_at = datetime.now(timezone.utc)
            if raw_path.exists():
                raw_path.unlink()
            clip.file_path = None
            accepted += 1
            logger.info(
                f"[GENERAL]   -> ANNOTATED: dets={clip_dets}  "
                f"file={Path(clip.mp4_path).name}"
            )

        session.commit()

    return {"accepted": accepted, "rejected": rejected, "errors": errors, "total": total}


def _shutdown_if_no_training() -> None:
    """Shut down this VM if no training run is QUEUED or RUNNING. No-op on Windows."""
    import sys
    import subprocess
    if sys.platform == "win32":
        return
    with get_session() as session:
        active = session.query(TrainingRun).filter(
            TrainingRun.status.in_([TrainingStatus.QUEUED, TrainingStatus.RUNNING])
        ).first()
    if active:
        logger.info("[shutdown] Training run active — keeping VM alive")
        return
    logger.info("[shutdown] No active training — shutting down VM")
    subprocess.run(["sudo", "shutdown", "-h", "now"])


def _cleanup_zero_score_clips() -> None:
    """Delete raw video files for DOWNLOADED clips that have all-zero scores."""
    deleted = 0
    with get_session() as session:
        clips = (
            session.query(Clip)
            .filter(
                Clip.status == ClipStatus.DOWNLOADED,
                Clip.file_path.isnot(None),
                Clip.score_aircraft == 0,
                Clip.score_vehicle == 0,
                Clip.score_personnel == 0,
                Clip.score_uas == 0,
            )
            .all()
        )
        for clip in clips:
            raw_path = _resolve_clip_path(clip.file_path)
            if raw_path.exists():
                raw_path.unlink()
            clip.file_path = None
            clip.status = ClipStatus.PENDING
            deleted += 1
        session.commit()
    if deleted:
        logger.info(f"[cleanup] Deleted {deleted} zero-score DOWNLOADED clip files")


@celery_app.task(
    bind=True,
    name="tasks.annotate_clips.annotate_clips",
    queue="pipeline",
    max_retries=0,
)
def annotate_clips(self) -> dict:
    """
    Sequential annotation pipeline: AIRCRAFT → VEHICLE → PERSONNEL → GENERAL.
    Each specialist loads its own weights and processes up to BATCH_SIZE candidates.
    Raw files deleted after annotation or rejection.
    """
    logger.info(f"[{self.request.id}] annotate_clips started")

    specialists = [
        ("AIRCRAFT",  "score_aircraft",  ["score_vehicle", "score_personnel"]),
        ("VEHICLE",   "score_vehicle",   ["score_aircraft", "score_personnel"]),
        ("PERSONNEL", "score_personnel", ["score_aircraft", "score_vehicle"]),
    ]

    results = {name: _run_specialist(name, col, ties) for name, col, ties in specialists}
    results["GENERAL"] = _run_general()

    logger.info(f"[{self.request.id}] annotate_clips done: {results}")

    _cleanup_zero_score_clips()
    _shutdown_if_no_training()

    return results
