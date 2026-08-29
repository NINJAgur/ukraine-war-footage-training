"""
inference-engine/tasks/annotate_clips.py

Celery tasks: run specialist YOLO models on DB-scored clips.

annotate_clips dispatches one task per model (AIRCRAFT → VEHICLE → PERSONNEL →
GENERAL) rather than running them in a single task, so a slow model cannot
starve the ones behind it. Each processes up to ANNOTATE_BATCH_SIZE candidates,
validates detection rate, saves an annotated MP4, and commits before deleting
the raw source.
"""
import logging
from datetime import datetime, timezone
from pathlib import Path

from celery_app import celery_app
from config import settings
from db.models import Clip, ClipStatus
from db.session import get_session
from tasks.weights import _latest_weights

logger = logging.getLogger(__name__)

INFERENCE_ENGINE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = INFERENCE_ENGINE_DIR.parent

CONF_THRESH = 0.25
MIN_RATE = 0.10

# (score column, columns it must tie-break against) per specialist
SPECIALISTS = {
    "AIRCRAFT":  ("score_aircraft",  ["score_vehicle", "score_personnel"]),
    "VEHICLE":   ("score_vehicle",   ["score_aircraft", "score_personnel"]),
    "PERSONNEL": ("score_personnel", ["score_aircraft", "score_vehicle"]),
}
MODEL_ORDER = ["AIRCRAFT", "VEHICLE", "PERSONNEL", "GENERAL"]

_CLASS_KEY = {"AIRCRAFT": "aircraft", "VEHICLE": "vehicle", "PERSONNEL": "personnel"}

def _detection_counts(model_name: str, det_counts: dict) -> dict:
    """Build canonical detection_counts dict from infer_video_multi_model output."""
    counts = {"aircraft": 0, "vehicle": 0, "personnel": 0}
    for k, v in det_counts.items():
        canon = _CLASS_KEY.get(k.upper())
        if canon:
            counts[canon] += v
    counts["total"] = sum(counts[c] for c in ("aircraft", "vehicle", "personnel"))
    return counts


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
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as fh:
        tmp = Path(fh.name)
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


def _reject(session, clip, raw_path: Path) -> None:
    """Mark a clip REJECTED, persist it, then drop its raw source."""
    original = clip.file_path
    clip.file_path = None
    clip.status = ClipStatus.REJECTED
    session.commit()
    _cleanup_raw(raw_path, original)


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
        candidates = q.limit(settings.ANNOTATE_BATCH_SIZE).all()
        total = len(candidates)
        logger.info(
            f"[{model_name}] {total} candidates  weights={weights.name}"
        )

        for clip in candidates:
            title = clip.title or f"clip_{clip.id}"
            logger.info(
                f"[{model_name}] clip_id={clip.id}  "
                f"aircraft={clip.score_aircraft:.2f}  vehicle={clip.score_vehicle:.2f}  personnel={clip.score_personnel:.2f}\n"
                f"    title: {title}"
            )
            try:
                raw_path = _resolve_clip_path(clip.file_path)
            except Exception as exc:
                logger.warning(f"[{model_name}]   -> ERROR: failed to resolve raw file: {exc}")
                clip.status = ClipStatus.ERROR
                session.commit()
                errors += 1
                continue

            if not raw_path.exists():
                logger.warning(f"[{model_name}]   -> ERROR: file missing: {raw_path}")
                clip.status = ClipStatus.ERROR
                session.commit()
                errors += 1
                continue

            passed, rate = validate_clip(model, raw_path, conf_thresh=CONF_THRESH, min_rate=MIN_RATE)
            if not passed:
                logger.info(f"[{model_name}]   -> REJECT: validate rate={rate:.0%} < {MIN_RATE:.0%}")
                _reject(session, clip, raw_path)
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
                _reject(session, clip, raw_path)
                rejected += 1
                continue

            clip.mp4_path = finalize_clip(clip, temp_out, model_name, base_name=clip.url_hash[:8])
            clip.det_class = model_name
            clip.detection_counts = _detection_counts(model_name, det_counts)
            clip.status = ClipStatus.ANNOTATED
            clip.updated_at = datetime.now(timezone.utc)
            # Commit before deleting the raw — a crash between the two must not
            # leave the DB pointing at a source that no longer exists.
            original = clip.file_path
            clip.file_path = None
            session.commit()
            _cleanup_raw(raw_path, original)
            accepted += 1
            logger.info(
                f"[{model_name}]   -> ANNOTATED: dets={clip_dets}  "
                f"file={Path(clip.mp4_path).name}"
            )

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
            .limit(settings.ANNOTATE_BATCH_SIZE)
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
                session.commit()
                errors += 1
                continue

            passed, rate = validate_clip(model, raw_path, conf_thresh=CONF_THRESH, min_rate=MIN_RATE)
            if not passed:
                logger.info(f"[GENERAL]   -> REJECT: validate rate={rate:.0%} < {MIN_RATE:.0%}")
                _reject(session, clip, raw_path)
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
                logger.info("[GENERAL]   -> REJECT: zero detections in full inference pass")
                if temp_out.exists():
                    temp_out.unlink()
                _reject(session, clip, raw_path)
                rejected += 1
                continue

            clip.mp4_path = finalize_clip(clip, temp_out, "GENERAL", base_name=clip.url_hash[:8])
            clip.det_class = "GENERAL"
            clip.detection_counts = _detection_counts("GENERAL", det_counts)
            clip.status = ClipStatus.ANNOTATED
            clip.updated_at = datetime.now(timezone.utc)
            original = clip.file_path
            clip.file_path = None
            session.commit()
            _cleanup_raw(raw_path, original)
            accepted += 1
            logger.info(
                f"[GENERAL]   -> ANNOTATED: dets={clip_dets}  "
                f"file={Path(clip.mp4_path).name}"
            )

    return {"accepted": accepted, "rejected": rejected, "errors": errors, "total": total}


def _shutdown_if_no_training() -> None:
    """Shut down this VM after annotation completes. No-op on Windows.
    Training runs are handled by training-engine (Q=training) — this VM has no training worker.
    """
    import sys
    import subprocess
    if sys.platform == "win32":
        return
    logger.info("[shutdown] Annotation complete — shutting down VM")
    subprocess.run(["sudo", "shutdown", "-h", "now"])


def _cleanup_zero_score_clips() -> None:
    """Drop raw video files for DOWNLOADED clips that have all-zero scores."""
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
            # Never resolve the path here — for a gs:// clip that downloads the
            # object just to unlink the copy, leaving the original orphaned.
            original = clip.file_path
            if original.startswith("gs://"):
                _delete_gcs_object(original)
            else:
                local = _resolve_clip_path(original)
                if local.exists():
                    local.unlink()
            clip.file_path = None
            clip.status = ClipStatus.REJECTED
            deleted += 1
        session.commit()
    if deleted:
        logger.info(f"[cleanup] Deleted {deleted} zero-score DOWNLOADED clip files")


@celery_app.task(
    bind=True,
    name="tasks.annotate_clips.annotate_model",
    queue="pipeline",
    max_retries=0,
)
def annotate_model(self, model_name: str) -> dict:
    """Annotate one model's candidates. One task per model, so a slow or failing
    model cannot consume the time budget of the ones queued behind it."""
    logger.info(f"[{self.request.id}] annotate_model {model_name} started")
    if model_name == "GENERAL":
        result = _run_general()
    else:
        score_col, tie_cols = SPECIALISTS[model_name]
        result = _run_specialist(model_name, score_col, tie_cols)
    logger.info(f"[{self.request.id}] annotate_model {model_name} done: {result}")
    return result


@celery_app.task(
    bind=True,
    name="tasks.annotate_clips.finish_annotation",
    queue="pipeline",
    max_retries=0,
)
def finish_annotation(self) -> dict:
    """Runs once after every model has had its turn."""
    _cleanup_zero_score_clips()
    _shutdown_if_no_training()
    return {"status": "done"}


@celery_app.task(
    bind=True,
    name="tasks.annotate_clips.annotate_clips",
    queue="pipeline",
    max_retries=0,
)
def annotate_clips(self) -> dict:
    """Dispatch one annotate_model task per model, then finish_annotation."""
    from celery import chain as _chain

    logger.info(f"[{self.request.id}] annotate_clips dispatching {MODEL_ORDER}")
    _chain(
        *[annotate_model.si(name) for name in MODEL_ORDER],
        finish_annotation.si(),
    ).apply_async(queue="pipeline")
    return {"dispatched": MODEL_ORDER}
