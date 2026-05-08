"""
ml-engine/tasks/annotate_clips.py

Celery task: run specialist YOLO models on DB-scored clips.
Sequential: AIRCRAFT → VEHICLE → PERSONNEL.
Each specialist processes up to BATCH_SIZE candidates, validates detection rate,
saves annotated MP4 to ANNOTATED_VIDEO_DIR, deletes raw file, updates DB.
"""
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from celery_app import celery_app
from config import settings
from db.models import (
    Clip, ClipStatus,
    Dataset, DatasetStatus,
    ModelType, TrainingRun, TrainingStage, TrainingStatus,
)
from db.session import get_session

logger = logging.getLogger(__name__)

ML_ENGINE_DIR = Path(__file__).resolve().parents[1]

CONF_THRESH = 0.15
MIN_RATE = 0.10
BATCH_SIZE = 10
FINETUNE_MIN_DATASETS = 5


def _latest_weights(model_name: str) -> Path:
    runs_dir = ML_ENGINE_DIR / "runs/baseline" / model_name
    if not runs_dir.exists():
        raise FileNotFoundError(f"No runs directory: {runs_dir}")
    candidates = sorted(
        (d for d in runs_dir.iterdir() if d.is_dir()),
        key=lambda d: int(d.name.rsplit("_", 1)[-1]) if d.name.rsplit("_", 1)[-1].isdigit() else 0,
        reverse=True,
    )
    for run_dir in candidates:
        w = run_dir / "weights" / "best.pt"
        if w.exists():
            return w
    raise FileNotFoundError(f"No best.pt found in {runs_dir}")


def _finalize(clip: Clip, temp_path: Path) -> str:
    clean_name = temp_path.stem.removeprefix("temp_").replace("_clip", "") + "_annotated.mp4"
    perm_path = temp_path.parent / clean_name
    shutil.move(str(temp_path), str(perm_path))
    if clip.file_path:
        if os.path.exists(clip.file_path):
            try:
                os.remove(clip.file_path)
            except PermissionError:
                logger.warning(f"Could not delete raw file (Windows lock): {clip.file_path}")
        clip.file_path = None
    return str(perm_path)


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
        logger.info(f"[{model_name}] {total} candidates")

        for clip in candidates:
            raw_path = Path(clip.file_path)

            if not raw_path.exists():
                logger.warning(f"[{model_name}] clip_id={clip.id} file missing: {raw_path}")
                clip.status = ClipStatus.ERROR
                errors += 1
                continue

            passed, rate = validate_clip(model, raw_path, conf_thresh=CONF_THRESH, min_rate=MIN_RATE)
            if not passed:
                logger.info(f"[{model_name}] clip_id={clip.id} REJECT rate={rate:.0%}")
                if raw_path.exists():
                    raw_path.unlink()
                    clip.file_path = None
                clip.status = ClipStatus.PENDING
                rejected += 1
                continue

            date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            out_dir = settings.ANNOTATED_VIDEO_DIR / model_name.lower() / date_str
            out_dir.mkdir(parents=True, exist_ok=True)
            temp_out = out_dir / f"temp_{raw_path.name}"
            _, det_counts = infer_video_multi_model(
                [(model, model_name, color)], str(raw_path),
                save_path=str(temp_out), no_display=True, conf_thresh=CONF_THRESH,
            )
            clip_dets = sum(det_counts.values())

            if clip_dets == 0:
                logger.info(f"[{model_name}] clip_id={clip.id} REJECT zero detections in full pass")
                if temp_out.exists():
                    temp_out.unlink()
                if raw_path.exists():
                    raw_path.unlink()
                    clip.file_path = None
                clip.status = ClipStatus.PENDING
                rejected += 1
                continue

            clip.mp4_path = _finalize(clip, temp_out)
            clip.det_class = model_name
            clip.status = ClipStatus.ANNOTATED
            clip.updated_at = datetime.now(timezone.utc)
            accepted += 1
            logger.info(f"[{model_name}] clip_id={clip.id} ANNOTATED dets={clip_dets}")

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
        logger.info(f"[GENERAL] {total} candidates (leftovers from specialists)")

        for clip in candidates:
            raw_path = Path(clip.file_path)

            if not raw_path.exists():
                logger.warning(f"[GENERAL] clip_id={clip.id} file missing: {raw_path}")
                clip.status = ClipStatus.ERROR
                errors += 1
                continue

            passed, rate = validate_clip(model, raw_path, conf_thresh=CONF_THRESH, min_rate=MIN_RATE)
            if not passed:
                logger.info(f"[GENERAL] clip_id={clip.id} REJECT rate={rate:.0%}")
                if raw_path.exists():
                    raw_path.unlink()
                    clip.file_path = None
                clip.status = ClipStatus.PENDING
                rejected += 1
                continue

            date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            out_dir = settings.ANNOTATED_VIDEO_DIR / "general" / date_str
            out_dir.mkdir(parents=True, exist_ok=True)
            temp_out = out_dir / f"temp_{raw_path.name}"
            _, det_counts = infer_video_multi_model(
                [(model, "GENERAL", color)], str(raw_path),
                save_path=str(temp_out), no_display=True, conf_thresh=CONF_THRESH,
            )
            clip_dets = sum(det_counts.values())

            if clip_dets == 0:
                logger.info(f"[GENERAL] clip_id={clip.id} REJECT zero detections in full pass")
                if temp_out.exists():
                    temp_out.unlink()
                if raw_path.exists():
                    raw_path.unlink()
                    clip.file_path = None
                clip.status = ClipStatus.PENDING
                rejected += 1
                continue

            clip.mp4_path = _finalize(clip, temp_out)
            clip.det_class = "GENERAL"
            clip.status = ClipStatus.ANNOTATED
            clip.updated_at = datetime.now(timezone.utc)
            accepted += 1
            logger.info(f"[GENERAL] clip_id={clip.id} ANNOTATED dets={clip_dets}")

        session.commit()

    return {"accepted": accepted, "rejected": rejected, "errors": errors, "total": total}


def _maybe_trigger_finetune() -> None:
    """Dispatch a GENERAL fine-tune run when enough PACKAGED datasets have accumulated."""
    with get_session() as session:
        active = (
            session.query(TrainingRun)
            .filter(TrainingRun.stage == TrainingStage.FINETUNE)
            .filter(TrainingRun.model_type == ModelType.GENERAL)
            .filter(TrainingRun.status.in_([TrainingStatus.QUEUED, TrainingStatus.RUNNING]))
            .first()
        )
        if active:
            logger.info(f"[finetune] GENERAL finetune already active (run_id={active.id}) — skipping")
            return

        packaged = (
            session.query(Dataset)
            .filter(Dataset.status == DatasetStatus.PACKAGED)
            .all()
        )
        if len(packaged) < FINETUNE_MIN_DATASETS:
            logger.info(f"[finetune] {len(packaged)} PACKAGED datasets — need {FINETUNE_MIN_DATASETS} to trigger")
            return

        dataset_ids = [d.id for d in packaged]
        try:
            baseline_weights = str(_latest_weights("GENERAL"))
        except FileNotFoundError:
            baseline_weights = None
            logger.warning("[finetune] No GENERAL baseline weights found — will use yolov8m.pt")

        run = TrainingRun(
            stage=TrainingStage.FINETUNE,
            model_type=ModelType.GENERAL,
            status=TrainingStatus.QUEUED,
            dataset_ids=dataset_ids,
            baseline_weights=baseline_weights,
        )
        session.add(run)
        session.flush()
        run_id = run.id

    from tasks.train_finetune import train_finetune
    train_finetune.delay(training_run_id=run_id)
    logger.info(f"[finetune] Dispatched GENERAL finetune run_id={run_id} with {len(dataset_ids)} PACKAGED datasets")


@celery_app.task(
    bind=True,
    name="tasks.annotate_clips.annotate_clips",
    queue="gpu",
    max_retries=0,
)
def annotate_clips(self) -> dict:
    """
    Sequential annotation pipeline: AIRCRAFT → VEHICLE → PERSONNEL.
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

    _maybe_trigger_finetune()

    return results
