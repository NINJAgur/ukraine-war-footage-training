"""
ml-engine/tasks/annotate_clips.py

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
FINETUNE_MAX_CYCLES = settings.YOLO_FINETUNE_MAX_CYCLES


def _best_map50(metrics: dict) -> float:
    if not metrics:
        return 0.0
    for key in ("mAP50(B)", "metrics/mAP50(B)", "mAP50"):
        if key in metrics:
            return float(metrics[key])
    return 0.0


def _resolve_weights_path(raw: str) -> Path:
    """Handle Windows absolute paths stored in DB when running inside a Linux container."""
    w = Path(raw)
    if w.exists():
        return w
    # Windows path (D:\...\ml-engine\runs\...) inside Linux container:
    # bind-mount maps ml-engine/runs → /app/ml-engine/runs
    normalized = raw.replace("\\", "/")
    marker = "ml-engine/runs/"
    if marker in normalized:
        rel = normalized[normalized.index(marker) + len("ml-engine/"):]
        return ML_ENGINE_DIR / rel
    return w


def _latest_weights(model_name: str) -> Path:
    with get_session() as session:
        runs = (
            session.query(TrainingRun)
            .filter(
                TrainingRun.model_type == ModelType[model_name],
                TrainingRun.status == TrainingStatus.DONE,
                TrainingRun.weights_path.isnot(None),
            )
            .all()
        )
    runs_by_map = sorted(runs, key=lambda r: _best_map50(r.metrics), reverse=True)
    for run in runs_by_map:
        w = _resolve_weights_path(run.weights_path)
        if w.exists():
            return w
    raise FileNotFoundError(f"No usable weights for {model_name} in DB")


from core.storage import finalize_clip


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

            clip.mp4_path = finalize_clip(clip, temp_out, model_name)
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

            clip.mp4_path = finalize_clip(clip, temp_out, "GENERAL")
            clip.det_class = "GENERAL"
            clip.status = ClipStatus.ANNOTATED
            clip.updated_at = datetime.now(timezone.utc)
            accepted += 1
            logger.info(f"[GENERAL] clip_id={clip.id} ANNOTATED dets={clip_dets}")

        session.commit()

    return {"accepted": accepted, "rejected": rejected, "errors": errors, "total": total}


def _maybe_trigger_finetune() -> None:
    """Dispatch fine-tune runs for any model type with enough PACKAGED datasets."""
    for model_type in [ModelType.AIRCRAFT, ModelType.VEHICLE, ModelType.PERSONNEL, ModelType.GENERAL]:
        _trigger_model_finetune(model_type)


def _trigger_model_finetune(model_type: ModelType) -> None:
    with get_session() as session:
        done_cycles = (
            session.query(TrainingRun)
            .filter(TrainingRun.stage == TrainingStage.FINETUNE)
            .filter(TrainingRun.model_type == model_type)
            .filter(TrainingRun.status == TrainingStatus.DONE)
            .count()
        )
        if done_cycles >= FINETUNE_MAX_CYCLES:
            logger.info(
                f"[finetune] {model_type.value}: max cycles reached "
                f"({done_cycles}/{FINETUNE_MAX_CYCLES}) — 50 total epochs complete"
            )
            return

        active = (
            session.query(TrainingRun)
            .filter(TrainingRun.stage == TrainingStage.FINETUNE)
            .filter(TrainingRun.model_type == model_type)
            .filter(TrainingRun.status.in_([TrainingStatus.QUEUED, TrainingStatus.RUNNING]))
            .first()
        )
        if active:
            logger.info(f"[finetune] {model_type.value} already active (run_id={active.id})")
            return

        all_packaged = (
            session.query(Dataset)
            .filter(Dataset.status == DatasetStatus.PACKAGED)
            .all()
        )

        if model_type == ModelType.GENERAL:
            relevant = all_packaged
        else:
            relevant = [
                d for d in all_packaged
                if model_type.value in (d.detected_model_types or [])
            ]

        if len(relevant) < FINETUNE_MIN_DATASETS:
            logger.info(f"[finetune] {model_type.value}: {len(relevant)}/{FINETUNE_MIN_DATASETS} PACKAGED datasets")
            return

        dataset_ids = [d.id for d in relevant]
        try:
            baseline_weights = str(_latest_weights(model_type.value))
        except FileNotFoundError:
            baseline_weights = None
            logger.warning(f"[finetune] No {model_type.value} baseline weights — will use yolov8m.pt")

        run = TrainingRun(
            stage=TrainingStage.FINETUNE,
            model_type=model_type,
            status=TrainingStatus.QUEUED,
            dataset_ids=dataset_ids,
            baseline_weights=baseline_weights,
        )
        session.add(run)
        session.flush()
        run_id = run.id

    from tasks.train_finetune import train_finetune
    train_finetune.delay(training_run_id=run_id)
    logger.info(f"[finetune] Dispatched {model_type.value} run_id={run_id} with {len(dataset_ids)} datasets")


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
