"""
ml-engine/tasks/render_annotated.py

Celery task: run all available YOLO models on a downloaded Clip video,
produce a single annotated H.264 MP4 with colour-coded detections per
model type, update Clip.mp4_path + status → ANNOTATED.

Weights priority per model type:
  1. Latest FINETUNE TrainingRun (DONE) for that model_type
  2. Latest BASELINE TrainingRun (DONE) for that model_type
  3. settings.YOLO_MODEL (yolov8m.pt pretrained COCO fallback)

Models with no trained weights use the pretrained fallback, so the
pipeline is functional from day one before any training has run.
"""
import logging
import sys
from pathlib import Path

from celery_app import celery_app
from config import settings
from db.models import Clip, ClipStatus, ModelType, TrainingRun, TrainingStage, TrainingStatus
from db.session import get_session

sys.path.insert(0, str(Path(__file__).parent.parent / "core"))

logger = logging.getLogger(__name__)


def _best_weights_per_model() -> dict[str, str]:
    """
    Return {model_type_value: weights_path} for every ModelType.
    Falls back to settings.YOLO_MODEL when no trained weights exist.
    """
    result: dict[str, str] = {}
    with get_session() as session:
        for mt in ModelType:
            weights = None
            for stage in [TrainingStage.FINETUNE, TrainingStage.BASELINE]:
                run = (
                    session.query(TrainingRun)
                    .filter(
                        TrainingRun.model_type == mt,
                        TrainingRun.stage == stage,
                        TrainingRun.status == TrainingStatus.DONE,
                        TrainingRun.weights_path.isnot(None),
                    )
                    .order_by(TrainingRun.completed_at.desc())
                    .first()
                )
                if run and Path(run.weights_path).exists():
                    weights = run.weights_path
                    logger.info(f"[{mt.value}] Using {stage.value} weights: {weights}")
                    break
            if weights is None:
                weights = settings.YOLO_MODEL
                logger.info(f"[{mt.value}] No trained weights — using pretrained {weights}")
            result[mt.value] = weights
    return result


@celery_app.task(
    bind=True,
    name="tasks.render_annotated.render_annotated_clip",
    queue="gpu",
    autoretry_for=(Exception,),
    max_retries=2,
    default_retry_delay=120,
)
def render_annotated_clip(self, clip_id: int) -> dict:
    """
    Run all 4 YOLO models on a Clip's raw video and save a single annotated
    MP4 with colour-coded bboxes per model type.
    Idempotent: skips if annotated file already on disk.
    """
    logger.info(f"[{self.request.id}] render_annotated_clip clip_id={clip_id}")

    with get_session() as session:
        clip = session.get(Clip, clip_id)
        if clip is None:
            raise ValueError(f"Clip {clip_id} not found")
        if not clip.file_path or not Path(clip.file_path).exists():
            raise ValueError(f"Clip {clip_id} raw video not on disk: {clip.file_path}")
        video_path = Path(clip.file_path)
        clip_hash  = clip.url_hash[:12]
        if clip.mp4_path and Path(clip.mp4_path).exists():
            logger.info(f"[{self.request.id}] Already annotated: {clip.mp4_path}")
            return {"status": "skipped", "clip_id": clip_id}

    output_path = settings.ANNOTATED_VIDEO_DIR / f"{clip_hash}_annotated.mp4"
    weights_map = _best_weights_per_model()

    from inference import load_model, infer_video_multi_model

    # Build models_info list — one entry per model type
    models_info = []
    for mt in ModelType:
        w = weights_map[mt.value]
        try:
            model = load_model(w)
        except Exception as exc:
            logger.warning(f"[{self.request.id}] Could not load {mt.value} weights {w}: {exc}")
            continue
        color = settings.MODEL_COLORS[mt.value]
        models_info.append((model, mt.value, color))

    if not models_info:
        raise RuntimeError("No models could be loaded for rendering")

    frame_count = infer_video_multi_model(
        models_info=models_info,
        video_path=str(video_path),
        conf_thresh=0.4,
        save_path=str(output_path),
        no_display=True,
    )

    if not output_path.exists():
        raise FileNotFoundError(f"Annotated video not created: {output_path}")

    with get_session() as session:
        clip = session.get(Clip, clip_id)
        clip.mp4_path = str(output_path)
        clip.status = ClipStatus.ANNOTATED

    logger.info(
        f"[{self.request.id}] Annotated MP4 saved: {output_path} "
        f"({frame_count} frames, {len(models_info)} models)"
    )
    return {
        "status": "annotated",
        "clip_id": clip_id,
        "mp4_path": str(output_path),
        "models_used": [m[1] for m in models_info],
        "frame_count": frame_count,
    }
