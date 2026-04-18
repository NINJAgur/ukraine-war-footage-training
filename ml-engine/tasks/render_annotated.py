"""
ml-engine/tasks/render_annotated.py

Celery task: run YOLO inference on a downloaded Clip video, produce an
annotated H.264 MP4, and update Clip.mp4_path + status → ANNOTATED.

Weights priority:
  1. Latest FINETUNE TrainingRun best.pt
  2. Latest BASELINE TrainingRun best.pt
  3. settings.YOLO_MODEL (yolov8m.pt fallback — pretrained, not trained on our data)

Pipeline:
  Clip(LABELED/DOWNLOADED) + weights → annotated MP4 → Clip(ANNOTATED)
"""
import logging
import sys
from pathlib import Path

from celery_app import celery_app
from config import settings
from db.models import Clip, ClipStatus, TrainingRun, TrainingStage, TrainingStatus
from db.session import get_session

sys.path.insert(0, str(Path(__file__).parent.parent / "core"))

logger = logging.getLogger(__name__)


def _best_weights() -> str:
    """Return path to the best available model weights."""
    with get_session() as session:
        for stage in [TrainingStage.FINETUNE, TrainingStage.BASELINE]:
            run = (
                session.query(TrainingRun)
                .filter(
                    TrainingRun.stage == stage,
                    TrainingRun.status == TrainingStatus.DONE,
                    TrainingRun.weights_path.isnot(None),
                )
                .order_by(TrainingRun.completed_at.desc())
                .first()
            )
            if run and Path(run.weights_path).exists():
                logger.info(f"Using {stage.value} weights: {run.weights_path}")
                return run.weights_path

    logger.info(f"No trained weights found — using pretrained {settings.YOLO_MODEL}")
    return settings.YOLO_MODEL


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
    Run YOLO inference on a Clip's raw video and save an annotated MP4.
    Updates Clip.mp4_path and status → ANNOTATED on success.
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
        clip_hash = clip.url_hash[:12]
        if clip.mp4_path and Path(clip.mp4_path).exists():
            logger.info(f"[{self.request.id}] Already annotated: {clip.mp4_path}")
            return {"status": "skipped", "clip_id": clip_id}

    output_path = settings.ANNOTATED_VIDEO_DIR / f"{clip_hash}_annotated.mp4"
    weights = _best_weights()

    from inference import load_model, infer_video

    model = load_model(weights)
    infer_video(
        model=model,
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

    logger.info(f"[{self.request.id}] Annotated MP4 saved: {output_path}")
    return {
        "status": "annotated",
        "clip_id": clip_id,
        "mp4_path": str(output_path),
        "weights_used": weights,
    }
