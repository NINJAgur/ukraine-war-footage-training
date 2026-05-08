"""
ml-engine/tasks/train_baseline.py

Celery task: Stage 1 (cold-start) baseline training.

Reads from pre-built specialist dataset folders produced by:
    python scripts/build_specialist_datasets.py

Expected structure per model:
    media/kaggle_datasets/merged/<MODEL>/
        train/images/   train/labels/
        val/images/     val/labels/
        dataset.yaml

Run the build script once before training. All training runs read from these
permanent folders — no on-the-fly dataset merging.
"""
import logging
from datetime import datetime
from pathlib import Path

from celery_app import celery_app
from config import settings
from db.models import ModelType, TrainingRun, TrainingStage, TrainingStatus
from db.session import get_session

logger = logging.getLogger(__name__)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _extract_metrics(results) -> dict:
    try:
        return dict(results.results_dict)
    except Exception:
        return {}


def _count_images(img_dir: Path) -> int:
    return sum(1 for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)


def _make_epoch_callback(run_id: int, total_epochs: int):
    def on_fit_epoch_end(trainer):
        try:
            m = trainer.metrics or {}
            progress = {
                "epoch":    trainer.epoch + 1,
                "epochs":   total_epochs,
                "map50":    round(float(m.get("metrics/mAP50(B)", 0)), 4),
                "box_loss": round(float(trainer.loss_items[0]), 4) if getattr(trainer, "loss_items", None) is not None else None,
                "cls_loss": round(float(trainer.loss_items[1]), 4) if getattr(trainer, "loss_items", None) is not None else None,
            }
            with get_session() as session:
                run = session.get(TrainingRun, run_id)
                if run:
                    combined = dict(run.metrics or {})
                    combined["epoch_progress"] = progress
                    run.metrics = combined
        except Exception:
            pass
    return on_fit_epoch_end


@celery_app.task(
    bind=True,
    name="tasks.train_baseline.train_baseline",
    queue="gpu",
    autoretry_for=(Exception,),
    max_retries=1,
    default_retry_delay=300,
)
def train_baseline(self, training_run_id: int, weights: str = None) -> dict:
    """
    Stage 1 baseline training. Reads pre-built merged dataset for this run's
    model_type; trains YOLOv8m; saves best.pt.  Idempotent via DB status check.
    """
    logger.info(f"[{self.request.id}] train_baseline training_run_id={training_run_id}")

    with get_session() as session:
        run = session.get(TrainingRun, training_run_id)
        if run is None:
            raise ValueError(f"TrainingRun {training_run_id} not found")
        if run.status == TrainingStatus.DONE:
            logger.info(f"[{self.request.id}] TrainingRun {training_run_id} already done — skipping")
            return {"status": "skipped", "training_run_id": training_run_id}
        model_type = run.model_type
        run.status = TrainingStatus.RUNNING
        run.celery_task_id = self.request.id
        run.started_at = datetime.utcnow()

    merged_dir = settings.KAGGLE_CACHE_DIR / "merged" / model_type.value
    yaml_path  = merged_dir / "dataset.yaml"

    if not yaml_path.exists():
        raise FileNotFoundError(
            f"Pre-built dataset not found: {yaml_path}\n"
            f"Run first:  cd ml-engine && python scripts/build_specialist_datasets.py --models {model_type.value}"
        )

    train_img_dir = merged_dir / "train" / "images"
    total_train   = _count_images(train_img_dir)

    from core.main import train_model

    run_dir      = settings.RUNS_DIR / "baseline" / model_type.value
    run_name     = f"baseline_{model_type.value}_{training_run_id}"
    weights_path = run_dir / run_name / "weights" / "best.pt"

    try:
        with get_session() as session:
            run = session.get(TrainingRun, training_run_id)
            run.metrics = {"total_train_images": total_train}

        logger.info(
            f"[{self.request.id}] [{model_type.value}] "
            f"Training: epochs={settings.YOLO_EPOCHS_BASELINE}  train_images={total_train}  "
            f"dataset={merged_dir}"
        )

        results = train_model(
            yaml_path=str(yaml_path),
            epochs=settings.YOLO_EPOCHS_BASELINE,
            imgsz=settings.YOLO_IMG_SIZE,
            batch=settings.YOLO_BATCH_SIZE,
            device=settings.GPU_DEVICE,
            project=str(run_dir),
            name=run_name,
            weights=weights,
            resume=False,
            extra_callbacks={"on_fit_epoch_end": _make_epoch_callback(training_run_id, settings.YOLO_EPOCHS_BASELINE)},
        )

        all_metrics = _extract_metrics(results)
        all_metrics["total_train_images"] = total_train

        if not weights_path.exists():
            raise FileNotFoundError(f"Training finished but best.pt not found: {weights_path}")

        with get_session() as session:
            run = session.get(TrainingRun, training_run_id)
            run.status = TrainingStatus.DONE
            run.weights_path = str(weights_path)
            run.metrics = all_metrics
            run.completed_at = datetime.utcnow()

        logger.info(f"[{self.request.id}] [{model_type.value}] Baseline done. Weights: {weights_path}")
        return {
            "status": "done",
            "training_run_id": training_run_id,
            "model_type": model_type.value,
            "weights_path": str(weights_path),
            "metrics": all_metrics,
        }

    except Exception as exc:
        logger.error(f"[{self.request.id}] [{model_type.value}] Baseline failed: {exc}")
        with get_session() as session:
            run = session.get(TrainingRun, training_run_id)
            if run:
                run.status = TrainingStatus.ERROR
                run.error_message = str(exc)[:2000]
                run.completed_at = datetime.utcnow()
        raise
