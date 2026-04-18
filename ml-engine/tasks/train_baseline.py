"""
ml-engine/tasks/train_baseline.py

Celery task: Stage 1 training — download Kaggle military datasets,
train YOLOv8m from pretrained weights, save best.pt.

Triggered by Admin selecting "Train Baseline" in the web UI.
On completion, weights are available for Stage 2 (train_finetune).
"""
import logging
import sys
from datetime import datetime
from pathlib import Path

from celery_app import celery_app
from config import settings
from db.models import TrainingRun, TrainingStage, TrainingStatus
from db.session import get_session

sys.path.insert(0, str(Path(__file__).parent.parent / "core"))

logger = logging.getLogger(__name__)

# Kaggle datasets to use for Stage 1 baseline training.
# Each entry: (dataset_handle, nc, class_names)
BASELINE_DATASETS = [
    (
        "sudipchakrabarty/kiit-mita",
        3,
        ["military vehicle", "soldier", "weapon"],
    ),
]


def _extract_metrics(results) -> dict:
    """Pull key metrics out of an ultralytics Results object."""
    try:
        return dict(results.results_dict)
    except Exception:
        return {}


@celery_app.task(
    bind=True,
    name="tasks.train_baseline.train_baseline",
    queue="gpu",
    autoretry_for=(Exception,),
    max_retries=1,
    default_retry_delay=300,
)
def train_baseline(self, training_run_id: int) -> dict:
    """
    Stage 1: download Kaggle datasets, train YOLOv8m, save best.pt.
    Updates TrainingRun status throughout. Idempotent via DB status check.
    """
    logger.info(f"[{self.request.id}] train_baseline training_run_id={training_run_id}")

    with get_session() as session:
        run = session.get(TrainingRun, training_run_id)
        if run is None:
            raise ValueError(f"TrainingRun {training_run_id} not found")
        if run.status == TrainingStatus.DONE:
            logger.info(f"[{self.request.id}] TrainingRun {training_run_id} already done — skipping")
            return {"status": "skipped", "training_run_id": training_run_id}
        run.status = TrainingStatus.RUNNING
        run.celery_task_id = self.request.id
        run.started_at = datetime.utcnow()

    from main import download_dataset, detect_dataset_structure, create_yaml, train_model

    run_dir = settings.RUNS_DIR / "baseline"
    run_name = f"baseline_{training_run_id}"
    weights_path = run_dir / run_name / "weights" / "best.pt"

    try:
        all_metrics = {}

        for dataset_handle, nc, class_names in BASELINE_DATASETS:
            logger.info(f"[{self.request.id}] Downloading Kaggle dataset: {dataset_handle}")
            dataset_path = download_dataset(dataset_handle)
            paths, dataset_path = detect_dataset_structure(dataset_path)
            if not paths:
                raise ValueError(f"No train/val structure found in {dataset_handle}")

            yaml_path = create_yaml(dataset_path, paths, nc, class_names)
            logger.info(
                f"[{self.request.id}] Training on {dataset_handle}: "
                f"nc={nc}  epochs={settings.YOLO_EPOCHS_BASELINE}"
            )

            results = train_model(
                yaml_path=yaml_path,
                epochs=settings.YOLO_EPOCHS_BASELINE,
                imgsz=settings.YOLO_IMG_SIZE,
                batch=settings.YOLO_BATCH_SIZE,
                device=settings.GPU_DEVICE,
                project=str(run_dir),
                name=run_name,
                weights=None,
                resume=False,
            )
            all_metrics.update(_extract_metrics(results))

        if not weights_path.exists():
            raise FileNotFoundError(f"Training finished but best.pt not found: {weights_path}")

        with get_session() as session:
            run = session.get(TrainingRun, training_run_id)
            run.status = TrainingStatus.DONE
            run.weights_path = str(weights_path)
            run.metrics = all_metrics
            run.completed_at = datetime.utcnow()

        logger.info(
            f"[{self.request.id}] Baseline training complete. "
            f"Weights: {weights_path}  Metrics: {all_metrics}"
        )
        return {
            "status": "done",
            "training_run_id": training_run_id,
            "weights_path": str(weights_path),
            "metrics": all_metrics,
        }

    except Exception as exc:
        logger.error(f"[{self.request.id}] Baseline training failed: {exc}")
        with get_session() as session:
            run = session.get(TrainingRun, training_run_id)
            if run:
                run.status = TrainingStatus.ERROR
                run.error_message = str(exc)[:2000]
                run.completed_at = datetime.utcnow()
        raise
