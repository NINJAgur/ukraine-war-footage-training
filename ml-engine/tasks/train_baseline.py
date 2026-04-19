"""
ml-engine/tasks/train_baseline.py

Celery task: Stage 1 training — download a Kaggle military dataset for the
requested model type, train YOLOv8m from pretrained weights, save best.pt.

Model types and their Kaggle datasets:
  GENERAL  — rawsi18/military-assets-dataset-12-classes-yolo8-format
  SOLDIER  — hillsworld/human-detection-yolo
  VEHICLE  — sudipchakrabarty/kiit-mita
  AIRCRAFT — rookieengg/military-aircraft-detection-dataset-yolo-format
           + muki2003/yolo-drone-detection-dataset (second pass, same run dir)

Triggered by Admin "Train Baseline" in the web UI, which creates one
TrainingRun per model type and dispatches this task four times.
"""
import logging
import sys
from datetime import datetime
from pathlib import Path

from celery_app import celery_app
from config import settings
from db.models import ModelType, TrainingRun, TrainingStage, TrainingStatus
from db.session import get_session

sys.path.insert(0, str(Path(__file__).parent.parent / "core"))

logger = logging.getLogger(__name__)

# Kaggle dataset config per model type.
# Each entry: (dataset_handle, nc, class_names)
# nc=None / class_names=None → auto-detect from dataset's own data.yaml.
# Using dataset-native classes for baseline (backbone learns domain features);
# fine-tuning will re-align to our standardised MODEL_CLASSES vocabulary.
BASELINE_DATASETS: dict[str, list[tuple]] = {
    # kiit-mita: 1700 imgs, nc=7 (Artilary, Missile, Radar, M. Rocket Launcher,
    #                                Soldier, Tank, Vehicle)
    ModelType.GENERAL:  [("sudipchakrabarty/kiit-mita", None, None)],
    ModelType.SOLDIER:  [("sudipchakrabarty/kiit-mita", None, None)],
    ModelType.VEHICLE:  [("sudipchakrabarty/kiit-mita", None, None)],
    # drone-detect: 37900 imgs, nc=2 (Dron, Dron2)
    ModelType.AIRCRAFT: [("mihprofi/drone-detect", None, None)],
}


def _extract_metrics(results) -> dict:
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
    Stage 1: download Kaggle dataset(s) for this run's model_type,
    train YOLOv8m, save best.pt. Idempotent via DB status check.
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

    datasets = BASELINE_DATASETS.get(model_type)
    if not datasets:
        raise ValueError(f"No baseline datasets configured for model_type={model_type}")

    from main import download_dataset, detect_dataset_structure, create_yaml, train_model

    run_dir = settings.RUNS_DIR / "baseline" / model_type.value
    run_name = f"baseline_{model_type.value}_{training_run_id}"
    weights_path = run_dir / run_name / "weights" / "best.pt"

    try:
        all_metrics = {}

        for dataset_handle, nc, class_names in datasets:
            logger.info(
                f"[{self.request.id}] [{model_type.value}] "
                f"Downloading Kaggle dataset: {dataset_handle}"
            )
            dataset_path = download_dataset(dataset_handle)
            paths, dataset_path = detect_dataset_structure(dataset_path)
            if not paths:
                raise ValueError(f"No train/val structure found in {dataset_handle}")

            # Auto-detect nc/names from dataset's own data.yaml when not supplied.
            if nc is None or class_names is None:
                import glob as _glob, yaml as _yaml
                existing_yamls = (
                    _glob.glob(f"{dataset_path}/**/*.yaml", recursive=True) +
                    _glob.glob(f"{dataset_path}/**/*.yml",  recursive=True)
                )
                for y in existing_yamls:
                    try:
                        with open(y) as _f:
                            _d = _yaml.safe_load(_f)
                        if isinstance(_d, dict) and "nc" in _d and "names" in _d:
                            nc          = _d["nc"]
                            class_names = _d["names"]
                            logger.info(
                                f"[{self.request.id}] Auto-detected from {y}: "
                                f"nc={nc}  names={class_names}"
                            )
                            break
                    except Exception:
                        pass
                if nc is None:
                    raise ValueError(
                        f"Could not auto-detect nc/names from {dataset_handle}. "
                        "Supply them explicitly in BASELINE_DATASETS."
                    )

            yaml_path = create_yaml(dataset_path, paths, nc, class_names)
            logger.info(
                f"[{self.request.id}] [{model_type.value}] "
                f"Training on {dataset_handle}: nc={nc}  epochs={settings.YOLO_EPOCHS_BASELINE}"
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
            f"[{self.request.id}] [{model_type.value}] Baseline done. "
            f"Weights: {weights_path}"
        )
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
