"""
ml-engine/tasks/train_finetune.py

Celery task: Stage 2 fine-tuning — merge auto-labeled custom datasets
(filtering to classes relevant to the run's model_type), train YOLOv8m
starting from GENERAL baseline weights, save best.pt.

Expects a TrainingRun with:
  - stage      = FINETUNE
  - model_type = GENERAL | SOLDIER | VEHICLE | AIRCRAFT
  - dataset_ids = [list of Dataset.id, all status=PACKAGED]
  - baseline_weights = path to GENERAL best.pt (or None → yolov8m.pt)
"""
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

import yaml

from celery_app import celery_app
from config import settings
from db.models import Dataset, DatasetStatus, ModelType, TrainingRun, TrainingStage, TrainingStatus
from db.session import get_session

sys.path.insert(0, str(Path(__file__).parent.parent / "core"))

logger = logging.getLogger(__name__)

# Full GDINO class list (order must match GDINO_TEXT_PROMPT)
_ALL_CLASSES = [c.strip() for c in settings.GDINO_TEXT_PROMPT.split(",") if c.strip()]


def _class_remap(model_type: ModelType) -> dict[int, int]:
    """
    Build old_class_id → new_class_id mapping for the given model_type.
    Returns -1 for classes not belonging to this model.
    """
    target = settings.MODEL_CLASSES[model_type.value]
    remap: dict[int, int] = {}
    new_id = 0
    for old_id, cls in enumerate(_ALL_CLASSES):
        if cls in target:
            remap[old_id] = new_id
            new_id += 1
        else:
            remap[old_id] = -1
    return remap


def _filter_label_file(src: Path, dst: Path, remap: dict[int, int]) -> int:
    """Copy a YOLO .txt label file keeping only lines whose class is in remap."""
    kept_lines = []
    for line in src.read_text().strip().splitlines():
        parts = line.split()
        if not parts:
            continue
        new_id = remap.get(int(parts[0]), -1)
        if new_id >= 0:
            kept_lines.append(f"{new_id} {' '.join(parts[1:])}")
    dst.write_text("\n".join(kept_lines) + ("\n" if kept_lines else ""))
    return len(kept_lines)


def _merge_datasets(
    datasets: list[Dataset],
    merged_dir: Path,
    model_type: ModelType,
) -> Path:
    """
    Merge multiple YOLO dataset dirs into one, filtering labels to
    the classes relevant for model_type and remapping class IDs.
    Returns path to merged data.yaml.
    """
    remap = _class_remap(model_type)
    class_names = settings.MODEL_CLASSES[model_type.value]

    for split in ("train", "val"):
        (merged_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (merged_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    for ds in datasets:
        ds_dir = Path(ds.yolo_dir_path)
        for split in ("train", "val"):
            for src_img in (ds_dir / split / "images").glob("*.jpg"):
                shutil.copy2(src_img, merged_dir / split / "images" / f"{ds.id}_{src_img.name}")
            for src_lbl in (ds_dir / split / "labels").glob("*.txt"):
                dst_lbl = merged_dir / split / "labels" / f"{ds.id}_{src_lbl.name}"
                _filter_label_file(src_lbl, dst_lbl, remap)

    yaml_path = merged_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(
            {
                "path": str(merged_dir),
                "train": "train/images",
                "val": "val/images",
                "nc": len(class_names),
                "names": class_names,
            },
            f,
            default_flow_style=False,
        )
    return yaml_path


def _extract_metrics(results) -> dict:
    try:
        return dict(results.results_dict)
    except Exception:
        return {}


@celery_app.task(
    bind=True,
    name="tasks.train_finetune.train_finetune",
    queue="gpu",
    autoretry_for=(Exception,),
    max_retries=1,
    default_retry_delay=300,
)
def train_finetune(self, training_run_id: int) -> dict:
    """
    Stage 2: merge class-filtered custom datasets, fine-tune from baseline.
    Idempotent via DB status check.
    """
    logger.info(f"[{self.request.id}] train_finetune training_run_id={training_run_id}")

    with get_session() as session:
        run = session.get(TrainingRun, training_run_id)
        if run is None:
            raise ValueError(f"TrainingRun {training_run_id} not found")
        if run.status == TrainingStatus.DONE:
            return {"status": "skipped", "training_run_id": training_run_id}

        model_type = run.model_type
        dataset_ids = run.dataset_ids or []
        baseline_weights = run.baseline_weights or settings.YOLO_MODEL

        datasets = session.query(Dataset).filter(Dataset.id.in_(dataset_ids)).all()
        if not datasets:
            raise ValueError(f"No datasets found for ids: {dataset_ids}")
        not_packaged = [d.id for d in datasets if d.status != DatasetStatus.PACKAGED]
        if not_packaged:
            raise ValueError(f"Datasets not yet packaged: {not_packaged}")

        run.status = TrainingStatus.RUNNING
        run.celery_task_id = self.request.id
        run.started_at = datetime.utcnow()
        datasets_snapshot = [(d.id, d.yolo_dir_path) for d in datasets]

    from main import train_model

    run_dir = settings.RUNS_DIR / "finetune" / model_type.value
    run_name = f"finetune_{model_type.value}_{training_run_id}"
    weights_path = run_dir / run_name / "weights" / "best.pt"
    merged_dir = settings.DATASETS_DIR / f"merged_{model_type.value}_{training_run_id}"

    try:
        with get_session() as session:
            datasets = [session.get(Dataset, did) for did, _ in datasets_snapshot]

        yaml_path = _merge_datasets(datasets, merged_dir, model_type)
        total_train = len(list((merged_dir / "train" / "images").glob("*.jpg")))
        total_val   = len(list((merged_dir / "val"   / "images").glob("*.jpg")))
        logger.info(
            f"[{self.request.id}] [{model_type.value}] Merged {len(datasets)} datasets: "
            f"train={total_train}  val={total_val}"
        )

        results = train_model(
            yaml_path=str(yaml_path),
            epochs=settings.YOLO_EPOCHS_FINETUNE,
            imgsz=settings.YOLO_IMG_SIZE,
            batch=settings.YOLO_BATCH_SIZE,
            device=settings.GPU_DEVICE,
            project=str(run_dir),
            name=run_name,
            weights=baseline_weights,
            resume=False,
        )

        if not weights_path.exists():
            raise FileNotFoundError(f"Training finished but best.pt not found: {weights_path}")

        metrics = _extract_metrics(results)
        with get_session() as session:
            run = session.get(TrainingRun, training_run_id)
            run.status = TrainingStatus.DONE
            run.weights_path = str(weights_path)
            run.metrics = metrics
            run.completed_at = datetime.utcnow()

        logger.info(
            f"[{self.request.id}] [{model_type.value}] Fine-tune done. Weights: {weights_path}"
        )
        return {
            "status": "done",
            "training_run_id": training_run_id,
            "model_type": model_type.value,
            "weights_path": str(weights_path),
            "metrics": metrics,
            "datasets_used": len(datasets),
        }

    except Exception as exc:
        logger.error(f"[{self.request.id}] [{model_type.value}] Fine-tune failed: {exc}")
        with get_session() as session:
            run = session.get(TrainingRun, training_run_id)
            if run:
                run.status = TrainingStatus.ERROR
                run.error_message = str(exc)[:2000]
                run.completed_at = datetime.utcnow()
        raise
