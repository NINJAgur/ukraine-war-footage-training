"""
ml-engine/tasks/train_finetune.py

Celery task: Stage 2 fine-tuning — merge auto-labeled custom datasets
(filtering to classes relevant to the run's model_type), train YOLOv8m
starting from GENERAL baseline weights, save best.pt.

Expects a TrainingRun with:
  - stage      = FINETUNE
  - model_type = GENERAL | AIRCRAFT | VEHICLE | PERSONNEL
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
from tasks.train_baseline import _make_epoch_callbacks

sys.path.insert(0, str(Path(__file__).parent.parent / "core"))

logger = logging.getLogger(__name__)

def _class_remap(model_type: ModelType) -> dict[int, int]:
    if model_type == ModelType.AIRCRAFT:
        return {0: 0}
    if model_type == ModelType.VEHICLE:
        return {1: 1}
    if model_type == ModelType.PERSONNEL:
        return {2: 2}
    return {0: 0, 1: 1, 2: 2}  # GENERAL


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
    datasets: list[tuple[int, str]],
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

    for ds_id, ds_yolo_dir in datasets:
        ds_dir = Path(ds_yolo_dir)
        for split in ("train", "val"):
            for src_lbl in (ds_dir / split / "labels").glob("*.txt"):
                dst_lbl = merged_dir / split / "labels" / f"{ds_id}_{src_lbl.name}"
                kept = _filter_label_file(src_lbl, dst_lbl, remap)
                if kept == 0:
                    dst_lbl.unlink(missing_ok=True)
                    continue
                src_img = ds_dir / split / "images" / (src_lbl.stem + ".jpg")
                if src_img.exists():
                    shutil.copy2(src_img, merged_dir / split / "images" / f"{ds_id}_{src_img.name}")

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
    logger.info(f"[train_finetune] task_id={self.request.id}  training_run_id={training_run_id}")

    with get_session() as session:
        run = session.get(TrainingRun, training_run_id)
        if run is None:
            raise ValueError(f"TrainingRun {training_run_id} not found")
        if run.status == TrainingStatus.DONE:
            return {"status": "skipped", "training_run_id": training_run_id}

        model_type = run.model_type
        dataset_ids = run.dataset_ids or []
        baseline_weights = run.baseline_weights or settings.YOLO_MODEL

        if dataset_ids:
            datasets = session.query(Dataset).filter(Dataset.id.in_(dataset_ids)).all()
            if not datasets:
                raise ValueError(f"No datasets found for ids: {dataset_ids}")
            not_packaged = [d.id for d in datasets if d.status not in (DatasetStatus.PACKAGED, DatasetStatus.TRAINED)]
            if not_packaged:
                raise ValueError(f"Datasets not yet packaged: {not_packaged}")
            datasets_snapshot = [(d.id, d.yolo_dir_path) for d in datasets]
        else:
            datasets_snapshot = []

        run.status = TrainingStatus.RUNNING
        run.celery_task_id = self.request.id
        run.started_at = datetime.utcnow()
        logger.info(
            f"[train_finetune] model={model_type.value}  datasets={len(dataset_ids)}  "
            f"ids={dataset_ids}  epochs={settings.YOLO_EPOCHS_FINETUNE}\n"
            f"    baseline_weights={baseline_weights}"
        )

    from core.main import train_model

    run_dir = settings.RUNS_DIR / "finetune" / model_type.value
    run_name = f"finetune_{model_type.value}_{training_run_id}"
    weights_path = run_dir / run_name / "weights" / "best.pt"
    merged_dir = settings.DATASETS_DIR / f"merged_{model_type.value}_{training_run_id}"

    try:
        import os
        os.chdir(Path(__file__).parent.parent)  # ensure CWD = ml-engine/ so YOLO writes runs/ there

        kaggle_dir = settings.KAGGLE_CACHE_DIR / "merged" / model_type.value
        if not datasets_snapshot:
            # No custom datasets — fine-tune on Kaggle merged only
            yaml_path = kaggle_dir / "dataset.yaml"
            if not yaml_path.exists():
                raise FileNotFoundError(f"Pre-built merged dataset not found: {yaml_path}")
            total_train = len(list((kaggle_dir / "train" / "images").glob("*.jpg")))
            total_val   = len(list((kaggle_dir / "val"   / "images").glob("*.jpg")))
            logger.info(
                f"[train_finetune] {model_type.value}: Kaggle only  train={total_train}  val={total_val}"
            )
        else:
            # Merge scraped datasets, then combine with Kaggle via multi-path YAML
            _merge_datasets(datasets_snapshot, merged_dir, model_type)
            class_names = settings.MODEL_CLASSES[model_type.value]
            yaml_path = merged_dir / "combined_data.yaml"
            with open(yaml_path, "w") as f:
                yaml.dump(
                    {
                        "train": [
                            str(kaggle_dir / "train" / "images"),
                            str(merged_dir / "train" / "images"),
                        ],
                        "val": [
                            str(kaggle_dir / "val" / "images"),
                            str(merged_dir / "val" / "images"),
                        ],
                        "nc": len(class_names),
                        "names": class_names,
                    },
                    f,
                    default_flow_style=False,
                )
            total_train = (
                len(list((kaggle_dir / "train" / "images").glob("*.jpg")))
                + len(list((merged_dir / "train" / "images").glob("*.jpg")))
            )
            total_val = (
                len(list((kaggle_dir / "val" / "images").glob("*.jpg")))
                + len(list((merged_dir / "val" / "images").glob("*.jpg")))
            )
            logger.info(
                f"[train_finetune] {model_type.value}: Kaggle + {len(datasets_snapshot)} scraped datasets  "
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
            extra_callbacks=dict(zip(
                ("on_train_epoch_start", "on_fit_epoch_end"),
                _make_epoch_callbacks(training_run_id, settings.YOLO_EPOCHS_FINETUNE),
            )),
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
            for did in [did for did, _ in datasets_snapshot]:
                ds = session.get(Dataset, did)
                if ds:
                    ds.status = DatasetStatus.TRAINED
            session.flush()

            # Clip dataset dirs safe to delete: no other queued/running run references them
            active_runs = session.query(TrainingRun).filter(
                TrainingRun.status.in_([TrainingStatus.QUEUED, TrainingStatus.RUNNING])
            ).all()
            pending_ds_ids = {did for r in active_runs for did in (r.dataset_ids or [])}
            clip_dirs_to_delete = [
                Path(dyolo) for did, dyolo in datasets_snapshot
                if did not in pending_ds_ids and dyolo
            ]

        for d in clip_dirs_to_delete:
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
                logger.info(f"[{self.request.id}] Deleted clip dataset dir {d.name}")

        if merged_dir.exists():
            shutil.rmtree(merged_dir, ignore_errors=True)
            logger.info(f"[{self.request.id}] Deleted merged dir {merged_dir.name}")

        map50 = metrics.get("metrics/mAP50(B)", metrics.get("mAP50(B)", 0.0))
        logger.info(
            f"[train_finetune] {model_type.value}: -> DONE  run_id={training_run_id}  "
            f"mAP50={map50:.3f}  datasets_used={len(datasets_snapshot)}\n"
            f"    weights={weights_path}"
        )
        return {
            "status": "done",
            "training_run_id": training_run_id,
            "model_type": model_type.value,
            "weights_path": str(weights_path),
            "metrics": metrics,
            "datasets_used": len(datasets_snapshot),
        }

    except Exception as exc:
        logger.error(f"[train_finetune] {model_type.value}: -> ERROR  run_id={training_run_id}  {exc}")
        with get_session() as session:
            run = session.get(TrainingRun, training_run_id)
            if run:
                run.status = TrainingStatus.ERROR
                run.error_message = str(exc)[:2000]
                run.completed_at = datetime.utcnow()
        raise


if __name__ == "__main__":
    import sys
    import os
    import logging as _logging

    _logging.basicConfig(level=_logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    import argparse as _argparse
    _parser = _argparse.ArgumentParser()
    _parser.add_argument("model_type", choices=[m.value for m in ModelType])
    _parser.add_argument("--epochs", type=int, default=None)
    _args = _parser.parse_args()
    _model_type = ModelType(_args.model_type.upper())
    if _args.epochs:
        settings.YOLO_EPOCHS_FINETUNE = _args.epochs

    # Find latest weights: prefer previous finetune, fall back to baseline
    def _latest_weights_for(model: ModelType):
        for stage in ("finetune", "baseline"):
            runs_dir = settings.RUNS_DIR / stage / model.value
            if not runs_dir.exists():
                continue
            candidates = sorted(
                (d for d in runs_dir.iterdir() if d.is_dir()),
                key=lambda d: int(d.name.rsplit("_", 1)[-1]) if d.name.rsplit("_", 1)[-1].isdigit() else 0,
                reverse=True,
            )
            for run_dir in candidates:
                w = run_dir / "weights" / "best.pt"
                if w.exists():
                    return str(w)
        return None

    _weights = _latest_weights_for(_model_type)
    if not _weights:
        print(f"ERROR: no baseline weights found for {_model_type.value} — run baseline first")
        sys.exit(1)

    print(f"Model:   {_model_type.value}")
    print(f"Weights: {_weights}")

    with get_session() as _session:
        _run = TrainingRun(
            stage=TrainingStage.FINETUNE,
            model_type=_model_type,
            status=TrainingStatus.QUEUED,
            baseline_weights=_weights,
            created_at=datetime.utcnow(),
        )
        _session.add(_run)
        _session.flush()
        _run_id = _run.id
        print(f"Created TrainingRun id={_run_id}")

    os.chdir(Path(__file__).parent.parent)
    train_finetune.run(training_run_id=_run_id)
