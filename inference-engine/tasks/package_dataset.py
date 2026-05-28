"""
inference-engine/tasks/package_dataset.py

Phase 2 — Packaging + Merging (per clip, Q=pipeline):
  Dataset(LABELED) → 80/20 split → filter+append into merged/<MODEL>/ → delete hash dir → Dataset(PACKAGED)

Phase 3 — Trigger check (chord callback, once after ALL clips in batch are done):
  Count PACKAGED per model → ≥5 → TrainingRun(QUEUED) → mark TRAINED → prepare_finetune_batch

Phase 4 — Finetune dispatch (Q=pipeline):
  prepare_finetune_batch: upload merged/<MODEL>/ to GCS (remote) or leave on disk (local)
  → start training VM → dispatch train_finetune × N → Q=training
"""
import logging
import random
import shutil
from pathlib import Path
from typing import Optional

import yaml

from celery_app import celery_app
from config import settings
from db.models import (
    Dataset, DatasetStatus,
    ModelType, TrainingRun, TrainingStage, TrainingStatus,
)
from db.session import get_session
from tasks.weights import _latest_weights

logger = logging.getLogger(__name__)

VAL_SPLIT = 0.2
SPLIT_SEED = 42
FINETUNE_MIN_DATASETS = 5


def create_train_val_split(dataset_dir: Path) -> tuple:
    """Split train/images + train/labels 80/20 into val/. Returns (train_count, val_count)."""
    train_img_dir = dataset_dir / "train" / "images"
    train_lbl_dir = dataset_dir / "train" / "labels"
    val_img_dir   = dataset_dir / "val" / "images"
    val_lbl_dir   = dataset_dir / "val" / "labels"
    val_img_dir.mkdir(parents=True, exist_ok=True)
    val_lbl_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(train_img_dir.glob("*.jpg"))
    random.seed(SPLIT_SEED)
    random.shuffle(images)

    val_count = max(1, int(len(images) * VAL_SPLIT))
    val_images = images[:val_count]
    train_images = images[val_count:]

    for img_path in val_images:
        label_path = train_lbl_dir / (img_path.stem + ".txt")
        shutil.move(str(img_path), val_img_dir / img_path.name)
        if label_path.exists():
            shutil.move(str(label_path), val_lbl_dir / label_path.name)

    return len(train_images), val_count


def update_data_yaml(yaml_path: Path, dataset_dir: Path) -> None:
    """Rewrite data.yaml with absolute paths and correct train/val split dirs."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    data["path"]  = str(dataset_dir)
    data["train"] = "train/images"
    data["val"]   = "val/images"
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)


def _class_remap(model_type: ModelType) -> dict:
    """Return {src_class_id: dst_class_id} filter for the given model type."""
    if model_type == ModelType.AIRCRAFT:
        return {0: 0}
    if model_type == ModelType.VEHICLE:
        return {1: 1}
    if model_type == ModelType.PERSONNEL:
        return {2: 2}
    return {0: 0, 1: 1, 2: 2}  # GENERAL


def _filter_label_file(src: Path, dst: Path, remap: dict) -> int:
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


def _append_to_merged(dataset_dir: Path, dataset_id: int, model_type: ModelType) -> int:
    """
    Filter and append one clip's YOLO data into the persistent merged/<MODEL>/ dir.
    Both train and val splits are appended. Returns number of images written.
    """
    remap = _class_remap(model_type)
    merged_dir = settings.DATASETS_DIR / "merged" / model_type.value

    appended = 0
    for split in ("train", "val"):
        (merged_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (merged_dir / split / "labels").mkdir(parents=True, exist_ok=True)

        for src_lbl in (dataset_dir / split / "labels").glob("*.txt"):
            dst_lbl = merged_dir / split / "labels" / f"{dataset_id}_{src_lbl.name}"
            kept = _filter_label_file(src_lbl, dst_lbl, remap)
            if kept == 0:
                dst_lbl.unlink(missing_ok=True)
                continue
            src_img = dataset_dir / split / "images" / (src_lbl.stem + ".jpg")
            if src_img.exists():
                shutil.copy2(src_img, merged_dir / split / "images" / f"{dataset_id}_{src_img.name}")
                appended += 1

    # Always rewrite data.yaml to keep path current
    class_names = settings.MODEL_CLASSES[model_type.value]
    with open(merged_dir / "data.yaml", "w") as f:
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

    return appended


def _upload_merged_to_gcs(merged_dir: Path, model: str, bucket: str) -> None:
    """Upload all files in merged_dir to gs://bucket/merged/<model>/."""
    from google.cloud import storage as gcs
    prefix = f"merged/{model}"
    client = gcs.Client()
    bucket_obj = client.bucket(bucket)
    for f in merged_dir.rglob("*"):
        if f.is_file():
            blob_name = f"{prefix}/{f.relative_to(merged_dir).as_posix()}"
            bucket_obj.blob(blob_name).upload_from_filename(str(f))
    logger.info(f"[prepare_finetune_batch] Uploaded {merged_dir.name} → gs://{bucket}/{prefix}/")


def _start_training_vm() -> None:
    """Start the training VM via GCP Compute Engine API. No-op if GCP_PROJECT_ID is unset."""
    if not settings.GCP_PROJECT_ID:
        return
    from googleapiclient import discovery
    compute = discovery.build("compute", "v1")
    compute.instances().start(
        project=settings.GCP_PROJECT_ID,
        zone=settings.GCP_TRAINING_VM_ZONE,
        instance=settings.GCP_TRAINING_VM_NAME,
    ).execute()
    logger.info(f"[prepare_finetune_batch] Started training VM: {settings.GCP_TRAINING_VM_NAME}")


def _create_finetune_run(model_type: ModelType) -> Optional[tuple]:
    """Create a QUEUED TrainingRun for model_type if eligible. Returns (run_id, dataset_ids) or None."""
    with get_session() as session:
        done_cycles = (
            session.query(TrainingRun)
            .filter(TrainingRun.stage == TrainingStage.FINETUNE)
            .filter(TrainingRun.model_type == model_type)
            .filter(TrainingRun.status == TrainingStatus.DONE)
            .count()
        )
        if done_cycles >= settings.YOLO_FINETUNE_MAX_CYCLES:
            logger.info(
                f"[finetune] {model_type.value}: -> SKIP: max cycles reached "
                f"({done_cycles}/{settings.YOLO_FINETUNE_MAX_CYCLES})"
            )
            return None

        active = (
            session.query(TrainingRun)
            .filter(TrainingRun.stage == TrainingStage.FINETUNE)
            .filter(TrainingRun.model_type == model_type)
            .filter(TrainingRun.status.in_([TrainingStatus.QUEUED, TrainingStatus.RUNNING]))
            .first()
        )
        if active:
            logger.info(
                f"[finetune] {model_type.value}: -> SKIP: run already active  "
                f"run_id={active.id}  status={active.status.value}"
            )
            return None

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
            logger.info(
                f"[finetune] {model_type.value}: -> SKIP: only {len(relevant)}/{FINETUNE_MIN_DATASETS} "
                f"PACKAGED datasets — threshold not met"
            )
            return None

        dataset_ids = [d.id for d in relevant]
        try:
            baseline_weights = str(_latest_weights(model_type.value))
        except FileNotFoundError:
            baseline_weights = None
            logger.warning(f"[finetune] {model_type.value}: no baseline weights found — will use yolov8m.pt")

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

    logger.info(
        f"[finetune] {model_type.value}: -> QUEUED  run_id={run_id}  "
        f"datasets={len(dataset_ids)}  ids={dataset_ids}\n"
        f"    baseline_weights={Path(baseline_weights).name if baseline_weights else 'yolov8m.pt'}"
    )
    return (run_id, dataset_ids)


def _maybe_trigger_finetune() -> None:
    """Create TrainingRun records for eligible models and dispatch prepare_finetune_batch."""
    run_ids = []
    all_consumed: set = set()
    for model_type in [ModelType.AIRCRAFT, ModelType.VEHICLE, ModelType.PERSONNEL, ModelType.GENERAL]:
        result = _create_finetune_run(model_type)
        if result:
            run_id, dataset_ids = result
            run_ids.append(run_id)
            all_consumed.update(dataset_ids)

    if run_ids:
        # Mark consumed datasets TRAINED immediately — resets their counter for next cycle
        with get_session() as session:
            for ds_id in all_consumed:
                ds = session.get(Dataset, ds_id)
                if ds and ds.status == DatasetStatus.PACKAGED:
                    ds.status = DatasetStatus.TRAINED
        logger.info(f"[finetune] Marked {len(all_consumed)} datasets as TRAINED")

        celery_app.send_task(
            "tasks.package_dataset.prepare_finetune_batch",
            kwargs={"run_ids": run_ids},
            queue="pipeline",
        )
        logger.info(f"[finetune] Dispatched prepare_finetune_batch  run_ids={run_ids}")


@celery_app.task(
    name="tasks.package_dataset.trigger_finetune_check",
    queue="pipeline",
    max_retries=0,
)
def trigger_finetune_check() -> dict:
    """Chord callback: fires once after ALL auto_label_clip | package_dataset chains complete."""
    logger.info("[trigger_finetune_check] All clips processed — checking finetune thresholds")
    _maybe_trigger_finetune()
    return {"status": "checked"}


@celery_app.task(
    name="tasks.package_dataset.prepare_finetune_batch",
    queue="pipeline",
    max_retries=0,
)
def prepare_finetune_batch(run_ids: list) -> dict:
    """
    Merged dirs already built incrementally by package_dataset.
    1. Remote: upload merged/<MODEL>/ → GCS; delete local merged dir
       Local: leave merged dir on disk (train_finetune reads directly)
    2. Start training VM (no-op locally)
    3. Dispatch train_finetune per run → Q=training
    """
    with get_session() as session:
        runs = session.query(TrainingRun).filter(TrainingRun.id.in_(run_ids)).all()
        run_snapshots = [(r.id, r.model_type) for r in runs]

    _start_training_vm()

    for run_id, model_type in run_snapshots:
        merged_dir = settings.DATASETS_DIR / "merged" / model_type.value

        if settings.STORAGE_MODE == "remote" and settings.REMOTE_STORAGE_BUCKET:
            if merged_dir.exists():
                _upload_merged_to_gcs(merged_dir, model_type.value, settings.REMOTE_STORAGE_BUCKET)
                shutil.rmtree(merged_dir, ignore_errors=True)
                logger.info(f"[prepare_finetune_batch] Deleted local merged dir: {merged_dir.name}")
            scraped_merged_path = f"gs://{settings.REMOTE_STORAGE_BUCKET}/merged/{model_type.value}"
        else:
            scraped_merged_path = str(merged_dir) if merged_dir.exists() else None

        celery_app.send_task(
            "tasks.train_finetune.train_finetune",
            kwargs={"training_run_id": run_id, "scraped_merged_path": scraped_merged_path},
            queue="training",
        )
        logger.info(
            f"[prepare_finetune_batch] Dispatched train_finetune  run_id={run_id}  "
            f"path={scraped_merged_path}"
        )

    return {"prepared": len(run_snapshots), "run_ids": run_ids}


@celery_app.task(
    bind=True,
    name="tasks.package_dataset.package_dataset",
    queue="pipeline",
    autoretry_for=(Exception,),
    max_retries=2,
    default_retry_delay=60,
)
def package_dataset(self, dataset_id: Optional[int]) -> dict:
    """
    Split a LABELED Dataset 80/20, append into persistent merged/<MODEL>/ dirs,
    delete the clip hash dir, and mark PACKAGED.

    Called as the second step in an auto_label_clip | package_dataset Celery chain.
    dataset_id is None when auto_label_clip was skipped — returns immediately.
    """
    if dataset_id is None:
        return {"status": "skipped"}

    logger.info(f"[{self.request.id}] package_dataset dataset_id={dataset_id}")

    with get_session() as session:
        dataset = session.get(Dataset, dataset_id)
        if dataset is None:
            raise ValueError(f"Dataset {dataset_id} not found")
        if dataset.status == DatasetStatus.PACKAGED:
            logger.info(f"[{self.request.id}] Dataset {dataset_id} already packaged — skipping")
            return {"status": "skipped", "dataset_id": dataset_id}
        dataset_dir = Path(dataset.yolo_dir_path)
        yaml_path = Path(dataset.yaml_path)
        detected_types: list = dataset.detected_model_types or []

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset dir not found: {dataset_dir}")

    train_count, val_count = create_train_val_split(dataset_dir)
    update_data_yaml(yaml_path, dataset_dir)

    with get_session() as session:
        dataset = session.get(Dataset, dataset_id)
        dataset.status = DatasetStatus.PACKAGED

    logger.info(
        f"[{self.request.id}] Dataset {dataset_id} packaged — "
        f"train={train_count}  val={val_count}"
    )

    # Append into each relevant persistent merged dir
    total_appended = 0
    for model_type in ModelType:
        # Skip specialist models if that class wasn't detected in this clip
        if model_type != ModelType.GENERAL and model_type.value not in detected_types:
            continue
        n = _append_to_merged(dataset_dir, dataset_id, model_type)
        total_appended += n
        logger.info(
            f"[{self.request.id}] Appended dataset {dataset_id} → "
            f"merged/{model_type.value}  images={n}"
        )

    # Delete clip hash dir immediately — no longer needed
    shutil.rmtree(dataset_dir, ignore_errors=True)
    logger.info(f"[{self.request.id}] Deleted clip dataset dir: {dataset_dir.name}")

    return {
        "status": "packaged",
        "dataset_id": dataset_id,
        "train_count": train_count,
        "val_count": val_count,
        "total_appended": total_appended,
    }
