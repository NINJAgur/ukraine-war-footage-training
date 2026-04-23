"""
ml-engine/tasks/train_baseline.py

Celery task: Stage 1 (cold-start) training on pre-labeled Kaggle datasets.
No GDINO, no frame extraction — Kaggle data is already labeled.

All datasets are remapped to our canonical 8-class vocabulary before merging:
  0=soldier  1=tank  2=armored vehicle  3=military vehicle
  4=artillery  5=aircraft  6=helicopter  7=drone

  GENERAL  — kiit-mita + nzigulic/military-equipment
  SOLDIER  — kiit-mita
  VEHICLE  — kiit-mita + nzigulic/military-equipment
  AIRCRAFT — mihprofi/drone-detect + shakedlevnat/military-aircraft-...

Triggered by Admin "Train Baseline" in the web UI, which creates one
TrainingRun per model type and dispatches this task four times.
"""
import logging
import os
import shutil
import sys
import yaml as _yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from celery_app import celery_app
from config import settings
from db.models import ModelType, TrainingRun, TrainingStage, TrainingStatus
from db.session import get_session

sys.path.insert(0, str(Path(__file__).parent.parent / "core"))

logger = logging.getLogger(__name__)

# ── Canonical class vocabulary ────────────────────────────────────────────────
# Must match GDINO prompt order defined in config.py.
CANONICAL_CLASSES = [
    "soldier",          # 0
    "tank",             # 1
    "armored vehicle",  # 2
    "military vehicle", # 3
    "artillery",        # 4
    "aircraft",         # 5
    "helicopter",       # 6
    "drone",            # 7
]
CANONICAL_NC = len(CANONICAL_CLASSES)  # 8

# Per-dataset remapping: old_class_id → canonical_id  (-1 = drop annotation)
#
# kiit-mita (nc=7): Artilary(0) Missile(1) Radar(2) M.RocketLauncher(3)
#                   Soldier(4) Tank(5) Vehicle(6)
# nzigulic (nc=11): class_0..class_10 — original class names unknown;
#                   all dropped until the mapping is researched.
# mihprofi (nc=2):  Dron(0) Dron2(1) — both are drones
# shakedlevnat (nc=83): 83 specific aircraft types;
#                   helicopters→6, known drones→7, everything else→5
_SHAKED_HELICOPTERS = {0, 1, 43, 44, 46, 60, 62, 66, 73, 76, 78, 82}
_SHAKED_DRONES      = {41, 49, 56, 64, 72}

DATASET_CLASS_MAPS: Dict[str, Dict[int, int]] = {
    "sudipchakrabarty/kiit-mita": {
        0: 4,   # Artilary      → artillery
        1: -1,  # Missile       → drop
        2: -1,  # Radar         → drop
        3: 4,   # M.RocketLaun. → artillery
        4: 0,   # Soldier       → soldier
        5: 1,   # Tank          → tank
        6: 3,   # Vehicle       → military vehicle
    },
    "nzigulic/military-equipment": {
        # Class names unknown — drop all until mapping is established
        **{i: -1 for i in range(11)},
    },
    "mihprofi/drone-detect": {
        0: 7,  # Dron  → drone
        1: 7,  # Dron2 → drone
    },
    "shakedlevnat/military-aircraft-database-prepared-for-yolo": {
        **{i: 6 for i in _SHAKED_HELICOPTERS},
        **{i: 7 for i in _SHAKED_DRONES},
        **{i: 5 for i in range(83)
           if i not in _SHAKED_HELICOPTERS and i not in _SHAKED_DRONES},
    },
}

# Pre-labeled Kaggle datasets per model type.
# Each entry: (dataset_handle,)  — class remapping applied via DATASET_CLASS_MAPS.
BASELINE_DATASETS: Dict[ModelType, List[str]] = {
    ModelType.GENERAL: [
        "sudipchakrabarty/kiit-mita",
        "nzigulic/military-equipment",
    ],
    ModelType.SOLDIER: [
        "sudipchakrabarty/kiit-mita",
    ],
    ModelType.VEHICLE: [
        "sudipchakrabarty/kiit-mita",
        "nzigulic/military-equipment",
    ],
    ModelType.AIRCRAFT: [
        "mihprofi/drone-detect",
        "shakedlevnat/military-aircraft-database-prepared-for-yolo",
    ],
}


def _extract_metrics(results) -> dict:
    try:
        return dict(results.results_dict)
    except Exception:
        return {}


def _remap_label_file(src: Path, dst: Path, class_map: Dict[int, int]) -> int:
    """
    Rewrite a YOLO label file applying class_map.
    Lines whose class remaps to -1 are dropped.
    Returns number of annotations written.
    """
    written = 0
    lines_out = []
    try:
        for line in src.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            old_id = int(parts[0])
            new_id = class_map.get(old_id, -1)
            if new_id == -1:
                continue
            lines_out.append(f"{new_id} {' '.join(parts[1:])}")
            written += 1
    except Exception as exc:
        logger.warning(f"Could not remap {src}: {exc}")
        return 0
    dst.write_text("\n".join(lines_out) + ("\n" if lines_out else ""))
    return written


def _merge_datasets(
    dataset_handles: List[str],
    combined_dir: Path,
    task_id: str,
) -> Tuple[Path, int, list]:
    """
    Download and merge multiple Kaggle datasets into one combined YOLO directory.
    Each dataset's labels are remapped to CANONICAL_CLASSES before copying.

    Returns (yaml_path, CANONICAL_NC, CANONICAL_CLASSES).
    """
    from main import download_dataset, detect_dataset_structure

    combined_dir.mkdir(parents=True, exist_ok=True)
    train_img = combined_dir / "train" / "images"
    train_lbl = combined_dir / "train" / "labels"
    val_img   = combined_dir / "val"   / "images"
    val_lbl   = combined_dir / "val"   / "labels"
    for d in (train_img, train_lbl, val_img, val_lbl):
        d.mkdir(parents=True, exist_ok=True)

    total_train = total_val = 0

    for handle in dataset_handles:
        class_map = DATASET_CLASS_MAPS.get(handle)
        if class_map is None:
            raise ValueError(f"No DATASET_CLASS_MAPS entry for '{handle}' — add one before training")

        logger.info(f"[{task_id}] Merging {handle}")
        dataset_path = download_dataset(handle)
        paths, dataset_path = detect_dataset_structure(dataset_path)
        if not paths:
            raise ValueError(f"No train/val structure found in {handle}")

        for src_img_key, src_lbl_key, dst_img_dir, dst_lbl_dir in [
            ("train_images", "train_labels", train_img, train_lbl),
            ("val_images",   "val_labels",   val_img,   val_lbl),
        ]:
            src_img = paths.get(src_img_key)
            src_lbl = paths.get(src_lbl_key)
            if not src_img:
                continue
            copied = annots = 0
            for img_src in Path(src_img).glob("*"):
                shutil.copy2(img_src, dst_img_dir / img_src.name)
                copied += 1
                if src_lbl:
                    lbl_src = Path(src_lbl) / (img_src.stem + ".txt")
                    if lbl_src.exists():
                        annots += _remap_label_file(
                            lbl_src, dst_lbl_dir / lbl_src.name, class_map
                        )
            if src_img_key == "train_images":
                total_train += copied
            else:
                total_val += copied
            logger.info(
                f"[{task_id}]   {handle} {src_img_key}: "
                f"{copied} images  {annots} annotations after remap"
            )

        logger.info(f"[{task_id}] Done merging {handle}")

    logger.info(
        f"[{task_id}] Combined: {total_train} train  {total_val} val  "
        f"nc={CANONICAL_NC}  names={CANONICAL_CLASSES}"
    )

    yaml_path = combined_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        _yaml.dump(
            {
                "path":  str(combined_dir),
                "train": "train/images",
                "val":   "val/images",
                "nc":    CANONICAL_NC,
                "names": CANONICAL_CLASSES,
            },
            f,
            default_flow_style=False,
        )

    return yaml_path, CANONICAL_NC, CANONICAL_CLASSES


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
    remap to canonical 8 classes, train YOLOv8m, save best.pt.
    Idempotent via DB status check.
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

    os.environ["KAGGLEHUB_CACHE"] = str(settings.KAGGLE_CACHE_DIR)
    logger.info(f"[{self.request.id}] Kaggle cache → {settings.KAGGLE_CACHE_DIR}")

    from main import train_model

    run_dir      = settings.RUNS_DIR / "baseline" / model_type.value
    run_name     = f"baseline_{model_type.value}_{training_run_id}"
    weights_path = run_dir / run_name / "weights" / "best.pt"

    combined_dir = settings.KAGGLE_CACHE_DIR / "combined" / model_type.value
    if combined_dir.exists():
        shutil.rmtree(combined_dir)

    try:
        yaml_path, nc, class_names = _merge_datasets(
            datasets, combined_dir, self.request.id
        )
        logger.info(
            f"[{self.request.id}] [{model_type.value}] "
            f"Training: nc={nc}  epochs={settings.YOLO_EPOCHS_BASELINE}"
        )
        results = train_model(
            yaml_path=str(yaml_path),
            epochs=settings.YOLO_EPOCHS_BASELINE,
            imgsz=settings.YOLO_IMG_SIZE,
            batch=settings.YOLO_BATCH_SIZE,
            device=settings.GPU_DEVICE,
            project=str(run_dir),
            name=run_name,
            weights=None,
            resume=False,
        )
        all_metrics = _extract_metrics(results)

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
