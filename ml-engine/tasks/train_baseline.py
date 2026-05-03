"""
ml-engine/tasks/train_baseline.py

Celery task: Stage 1 (cold-start) training on pre-labeled Kaggle datasets.
No GDINO, no frame extraction — Kaggle data is already labeled.

All datasets are remapped to the canonical 3-class vocabulary before merging:
  0=AIRCRAFT  1=VEHICLE  2=PERSONNEL

  AIRCRAFT  — mihprofi + shakedlevnat + nzigulic + piterfm + rookieengg + rawsi18
  VEHICLE   — kiit-mita + nzigulic + piterfm + rawsi18 + amad-5
  PERSONNEL — kiit-mita + rawsi18 + amad-5
  GENERAL   — all eight datasets combined (trains after specialists)

Triggered by Admin "Train Baseline" in the web UI, which creates one
TrainingRun per model type and dispatches this task four times.
"""
import logging
import shutil
import yaml as _yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from celery_app import celery_app
from config import settings
from db.models import ModelType, TrainingRun, TrainingStage, TrainingStatus
from db.session import get_session

logger = logging.getLogger(__name__)

# ── Canonical 3-class vocabulary ─────────────────────────────────────────────
# Must align with GDINO_TEXT_PROMPT order in config.py.
CANONICAL_CLASSES = [
    "aircraft",   # 0 — drones, helicopters, fixed-wing, missiles
    "vehicle",    # 1 — tanks, APCs, artillery, radar, ground military vehicles
    "personnel",  # 2 — soldiers, fighters, RPG/ATGM operators
]
CANONICAL_NC = len(CANONICAL_CLASSES)  # 3

# Per-dataset remapping: old_class_id → canonical_id  (-1 = drop annotation)
#
# kiit-mita (nc=7): Artilary(0) Missile(1) Radar(2) M.RocketLauncher(3)
#                   Soldier(4) Tank(5) Vehicle(6)
# mihprofi (nc=2):  Dron(0) Dron2(1)
# shakedlevnat (nc=83): 83 specific aircraft types (all map to AIRCRAFT)
_SHAKED_AIRCRAFT = set(range(83))  # every type is an aircraft

DATASET_CLASS_MAPS: Dict[str, Dict[int, int]] = {
    "sudipchakrabarty/kiit-mita": {
        0: 1,   # Artilary      → vehicle
        1: 0,   # Missile       → aircraft
        2: 1,   # Radar         → vehicle
        3: 1,   # M.RocketLaun. → vehicle
        4: 2,   # Soldier       → personnel
        5: 1,   # Tank          → vehicle
        6: 1,   # Vehicle       → vehicle
    },
    "mihprofi/drone-detect": {
        0: 0,  # Dron  → aircraft
        1: 0,  # Dron2 → aircraft
    },
    "shakedlevnat/military-aircraft-database-prepared-for-yolo": {
        **{i: 0 for i in _SHAKED_AIRCRAFT},  # all 83 types → aircraft
    },
    # nzigulic/military-equipment (nc=11): visually identified via contact sheets
    # classes 0-3, 8-10 are ground vehicles; 4-7 are aircraft; no personnel
    "nzigulic/military-equipment": {
        0: 1,   # tanks/APCs (aerial + parade)     → vehicle
        1: 1,   # trucks/vehicles (top-down)        → vehicle
        2: 1,   # wheeled APC                       → vehicle
        3: 1,   # tanks (top-down)                  → vehicle
        4: 0,   # attack helicopters                → aircraft
        5: 0,   # transport helicopters (Chinook)   → aircraft
        6: 0,   # drones / small aircraft           → aircraft
        7: 0,   # fixed-wing transport              → aircraft
        8: 1,   # wheeled SPAA / armored vehicle    → vehicle
        9: 1,   # towed artillery                   → vehicle
        10: 1,  # vehicles (thermal/night-vision)   → vehicle
    },
    # piterfm/2022-ukraine-russia-war-equipment-losses-oryx: already nc=3 canonical — GDINO-labeled with category-aware prompts
    # no personnel (Oryx tracks equipment losses only)
    "piterfm/2022-ukraine-russia-war-equipment-losses-oryx": {0: 0, 1: 1, 2: 2},
    # rookieengg/military-aircraft-detection-dataset-yolo-format (nc=73): 73 specific aircraft types, all → aircraft
    "rookieengg/military-aircraft-detection-dataset-yolo-format": {i: 0 for i in range(73)},
    # rawsi18/military-assets-dataset-12-classes-yolo8-format (nc=12): mixed military assets
    "rawsi18/military-assets-dataset-12-classes-yolo8-format": {
        0: 2,   # camouflage_soldier → personnel
        1: -1,  # weapon             → drop
        2: 1,   # military_tank      → vehicle
        3: 1,   # military_truck     → vehicle
        4: 1,   # military_vehicle   → vehicle
        5: -1,  # civilian           → drop
        6: 2,   # soldier            → personnel
        7: -1,  # civilian_vehicle   → drop
        8: 1,   # military_artillery → vehicle
        9: -1,  # trench             → drop
        10: 0,  # military_aircraft  → aircraft
        11: -1, # military_warship   → drop
    },
    # rupankarmajumdar/amad-5-aerial-military-asset-detection-dataset (nc=5): aerial view, mixed
    "rupankarmajumdar/amad-5-aerial-military-asset-detection-dataset": {
        0: 1,   # military_tank    → vehicle
        1: 1,   # military_vehicle → vehicle
        2: 2,   # soldier          → personnel
        3: -1,  # civilian         → drop
        4: -1,  # civilian_vehicle → drop
    },
}

SPECIALIST_CLASS: Dict[ModelType, Optional[int]] = {
    ModelType.AIRCRAFT:  0,
    ModelType.VEHICLE:   1,
    ModelType.PERSONNEL: 2,
    ModelType.GENERAL:   None,
}

BASELINE_DATASETS: Dict[ModelType, List[str]] = {
    ModelType.AIRCRAFT: [
        "mihprofi/drone-detect",
        "shakedlevnat/military-aircraft-database-prepared-for-yolo",
        "nzigulic/military-equipment",
        "piterfm/2022-ukraine-russia-war-equipment-losses-oryx",
        "rookieengg/military-aircraft-detection-dataset-yolo-format",
        "rawsi18/military-assets-dataset-12-classes-yolo8-format",
    ],
    ModelType.VEHICLE: [
        "sudipchakrabarty/kiit-mita",
        "nzigulic/military-equipment",
        "piterfm/2022-ukraine-russia-war-equipment-losses-oryx",
        "rawsi18/military-assets-dataset-12-classes-yolo8-format",
        "rupankarmajumdar/amad-5-aerial-military-asset-detection-dataset",
    ],
    ModelType.PERSONNEL: [
        "sudipchakrabarty/kiit-mita",
        "rawsi18/military-assets-dataset-12-classes-yolo8-format",
        "rupankarmajumdar/amad-5-aerial-military-asset-detection-dataset",
    ],
    ModelType.GENERAL: [
        "mihprofi/drone-detect",
        "shakedlevnat/military-aircraft-database-prepared-for-yolo",
        "sudipchakrabarty/kiit-mita",
        "nzigulic/military-equipment",
        "piterfm/2022-ukraine-russia-war-equipment-losses-oryx",
        "rookieengg/military-aircraft-detection-dataset-yolo-format",
        "rawsi18/military-assets-dataset-12-classes-yolo8-format",
        "rupankarmajumdar/amad-5-aerial-military-asset-detection-dataset",
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
            old_id = int(float(parts[0]))  # handles "0.0" style labels (e.g. amad-5)
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


def _local_dataset_path(handle: str) -> Path:
    """Return the on-disk root for a pre-downloaded Kaggle dataset handle.

    Checks KAGGLE_CACHE_DIR first, then falls back to the kagglehub system cache
    (~/.cache/kagglehub/datasets/) so datasets downloaded via kagglehub CLI work
    without needing a manual copy.
    """
    owner, name = handle.split("/")
    _KAGGLEHUB_CACHE = Path.home() / ".cache" / "kagglehub" / "datasets"

    for search_root in (settings.KAGGLE_CACHE_DIR, _KAGGLEHUB_CACHE):
        base = search_root / owner / name / "versions"
        if not base.exists():
            continue
        versions = sorted(
            (d for d in base.iterdir() if d.is_dir()),
            key=lambda p: int(p.name) if p.name.isdigit() else 0,
        )
        if versions:
            return versions[-1]

    raise FileNotFoundError(
        f"Dataset not found locally: {handle}\n"
        f"Download it with: kagglehub.dataset_download('{handle}')"
    )


def _merge_datasets(
    dataset_handles: List[str],
    combined_dir: Path,
    task_id: str,
    specialist_class: Optional[int] = None,
) -> Tuple[Path, int, list]:
    """
    Merge pre-downloaded Kaggle datasets into one combined YOLO directory.
    Each dataset's labels are remapped to CANONICAL_CLASSES before copying.

    specialist_class: when set (0/1/2), only annotations mapping to that
    canonical ID are kept and remapped to 0 — producing a true nc=1
    specialist.  None → nc=3 GENERAL model (all classes kept as-is).

    Returns (yaml_path, nc, class_names).
    """
    from core.main import detect_dataset_structure

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

        # For specialist models: drop all annotations except the target class,
        # then remap that class to 0 (nc=1).
        if specialist_class is not None:
            class_map = {
                k: (0 if v == specialist_class else -1)
                for k, v in class_map.items()
            }

        logger.info(f"[{task_id}] Merging {handle}")
        dataset_path = str(_local_dataset_path(handle))
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

    if specialist_class is not None:
        nc    = 1
        names = [CANONICAL_CLASSES[specialist_class]]
    else:
        nc    = CANONICAL_NC
        names = CANONICAL_CLASSES

    logger.info(
        f"[{task_id}] Combined: {total_train} train  {total_val} val  "
        f"nc={nc}  names={names}"
    )

    yaml_path = combined_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        _yaml.dump(
            {
                "path":  str(combined_dir),
                "train": "train/images",
                "val":   "val/images",
                "nc":    nc,
                "names": names,
            },
            f,
            default_flow_style=False,
        )

    return yaml_path, nc, names, total_train


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

    specialist_class = SPECIALIST_CLASS[model_type]

    from core.main import train_model

    run_dir      = settings.RUNS_DIR / "baseline" / model_type.value
    run_name     = f"baseline_{model_type.value}_{training_run_id}"
    weights_path = run_dir / run_name / "weights" / "best.pt"

    combined_dir = settings.KAGGLE_CACHE_DIR / "combined" / model_type.value
    if combined_dir.exists():
        shutil.rmtree(combined_dir)

    try:
        yaml_path, nc, class_names, total_train = _merge_datasets(
            datasets, combined_dir, self.request.id,
            specialist_class=specialist_class,
        )
        with get_session() as session:
            run = session.get(TrainingRun, training_run_id)
            run.metrics = {"total_train_images": total_train}
        logger.info(
            f"[{self.request.id}] [{model_type.value}] "
            f"Training: nc={nc}  epochs={settings.YOLO_EPOCHS_BASELINE}  train_images={total_train}"
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
