"""
build_specialist_datasets.py

ONE-TIME offline script: merge raw Kaggle datasets into clean, filtered
specialist directories. Run this once; all future training reads from the
output folders directly.

Output:
    ml-engine/media/kaggle_datasets/merged/<MODEL>/
        train/images/   train/labels/
        val/images/     val/labels/
        dataset.yaml

THE BUG THIS FIXES
------------------
The old on-the-fly _merge_datasets() copied every image unconditionally, then
remapped labels.  Images whose every annotation remapped to -1 got an empty
label file — which YOLOv8 treats as a background image.  For PERSONNEL, ~90%
of source images are vehicles, so the training set was flooded with falsely-
labelled background images, destroying convergence.

FIX: remap labels first; skip the image entirely if zero annotations survive.

Usage (from repo root):
    cd ml-engine && python scripts/build_specialist_datasets.py
    cd ml-engine && python scripts/build_specialist_datasets.py --models PERSONNEL VEHICLE
"""
import sys
import shutil
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("build_datasets")

REPO_ROOT = Path(__file__).resolve().parents[2]
ML_ENGINE_DIR = REPO_ROOT / "ml-engine"
sys.path.insert(0, str(ML_ENGINE_DIR))
sys.path.insert(0, str(REPO_ROOT))

from core.main import detect_dataset_structure
from config import settings

# ── Canonical vocabulary ──────────────────────────────────────────────────────
CANONICAL_CLASSES = ["aircraft", "vehicle", "personnel"]

SPECIALIST_CLASS: Dict[str, Optional[int]] = {
    "AIRCRAFT":  0,
    "VEHICLE":   1,
    "PERSONNEL": 2,
    "GENERAL":   None,
}

# ── Per-dataset class remapping ───────────────────────────────────────────────
# old_class_id → canonical_id   (-1 = drop annotation)

_SHAKED_AIRCRAFT = set(range(83))

DATASET_CLASS_MAPS: Dict[str, Dict[int, int]] = {
    # kiit-mita (nc=7): Artilary(0) Missile(1) Radar(2) M.RocketLauncher(3)
    #                   Soldier(4) Tank(5) Vehicle(6)
    "sudipchakrabarty/kiit-mita": {
        0: 1,   # Artillery      → vehicle
        1: 0,   # Missile        → aircraft
        2: 1,   # Radar          → vehicle
        3: 1,   # M.RocketLaun.  → vehicle
        4: 2,   # Soldier        → personnel
        5: 1,   # Tank           → vehicle
        6: 1,   # Vehicle        → vehicle
    },
    # mihprofi (nc=2): Dron(0) Dron2(1)
    "mihprofi/drone-detect": {
        0: 0,
        1: 0,
    },
    # shakedlevnat (nc=83): 83 specific aircraft types, all → aircraft
    "shakedlevnat/military-aircraft-database-prepared-for-yolo": {
        **{i: 0 for i in _SHAKED_AIRCRAFT},
    },
    # nzigulic (nc=11): 0-3,8-10 → vehicle; 4-7 → aircraft; no personnel
    "nzigulic/military-equipment": {
        0: 1, 1: 1, 2: 1, 3: 1,   # tanks/APCs/trucks
        4: 0, 5: 0, 6: 0, 7: 0,   # helicopters/drones/fixed-wing
        8: 1, 9: 1, 10: 1,         # SPAA/artillery/thermal vehicles
    },
    # piterfm: already nc=3 canonical (GDINO-labeled)
    "piterfm/2022-ukraine-russia-war-equipment-losses-oryx": {0: 0, 1: 1, 2: 2},
    # rookieengg (nc=73): all aircraft types → aircraft
    "rookieengg/military-aircraft-detection-dataset-yolo-format": {i: 0 for i in range(73)},
    # rawsi18 (nc=12): mixed military assets
    "rawsi18/military-assets-dataset-12-classes-yolo8-format": {
        0: 2,    # camouflage_soldier → personnel
        1: -1,   # weapon             → drop
        2: 1,    # military_tank      → vehicle
        3: 1,    # military_truck     → vehicle
        4: 1,    # military_vehicle   → vehicle
        5: -1,   # civilian           → drop
        6: 2,    # soldier            → personnel
        7: -1,   # civilian_vehicle   → drop
        8: 1,    # military_artillery → vehicle
        9: -1,   # trench             → drop
        10: 0,   # military_aircraft  → aircraft
        11: -1,  # military_warship   → drop
    },
    # amad-5: remapped in-place to canonical nc=3 (civilian/civilian_vehicle deleted)
    "rupankarmajumdar/amad-5-aerial-military-asset-detection-dataset": {0: 0, 1: 1, 2: 2},
}

BASELINE_DATASETS: Dict[str, List[str]] = {
    "AIRCRAFT": [
        "mihprofi/drone-detect",
        "shakedlevnat/military-aircraft-database-prepared-for-yolo",
        "nzigulic/military-equipment",
        "piterfm/2022-ukraine-russia-war-equipment-losses-oryx",
        "rookieengg/military-aircraft-detection-dataset-yolo-format",
        "rawsi18/military-assets-dataset-12-classes-yolo8-format",
    ],
    "VEHICLE": [
        "sudipchakrabarty/kiit-mita",
        "nzigulic/military-equipment",
        "piterfm/2022-ukraine-russia-war-equipment-losses-oryx",
        "rawsi18/military-assets-dataset-12-classes-yolo8-format",
        "rupankarmajumdar/amad-5-aerial-military-asset-detection-dataset",
    ],
    "PERSONNEL": [
        "sudipchakrabarty/kiit-mita",
        "rawsi18/military-assets-dataset-12-classes-yolo8-format",
        "rupankarmajumdar/amad-5-aerial-military-asset-detection-dataset",
    ],
    "GENERAL": [
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

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

_KAGGLEHUB_CACHE = Path.home() / ".cache" / "kagglehub" / "datasets"


def _local_dataset_path(handle: str) -> Path:
    owner, name = handle.split("/")
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
        f"Download with: kagglehub.dataset_download('{handle}')"
    )


def _remap_annotations(lbl_path: Path, class_map: Dict[int, int]) -> List[str]:
    """Remap label file lines. Returns only lines that survive (new_id != -1)."""
    if not lbl_path.exists():
        return []
    lines_out = []
    try:
        for line in lbl_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            old_id = int(float(parts[0]))  # handles "0.0" style (e.g. amad-5)
            new_id = class_map.get(old_id, -1)
            if new_id == -1:
                continue
            lines_out.append(f"{new_id} {' '.join(parts[1:])}")
    except Exception as exc:
        log.warning(f"Could not remap {lbl_path}: {exc}")
    return lines_out


def build_model_dataset(model_name: str, out_root: Path) -> tuple[int, int]:
    """Build merged dataset for one model. Returns (train_count, val_count)."""
    handles = BASELINE_DATASETS[model_name]
    specialist_class = SPECIALIST_CLASS[model_name]

    out_dir  = out_root / model_name
    train_img = out_dir / "train" / "images"
    train_lbl = out_dir / "train" / "labels"
    val_img   = out_dir / "val"   / "images"
    val_lbl   = out_dir / "val"   / "labels"

    if out_dir.exists():
        shutil.rmtree(out_dir)
    for d in (train_img, train_lbl, val_img, val_lbl):
        d.mkdir(parents=True, exist_ok=True)

    total_train = total_val = 0

    for handle in handles:
        base_map = DATASET_CLASS_MAPS[handle]

        # Specialist: keep only the target class (remapped to 0); drop everything else.
        # GENERAL: keep all three canonical classes as-is.
        if specialist_class is not None:
            class_map = {k: (0 if v == specialist_class else -1) for k, v in base_map.items()}
        else:
            class_map = base_map

        dataset_path = str(_local_dataset_path(handle))
        paths, _ = detect_dataset_structure(dataset_path)
        if not paths:
            log.warning(f"No train/val structure in {handle} — skipping")
            continue

        for src_img_key, src_lbl_key, dst_img_dir, dst_lbl_dir in [
            ("train_images", "train_labels", train_img, train_lbl),
            ("val_images",   "val_labels",   val_img,   val_lbl),
        ]:
            src_img = paths.get(src_img_key)
            src_lbl = paths.get(src_lbl_key)
            if not src_img:
                continue

            copied = skipped = 0
            for img_src in Path(src_img).iterdir():
                if img_src.suffix.lower() not in IMG_EXTS:
                    continue

                lbl_src = Path(src_lbl) / (img_src.stem + ".txt") if src_lbl else None
                ann_lines = _remap_annotations(lbl_src, class_map) if lbl_src else []

                # THE KEY FIX: skip images with no surviving annotations.
                # An empty label file = background image in YOLO. A tank photo
                # must never appear as background in PERSONNEL training.
                if specialist_class is not None and not ann_lines:
                    skipped += 1
                    continue

                shutil.copy2(img_src, dst_img_dir / img_src.name)
                dst_lbl = dst_lbl_dir / (img_src.stem + ".txt")
                dst_lbl.write_text("\n".join(ann_lines) + ("\n" if ann_lines else ""))
                copied += 1

            split = "train" if src_img_key == "train_images" else "val"
            log.info(f"  [{model_name}] {handle} {split}: copied={copied}  skipped={skipped}")
            if src_img_key == "train_images":
                total_train += copied
            else:
                total_val += copied

    nc    = 1 if specialist_class is not None else len(CANONICAL_CLASSES)
    names = [CANONICAL_CLASSES[specialist_class]] if specialist_class is not None else CANONICAL_CLASSES

    yaml_path = out_dir / "dataset.yaml"
    yaml_path.write_text(
        f"path: {out_dir.as_posix()}\n"
        f"train: train/images\n"
        f"val: val/images\n"
        f"nc: {nc}\n"
        f"names: {names}\n"
    )

    log.info(f"[{model_name}] DONE — train={total_train}  val={total_val}  yaml={yaml_path}")
    return total_train, total_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="+",
        choices=list(BASELINE_DATASETS.keys()),
        default=list(BASELINE_DATASETS.keys()),
        help="Which datasets to build (default: all)",
    )
    args = parser.parse_args()

    out_root = settings.KAGGLE_CACHE_DIR / "merged"
    log.info(f"Output root: {out_root}")

    for model_name in args.models:
        log.info(f"=== Building {model_name} ===")
        train_n, val_n = build_model_dataset(model_name, out_root)
        log.info(f"=== {model_name}: {train_n} train / {val_n} val ===\n")


if __name__ == "__main__":
    main()
