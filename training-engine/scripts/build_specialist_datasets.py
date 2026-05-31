"""
build_specialist_datasets.py

Merges Kaggle source datasets into specialist merged folders for YOLO training.
Source files are NEVER modified — class remapping happens in memory during copy.
Labels are remapped to canonical nc=3: AIRCRAFT=0, VEHICLE=1, PERSONNEL=2.

Usage:
    cd training-engine && python scripts/build_specialist_datasets.py
    cd training-engine && python scripts/build_specialist_datasets.py --models AIRCRAFT
"""
import sys
import shutil
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("build_datasets")

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ENGINE_DIR = REPO_ROOT / "training-engine"
sys.path.insert(0, str(TRAINING_ENGINE_DIR))
sys.path.insert(0, str(REPO_ROOT))

from core.main import detect_dataset_structure
from config import settings

CANONICAL_CLASSES = ["aircraft", "vehicle", "personnel"]

# Required class (after remapping) for image to be included in a specialist model.
# None = include everything (GENERAL).
SPECIALIST_REQUIRED_CLASS: Dict[str, Optional[int]] = {
    "AIRCRAFT":  0,
    "VEHICLE":   1,
    "PERSONNEL": 2,
    "GENERAL":   None,
}

# Per-dataset class remapping: {src_id: canonical_id}.
# IDs absent from the dict are SKIPPED (annotation line dropped).
# None = pass-through (already canonical nc=3).
DATASET_CLASS_MAP: Dict[str, Optional[Dict[int, int]]] = {
    "mihprofi/drone-detect": {
        0: 0,  # Dron → AIRCRAFT
        1: 0,  # Dron2 → AIRCRAFT
    },
    "shakedlevnat/military-aircraft-database-prepared-for-yolo": {
        i: 0 for i in range(83)  # all aircraft model names → AIRCRAFT
    },
    "nzigulic/military-equipment": {
        # anonymous nc=11; visually identified (commit 6d1d8d5)
        0: 1, 1: 1, 2: 1, 3: 1,  # ground vehicles → VEHICLE
        4: 0, 5: 0, 6: 0, 7: 0,  # aerial → AIRCRAFT
        8: 1, 9: 1, 10: 1,        # ground vehicles → VEHICLE
    },
    "sudipchakrabarty/kiit-mita": {
        0: 1,  # Artilary → VEHICLE
        1: 0,  # Missile → AIRCRAFT
        2: 1,  # Radar → VEHICLE
        3: 1,  # M. Rocket Launcher → VEHICLE
        4: 2,  # Soldier → PERSONNEL
        5: 1,  # Tank → VEHICLE
        6: 1,  # Vehicle → VEHICLE
    },
    "rookieengg/military-aircraft-detection-dataset-yolo-format": {
        i: 0 for i in range(43)  # all aircraft model names → AIRCRAFT
    },
    "rawsi18/military-assets-dataset-12-classes-yolo8-format": {
        0: 2,   # camouflage_soldier → PERSONNEL
        # 1 weapon → skip
        2: 1,   # military_tank → VEHICLE
        3: 1,   # military_truck → VEHICLE
        4: 1,   # military_vehicle → VEHICLE
        # 5 civilian → skip
        6: 2,   # soldier → PERSONNEL
        # 7 civilian_vehicle → skip
        8: 1,   # military_artillery → VEHICLE
        # 9 trench → skip
        10: 0,  # military_aircraft → AIRCRAFT
        # 11 military_warship → skip
    },
    "rupankarmajumdar/amad-5-aerial-military-asset-detection-dataset": {
        0: 1,  # military_tank → VEHICLE
        1: 1,  # military_vehicle → VEHICLE
        # 2 civilian → skip
        3: 2,  # soldier → PERSONNEL
        # 4 civilian_vehicle → skip
    },
    "piterfm/2022-ukraine-russia-war-equipment-losses-oryx": None,  # canonical nc=3
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


def _make_split_links(parent: Path, splits: Dict[str, tuple]) -> None:
    """Create train/ and val/ dirs with images/ and labels/ symlinks pointing to actual dirs."""
    for split_name, (img_src, lbl_src) in splits.items():
        split_dir = parent / split_name
        split_dir.mkdir(exist_ok=True)
        for link_name, src in (("images", img_src), ("labels", lbl_src)):
            link = split_dir / link_name
            if not link.exists() and src is not None and Path(src).exists():
                link.symlink_to(Path(src).resolve())


def _normalize_dataset_structure(dataset_path: Path) -> None:
    """
    Detect non-standard dataset layouts and create standard train/val symlinks.
    Handles two patterns found in our Kaggle datasets:

    Pattern A — flat split dirs (nzigulic):
      <root>/images_train/  labels_train/  images_val/  labels_val/

    Pattern B — nested split subdirs (rookieengg):
      <root>/images/<X>_train/  labels/<X>_train/
             images/<X>_val/    labels/<X>_val/
    """
    # Pattern A: images_train / labels_train at any depth
    for img_train in dataset_path.rglob("images_train"):
        if not img_train.is_dir():
            continue
        parent = img_train.parent
        _make_split_links(parent, {
            "train": (parent / "images_train", parent / "labels_train"),
            "val":   (parent / "images_val",   parent / "labels_val"),
        })
        log.info(f"  Normalized pattern-A structure in {parent.relative_to(dataset_path)}")
        return

    # Pattern B: images/<X>_train + labels/<X>_train nested dirs
    for images_dir in dataset_path.rglob("images"):
        if not images_dir.is_dir():
            continue
        labels_dir = images_dir.parent / "labels"
        if not labels_dir.is_dir():
            continue
        subdirs = {d.name.lower(): d for d in images_dir.iterdir() if d.is_dir()}
        train_img = next((d for k, d in subdirs.items() if "train" in k), None)
        val_img   = next((d for k, d in subdirs.items() if "val" in k), None)
        if train_img and val_img:
            parent = images_dir.parent
            _make_split_links(parent, {
                "train": (train_img, labels_dir / train_img.name),
                "val":   (val_img,   labels_dir / val_img.name),
            })
            log.info(f"  Normalized pattern-B structure in {parent.relative_to(dataset_path)}")
            return


def _local_dataset_path(handle: str) -> Path:
    owner, name = handle.split("/")
    for search_root in (settings.KAGGLE_CACHE_DIR / "imported", settings.KAGGLE_CACHE_DIR, _KAGGLEHUB_CACHE):
        base = search_root / owner / name / "versions"
        if not base.exists():
            continue
        versions = sorted(
            (d for d in base.iterdir() if d.is_dir()),
            key=lambda p: int(p.name) if p.name.isdigit() else 0,
        )
        if versions:
            return versions[-1]
    raise FileNotFoundError(f"Dataset not found locally: {handle}")


def _remap_lines(text: str, class_map: Optional[Dict[int, int]]) -> Tuple[str, set]:
    """Remap label text in memory. Returns (remapped_text, set_of_canonical_class_ids)."""
    if class_map is None:
        classes = {int(float(line.split()[0])) for line in text.splitlines() if line.strip()}
        return text, classes
    kept = []
    classes = set()
    for line in text.splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        dst_cls = class_map.get(int(float(parts[0])), -1)
        if dst_cls >= 0:
            kept.append(f"{dst_cls} {' '.join(parts[1:])}")
            classes.add(dst_cls)
    return ("\n".join(kept) + ("\n" if kept else "")), classes


def build_all_models(out_root: Path, models: List[str]) -> None:
    """
    Single-pass build: process each source dataset once, write to all relevant
    model dirs simultaneously. ~4x faster than building each model separately.
    """
    # Set up output dirs for all requested models
    dirs: Dict[str, Dict] = {}
    for model_name in models:
        out_dir = out_root / model_name
        if out_dir.exists():
            shutil.rmtree(out_dir)
        d = {}
        for split in ("train", "val"):
            (out_dir / split / "images").mkdir(parents=True, exist_ok=True)
            (out_dir / split / "labels").mkdir(parents=True, exist_ok=True)
            d[f"{split}_img"] = out_dir / split / "images"
            d[f"{split}_lbl"] = out_dir / split / "labels"
        d["required_class"] = SPECIALIST_REQUIRED_CLASS[model_name]
        d["train"] = d["val"] = 0
        dirs[model_name] = d

    # Collect all unique handles and which models use each
    all_handles: Dict[str, List[str]] = {}
    for model_name in models:
        for handle in BASELINE_DATASETS[model_name]:
            all_handles.setdefault(handle, []).append(model_name)

    # Process each source dataset exactly once
    for handle, model_names in all_handles.items():
        prefix = handle.split("/")[0] + "_"
        class_map = DATASET_CLASS_MAP.get(handle)

        try:
            dataset_path = _local_dataset_path(handle)
        except FileNotFoundError as e:
            log.error(str(e))
            continue

        _normalize_dataset_structure(dataset_path)
        paths, _ = detect_dataset_structure(str(dataset_path))
        if not paths:
            log.warning(f"No train/val structure in {handle} — skipping")
            continue

        for src_img_key, src_lbl_key, split in [
            ("train_images", "train_labels", "train"),
            ("val_images",   "val_labels",   "val"),
        ]:
            src_img = paths.get(src_img_key)
            src_lbl = paths.get(src_lbl_key)
            if not src_img:
                continue

            counters: Dict[str, Dict] = {m: {"copied": 0, "skipped": 0} for m in model_names}

            for img_src in Path(src_img).iterdir():
                if img_src.suffix.lower() not in IMG_EXTS:
                    continue

                lbl_src = Path(src_lbl) / (img_src.stem + ".txt") if src_lbl else None
                raw_text = lbl_src.read_text() if (lbl_src and lbl_src.exists()) else ""
                # Remap labels once, shared across all models
                remapped_text, classes_present = _remap_lines(raw_text, class_map)

                dst_name = prefix + img_src.name
                img_copied = False

                for model_name in model_names:
                    d = dirs[model_name]
                    required_class = d["required_class"]

                    if required_class is not None:
                        filtered = [l for l in remapped_text.splitlines() if l.strip() and int(l.split()[0]) == required_class]
                        label_text = "\n".join(filtered) + ("\n" if filtered else "")
                        if not filtered:
                            counters[model_name]["skipped"] += 1
                            continue
                    else:
                        label_text = remapped_text

                    # Copy image once (first model to need it), symlink for the rest
                    dst_img = d[f"{split}_img"] / dst_name
                    if not img_copied:
                        shutil.copy2(img_src, dst_img)
                        img_copied = True
                    else:
                        shutil.copy2(img_src, dst_img)

                    (d[f"{split}_lbl"] / (prefix + img_src.stem + ".txt")).write_text(label_text)
                    counters[model_name]["copied"] += 1

            for model_name in model_names:
                c = counters[model_name]
                log.info(f"  [{model_name}] {handle} {split}: {c['copied']} copied, {c['skipped']} skipped")
                dirs[model_name][split] += counters[model_name]["copied"]

    # Write dataset.yaml for each model
    for model_name in models:
        out_dir = out_root / model_name
        (out_dir / "dataset.yaml").write_text(
            f"path: {out_dir.as_posix()}\n"
            f"train: train/images\n"
            f"val: val/images\n"
            f"nc: 3\n"
            f"names: {CANONICAL_CLASSES}\n"
        )
        d = dirs[model_name]
        log.info(f"[{model_name}] DONE — train={d['train']}  val={d['val']}")


def verify(out_root: Path, models: List[str]) -> None:
    log.info("=== Verification ===")
    for model_name in models:
        out_dir = out_root / model_name
        for split in ("train", "val"):
            img_dir = out_dir / split / "images"
            lbl_dir = out_dir / split / "labels"
            img_count = sum(1 for f in img_dir.iterdir() if f.suffix.lower() in IMG_EXTS) if img_dir.exists() else 0
            bad_classes: set = set()
            for lbl in (lbl_dir.iterdir() if lbl_dir.exists() else []):
                if lbl.suffix != ".txt":
                    continue
                for line in lbl.read_text().splitlines():
                    parts = line.strip().split()
                    if parts:
                        cls = int(float(parts[0]))
                        if cls not in (0, 1, 2):
                            bad_classes.add(cls)
            status = "FAIL — bad classes: " + str(bad_classes) if bad_classes else "OK"
            log.info(f"  [{model_name}] {split}: {img_count} images — {status}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="+",
        choices=list(BASELINE_DATASETS.keys()),
        default=list(BASELINE_DATASETS.keys()),
    )
    args = parser.parse_args()

    out_root = settings.KAGGLE_CACHE_DIR / "merged"
    log.info(f"Building models: {args.models} (single-pass)")
    build_all_models(out_root=out_root, models=args.models)

    verify(out_root, args.models)


if __name__ == "__main__":
    main()
