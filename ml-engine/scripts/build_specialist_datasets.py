"""
build_specialist_datasets.py

Merges pre-cleaned Kaggle datasets into specialist merged folders for training.
All source datasets are already canonical nc=3 (aircraft=0, vehicle=1, personnel=2).
Filenames are prefixed with the dataset owner to prevent cross-dataset collisions.

Usage:
    cd ml-engine && python scripts/build_specialist_datasets.py
    cd ml-engine && python scripts/build_specialist_datasets.py --models GENERAL
"""
import sys
import shutil
import logging
import argparse
from pathlib import Path
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("build_datasets")

REPO_ROOT = Path(__file__).resolve().parents[2]
ML_ENGINE_DIR = REPO_ROOT / "ml-engine"
sys.path.insert(0, str(ML_ENGINE_DIR))
sys.path.insert(0, str(REPO_ROOT))

from core.main import detect_dataset_structure
from config import settings

CANONICAL_CLASSES = ["aircraft", "vehicle", "personnel"]

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
    raise FileNotFoundError(f"Dataset not found locally: {handle}")


def build_model_dataset(model_name: str, out_root: Path) -> tuple[int, int]:
    handles = BASELINE_DATASETS[model_name]

    out_dir   = out_root / model_name
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
        prefix = handle.split("/")[0] + "_"
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

            copied = 0
            for img_src in Path(src_img).iterdir():
                if img_src.suffix.lower() not in IMG_EXTS:
                    continue
                dst_name = prefix + img_src.name
                shutil.copy2(img_src, dst_img_dir / dst_name)
                lbl_src = Path(src_lbl) / (img_src.stem + ".txt") if src_lbl else None
                dst_lbl = dst_lbl_dir / (prefix + img_src.stem + ".txt")
                if lbl_src and lbl_src.exists():
                    shutil.copy2(lbl_src, dst_lbl)
                else:
                    dst_lbl.write_text("")
                copied += 1

            split = "train" if src_img_key == "train_images" else "val"
            log.info(f"  [{model_name}] {handle} {split}: {copied}")
            if src_img_key == "train_images":
                total_train += copied
            else:
                total_val += copied

    (out_dir / "dataset.yaml").write_text(
        f"path: {out_dir.as_posix()}\n"
        f"train: train/images\n"
        f"val: val/images\n"
        f"nc: 3\n"
        f"names: {CANONICAL_CLASSES}\n"
    )

    log.info(f"[{model_name}] DONE — train={total_train}  val={total_val}")
    return total_train, total_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="+",
        choices=list(BASELINE_DATASETS.keys()),
        default=list(BASELINE_DATASETS.keys()),
    )
    args = parser.parse_args()

    out_root = settings.KAGGLE_CACHE_DIR / "merged"
    for model_name in args.models:
        log.info(f"=== Building {model_name} ===")
        build_model_dataset(out_root=out_root, model_name=model_name)


if __name__ == "__main__":
    main()
