"""
Download the 3 new Kaggle datasets into the project's KAGGLE_CACHE_DIR.

rawsi18 is already in the kagglehub system cache — this script just downloads
rupankarmajumdar/amad-5 and rookieengg (the previous download was corrupted).

Run from ml-engine/: python scripts/download_new_datasets.py
"""
import shutil
import sys
from pathlib import Path

import kagglehub

KAGGLE_CACHE_DIR = Path(__file__).parent.parent / "media" / "kaggle_datasets"
KAGGLEHUB_CACHE  = Path.home() / ".cache" / "kagglehub" / "datasets"

NEW_DATASETS = [
    "rookieengg/military-aircraft-detection-dataset-yolo-format",
    "rupankarmajumdar/amad-5-aerial-military-asset-detection-dataset",
    "rawsi18/military-assets-dataset-12-classes-yolo8-format",
]


def _copy_to_project(handle: str, src: Path) -> None:
    owner, name = handle.split("/")
    dst = KAGGLE_CACHE_DIR / owner / name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    print(f"  copied to {dst}")


def main() -> None:
    for handle in NEW_DATASETS:
        owner, name = handle.split("/")
        project_base = KAGGLE_CACHE_DIR / owner / name / "versions"

        # Already in project dir with actual images — skip
        if project_base.exists():
            versions = [d for d in project_base.iterdir() if d.is_dir()]
            if versions:
                img_count = sum(1 for _ in versions[-1].rglob("*.jpg")) + \
                            sum(1 for _ in versions[-1].rglob("*.png"))
                if img_count > 0:
                    print(f"[skip] {handle} already in project cache ({img_count} images)")
                    continue

        # Already in kagglehub system cache — just copy
        hub_base = KAGGLEHUB_CACHE / owner / name / "versions"
        if hub_base.exists():
            versions = sorted(
                (d for d in hub_base.iterdir() if d.is_dir()),
                key=lambda p: int(p.name) if p.name.isdigit() else 0,
            )
            if versions:
                print(f"[copy]     {handle} from kagglehub cache")
                _copy_to_project(handle, KAGGLEHUB_CACHE / owner / name)
                continue

        # Download fresh via kagglehub
        print(f"[download] {handle}")
        try:
            path = kagglehub.dataset_download(handle)
            print(f"  downloaded to {path}")
            src_root = KAGGLEHUB_CACHE / owner / name
            if src_root.exists():
                _copy_to_project(handle, src_root)
        except Exception as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
