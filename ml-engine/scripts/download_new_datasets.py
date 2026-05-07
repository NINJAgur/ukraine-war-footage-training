"""
Download one or more Kaggle datasets into the project's KAGGLE_CACHE_DIR.

Skips datasets already present with images. Copies from kagglehub system cache
if available, otherwise downloads fresh.

Usage:
    python scripts/download_new_datasets.py owner/dataset-name [owner/dataset-name ...]
    python scripts/download_new_datasets.py --list   # show all datasets in build_specialist_datasets.py
"""
import shutil
import sys
from pathlib import Path

import kagglehub

KAGGLE_CACHE_DIR = Path(__file__).parent.parent / "media" / "kaggle_datasets"
KAGGLEHUB_CACHE  = Path.home() / ".cache" / "kagglehub" / "datasets"


def _copy_to_project(handle: str, src: Path) -> None:
    owner, name = handle.split("/")
    dst = KAGGLE_CACHE_DIR / owner / name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    print(f"  copied to {dst}")


def download(handle: str) -> None:
    owner, name = handle.split("/")
    project_base = KAGGLE_CACHE_DIR / owner / name / "versions"

    if project_base.exists():
        versions = [d for d in project_base.iterdir() if d.is_dir()]
        if versions:
            img_count = sum(1 for _ in versions[-1].rglob("*.jpg")) + \
                        sum(1 for _ in versions[-1].rglob("*.png"))
            if img_count > 0:
                print(f"[skip]     {handle} already in project cache ({img_count} images)")
                return

    hub_base = KAGGLEHUB_CACHE / owner / name / "versions"
    if hub_base.exists():
        versions = sorted(
            (d for d in hub_base.iterdir() if d.is_dir()),
            key=lambda p: int(p.name) if p.name.isdigit() else 0,
        )
        if versions:
            print(f"[copy]     {handle} from kagglehub cache")
            _copy_to_project(handle, KAGGLEHUB_CACHE / owner / name)
            return

    print(f"[download] {handle}")
    try:
        kagglehub.dataset_download(handle)
        src_root = KAGGLEHUB_CACHE / owner / name
        if src_root.exists():
            _copy_to_project(handle, src_root)
    except Exception as exc:
        print(f"  ERROR: {exc}", file=sys.stderr)


if __name__ == "__main__":
    if not sys.argv[1:] or sys.argv[1] == "--list":
        print("Usage: python scripts/download_new_datasets.py owner/dataset [owner/dataset ...]")
        print("\nKnown datasets (from build_specialist_datasets.py):")
        known = [
            "mihprofi/drone-detect",
            "shakedlevnat/military-aircraft-database",
            "nzigulic/military-equipment",
            "piterfm/2022-ukraine-russia-war-equipment-losses-oryx",
            "sudipchakrabarty/kiit-mita",
            "rookieengg/military-aircraft-detection-dataset-yolo-format",
            "rawsi18/military-assets-dataset-12-classes-yolo8-format",
            "rupankarmajumdar/amad-5-aerial-military-asset-detection-dataset",
        ]
        for d in known:
            print(f"  {d}")
        sys.exit(0)

    for handle in sys.argv[1:]:
        download(handle)
