#!/usr/bin/env bash
# One-time dataset setup for a fresh deployment.
# Downloads 8 Kaggle datasets (~10 GB) and builds the merged specialist folders.
# Skips automatically if merged/ already exists in the volume.
#
# Local:  cd training-engine && bash scripts/setup_datasets.sh
# Docker: docker compose run --rm training-worker bash scripts/setup_datasets.sh

set -e
cd "$(dirname "$0")/.."

# Use venv python if available (GCP deployment), otherwise fall back to system python3
PYTHON="python3"
[ -f "venv/bin/python3" ] && PYTHON="venv/bin/python3"


MERGED_DIR="media/kaggle_datasets/merged"
if [ -d "$MERGED_DIR/GENERAL/train/images" ] && [ -n "$(ls -A "$MERGED_DIR/GENERAL/train/images" 2>/dev/null)" ]; then
    echo "[setup_datasets] Merged datasets already present — nothing to do."
    echo "[setup_datasets] Delete $MERGED_DIR to force a full re-download and rebuild."
    exit 0
fi

echo "[setup_datasets] Downloading 8 Kaggle datasets (~10 GB, may take 30-60 min)..."
$PYTHON - <<'PY'
import sys
sys.path.insert(0, ".")
import kagglehub

datasets = [
    "mihprofi/drone-detect",
    "shakedlevnat/military-aircraft-database-prepared-for-yolo",
    "nzigulic/military-equipment",
    "piterfm/2022-ukraine-russia-war-equipment-losses-oryx",
    "sudipchakrabarty/kiit-mita",
    "rookieengg/military-aircraft-detection-dataset-yolo-format",
    "rawsi18/military-assets-dataset-12-classes-yolo8-format",
    "rupankarmajumdar/amad-5-aerial-military-asset-detection-dataset",
]
import pathlib

cache_root = pathlib.Path.home() / ".cache" / "kagglehub"

failed = []
for handle in datasets:
    print(f"  {handle}...", flush=True)
    try:
        kagglehub.dataset_download(handle)
        # Delete zip files after extraction to save disk space
        for z in cache_root.rglob("*.zip"):
            z.unlink()
            print(f"  Deleted zip: {z.name}", flush=True)
    except Exception as e:
        print(f"  FAILED {handle}: {e}", flush=True)
        failed.append(handle)
if failed:
    print(f"Failed datasets (will retry next boot): {failed}")
    sys.exit(1)
print("All datasets downloaded.")
PY

echo "[setup_datasets] Building merged specialist datasets..."
$PYTHON scripts/build_specialist_datasets.py

echo "[setup_datasets] Cleaning up source dataset cache to free disk space..."
rm -rf "$HOME/.cache/kagglehub/datasets"
echo "[setup_datasets] Done. Trigger baseline training via the Admin panel or:"
echo "  POST /api/admin/train  {\"model_type\": \"AIRCRAFT\"}"
echo "  POST /api/admin/train  {\"model_type\": \"VEHICLE\"}"
echo "  POST /api/admin/train  {\"model_type\": \"PERSONNEL\"}"
echo "  POST /api/admin/train  {\"model_type\": \"GENERAL\"}"
