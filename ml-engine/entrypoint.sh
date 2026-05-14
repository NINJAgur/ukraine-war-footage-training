#!/usr/bin/env bash
# ml-engine container entrypoint.
# Downloads model weights into the ml_checkpoints volume on first startup.
# Subsequent restarts skip downloads if files already exist.
set -e

CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-/checkpoints}"
mkdir -p "$CHECKPOINTS_DIR"

GDINO_PTH="$CHECKPOINTS_DIR/groundingdino_swint_ogc.pth"
if [ ! -f "$GDINO_PTH" ]; then
    echo "[entrypoint] Downloading GroundingDINO checkpoint (~694 MB)..."
    wget -q --show-progress -O "$GDINO_PTH" \
        https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
    echo "[entrypoint] GroundingDINO checkpoint ready."
fi

YOLO_PTH="$CHECKPOINTS_DIR/yolov8m.pt"
if [ ! -f "$YOLO_PTH" ]; then
    echo "[entrypoint] Downloading YOLOv8m base weights (~52 MB)..."
    wget -q --show-progress -O "$YOLO_PTH" \
        https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt
    echo "[entrypoint] YOLOv8m weights ready."
fi

exec "$@"
