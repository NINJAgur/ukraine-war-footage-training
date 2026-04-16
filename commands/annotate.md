# /annotate — Run Auto-Labeling on a Folder

## Description
Runs GroundingDINO zero-shot auto-labeling on a folder of images or video frames.
Generates YOLO-format `.txt` label files and optionally packages the result as a dataset.

## Usage
```
/annotate [input_path] [--prompt TEXT] [--box-threshold FLOAT] [--package]
```

### Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `input_path` | Required | Path to folder of images (or a video file to extract frames from) |
| `--prompt` | See below | Comma-separated class names for GroundingDINO |
| `--box-threshold` | `0.35` | Confidence threshold for bounding box detection |
| `--text-threshold` | `0.25` | Confidence threshold for text-image matching |
| `--package` | False | If set, package labeled frames as a YOLO dataset after labeling |

### Default Prompt
```
military vehicle, tank, armored vehicle, armored personnel carrier,
soldier, armed personnel, drone, UAV, artillery, mortar, truck, helicopter
```

### Examples
```bash
/annotate ./media/raw/funker530/abc12345/   # auto-label extracted frames
/annotate ./media/raw/youtube/def67890/ --prompt "soldier, vehicle, tank"
/annotate ./media/raw/ --package           # label + package all raw frames as dataset
/annotate video.mp4                        # extract frames then label
```

## What This Command Does

1. Validate GroundingDINO model files exist:
   - Config: `ml-engine/core/autolabeling/GroundingDINO_SwinT_OGC.py`
   - Checkpoint: `ml-engine/core/autolabeling/groundingdino_swint_ogc.pth`
2. If input is a video file: extract frames at 1 frame per 5 frames using OpenCV
3. Run GroundingDINO on each frame with the specified prompt
4. Write `.txt` label files alongside each image in YOLO format
5. If `--package` is set:
   - Create YOLO directory structure (`images/train/`, `labels/train/`, etc.)
   - Generate `data.yaml`
   - Write a `Dataset` record to the database

## Expected Output Structure

```
input_folder/
  frame_0001.jpg    ← original frame
  frame_0001.txt    ← generated label: "0 0.512 0.384 0.120 0.089"
  frame_0002.jpg
  frame_0002.txt
  ...

# If --package is used, additionally creates:
datasets/{clip_hash}/
  images/train/
  images/val/
  labels/train/
  labels/val/
  data.yaml
```

## GroundingDINO Model Files

If not yet downloaded, fetch them:
```bash
# Download config and checkpoint
wget -q https://github.com/IDEA-Research/GroundingDINO/raw/main/groundingdino/config/GroundingDINO_SwinT_OGC.py \
     -O ml-engine/core/autolabeling/GroundingDINO_SwinT_OGC.py

wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth \
     -O ml-engine/core/autolabeling/groundingdino_swint_ogc.pth
```

## Troubleshooting

- **"Config file not found"**: Download GroundingDINO model files (see above)
- **"CUDA out of memory"**: GroundingDINO uses ~4GB VRAM. Close other GPU processes.
- **"0 detections on all frames"**: Lower `--box-threshold` to 0.25 and refine `--prompt`
- **"Invalid YOLO label"**: Check that bounding box values are all in [0, 1] range
