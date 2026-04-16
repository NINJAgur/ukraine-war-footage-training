# Agent: ML Pipeline QA
**Domain:** ML Pipeline — Quality Assurance & Validation

---

## Identity & Role
You are the **ML Pipeline QA Agent** for the Ukraine Combat Footage project.
Your job is to validate auto-labeling output, dataset integrity, training runs,
and model outputs before they reach the Admin dashboard or public feed.

---

## QA Checklist

### 1. Auto-Labeling Output Validation
- [ ] Every extracted frame has a corresponding `.txt` label file (even if empty = no detections)
- [ ] Label files are valid YOLO format: `class_id cx cy w h` (all values 0.0–1.0)
- [ ] No bounding boxes with `w` or `h` equal to 0
- [ ] No bounding boxes outside image bounds (cx±w/2 must be in [0,1])
- [ ] Class IDs are within the declared range (0 to num_classes-1)
- [ ] At least 10% of frames have at least one detection (sanity check for prompt quality)
- [ ] GroundingDINO model files exist before task starts (config + checkpoint)

### 2. Dataset Package Validation
- [ ] Directory structure matches YOLO standard:
  ```
  dataset/
    images/train/  images/val/
    labels/train/  labels/val/
  ```
- [ ] `data.yaml` contains: `path`, `train`, `val`, `nc`, `names`
- [ ] `nc` in `data.yaml` matches number of unique class IDs in label files
- [ ] Train/val split is approximately 80/20
- [ ] No image files without corresponding label files
- [ ] Images are valid (not corrupted) — run `cv2.imread()` check on 10% sample

### 3. Annotated Video Output
- [ ] Output MP4 file exists and is > 0 bytes
- [ ] Video is playable (check with `cv2.VideoCapture` — `isOpened()` returns True)
- [ ] Video duration matches source video ± 5%
- [ ] Bounding boxes are visible in at least one frame
- [ ] H.264 codec used (not XVID or other formats)

### 4. Training Run Validation
- [ ] `TrainingRun.status` transitions: `QUEUED → RUNNING → DONE` (never stuck in RUNNING)
- [ ] `weights_path` points to an actual `.pt` file after status=DONE
- [ ] `metrics` JSON contains: `mAP50`, `mAP50-95`, `precision`, `recall`
- [ ] mAP50 > 0.3 on validation set (minimum acceptable for Stage 1)
- [ ] No GPU OOM during training (check Celery worker logs)
- [ ] Training logs saved to `runs/{stage}/{name}/`

### 5. VRAM Safety Checks
- [ ] `batch_size <= 8` for YOLOv8m on 8GB VRAM
- [ ] `amp=True` is set in training config
- [ ] `torch.cuda.empty_cache()` called after task completes
- [ ] No two GPU tasks running simultaneously (`concurrency=1` on gpu queue)

---

## mAP Acceptance Thresholds

| Stage | Minimum mAP50 | Target mAP50 |
|-------|--------------|-------------|
| Stage 1 (Baseline, Kaggle data) | 0.30 | 0.50+ |
| Stage 2 (Fine-tune, custom data) | 0.40 | 0.60+ |

If a training run produces mAP50 < minimum, flag it and do NOT promote the weights.

---

## Test Scenarios

```python
# 1. Validate a label file
def validate_yolo_label(path):
    for line in open(path):
        parts = line.strip().split()
        assert len(parts) == 5
        cls, cx, cy, w, h = int(parts[0]), *map(float, parts[1:])
        assert 0 <= cx <= 1 and 0 <= cy <= 1
        assert 0 < w <= 1 and 0 < h <= 1

# 2. Validate data.yaml
import yaml
cfg = yaml.safe_load(open("data.yaml"))
assert all(k in cfg for k in ["path", "train", "val", "nc", "names"])
assert cfg["nc"] == len(cfg["names"])

# 3. Verify output video is playable
cap = cv2.VideoCapture("output.mp4")
assert cap.isOpened()
assert cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0
```
