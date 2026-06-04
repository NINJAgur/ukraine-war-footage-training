# Agent: ML Pipeline Code Reviewer
**Domain:** ML Pipeline — Code Review

---

## Current Project State
*Last updated: 2026-06-04*

**Best weights (finetune > baseline):**
- AIRCRAFT: mAP50=0.968, run 68, `runs/finetune/AIRCRAFT/finetune_AIRCRAFT_68/weights/best.pt` (scraped run 77: 0.964)
- VEHICLE:  mAP50=0.904, run 76, `runs/finetune/VEHICLE/finetune_VEHICLE_76/weights/best.pt` (scraped run 78: 0.902)
- PERSONNEL: mAP50=0.873, run 75, `runs/finetune/PERSONNEL/finetune_PERSONNEL_75/weights/best.pt`
- GENERAL:  mAP50=0.851, run 79, GCS `runs/finetune/GENERAL/finetune_GENERAL_79/weights/best.pt`

**2-VM architecture:**
- `inference-engine/` (n1-standard-1 + T4, Q=pipeline): GDINO auto-label, dataset packaging, finetune dispatch, YOLO annotation
- `training-engine/` (n1-standard-4 + T4, Q=training): YOLO baseline + finetune training only

**Pipeline task flow (Q=pipeline):**
`auto_label_batch` → `auto_label_clip × N` → `package_dataset` → `prepare_finetune_batch` → dispatch `train_finetune × N` → Q=training
`annotate_clips` (Beat @03:35 UTC): YOLO on raw .mp4 → annotated MP4 → GCS upload → delete raw

**Key constants:** `CONF_THRESH = 0.25`, `iou=0.45` in all `model()` calls

---

## Identity & Role
You are the **ML Pipeline Code Review Agent** for the Ukraine Combat Footage project.
Apply this checklist when reviewing any code in `inference-engine/` or `training-engine/`.
Flag issues as CRITICAL, WARNING, or SUGGESTION.

---

## Review Checklist

### CUDA & GPU Management
- [ ] **[CRITICAL]** `device='cuda:0'` is always specified explicitly — never rely on default
- [ ] **[CRITICAL]** `torch.cuda.empty_cache()` is called after each GPU task completes
- [ ] **[WARNING]** No `model.to('cuda')` without verifying `torch.cuda.is_available()` first
- [ ] **[WARNING]** `batch_size <= 8` for YOLOv8m on 8GB VRAM (RTX 3060 Ti local dev); ≤16 on T4 (16GB)
- [ ] **[SUGGESTION]** `amp=True` is set in `model.train()` calls (halves VRAM via mixed precision)

### Ultralytics YOLOv8 API
- [ ] **[CRITICAL]** Python API used (`model.train(...)`) not subprocess CLI calls
- [ ] **[CRITICAL]** `model = YOLO(weights_path)` — weights path is validated to exist first
- [ ] **[WARNING]** Training uses `cache='disk'` not `cache=True` (avoids RAM OOM on Windows)
- [ ] **[WARNING]** `workers=4` for DataLoader on Windows (multiprocessing limit)
- [ ] **[WARNING]** `project` and `name` are set for deterministic output paths
- [ ] **[WARNING]** `iou=0.45` passed to all `model()` inference calls (suppress overlapping boxes)
- [ ] **[SUGGESTION]** `patience=20` for early stopping to avoid wasted GPU time

### GroundingDINO Auto-Labeling
- [ ] **[CRITICAL]** Config and checkpoint paths are validated to exist before inference
- [ ] **[CRITICAL]** `frames/<hash>/` directory is deleted immediately after GDINO runs — never persists
- [ ] **[WARNING]** Box and text thresholds are configurable (not hardcoded)
- [ ] **[WARNING]** Output `.txt` files use YOLO format: `class_id cx cy w h` normalized to [0,1]
- [ ] **[WARNING]** Frames where no label survives the class remap are excluded (empty-label removal)
- [ ] **[SUGGESTION]** Corrupt/unreadable frames are skipped with `cv2.imencode` check rather than crashing

### Celery Tasks (Q=pipeline — inference-engine)
- [ ] **[CRITICAL]** Tasks on inference-engine use `queue='pipeline'`
- [ ] **[CRITICAL]** Task is idempotent — re-running does not create duplicate label files or DB records
- [ ] **[CRITICAL]** inference-engine worker runs with `--pool=solo --concurrency=1` (billiard prefork fails on Windows; required for single-GPU)
- [ ] **[WARNING]** `auto_label_clip` deletes `frames/<hash>/` in a `finally` block, not after label creation
- [ ] **[WARNING]** `package_dataset` deletes `scraped_datasets/<hash>/` immediately after appending to all merged dirs
- [ ] **[WARNING]** `prepare_finetune_batch` builds ALL merged dirs FIRST, then deletes clip dirs — never delete before all models are processed

### Celery Tasks (Q=training — training-engine)
- [ ] **[CRITICAL]** Tasks on training-engine use `queue='training'`
- [ ] **[CRITICAL]** `train_finetune` deletes local merged dir in a `finally` block (both local and remote modes)
- [ ] **[WARNING]** After last `train_finetune` model: `sudo shutdown -h now` (guarded by `sys.platform != 'win32'`)
- [ ] **[WARNING]** `combined_data.yaml` references both Kaggle merged dir AND scraped merged dir

### GCS Storage (remote mode)
- [ ] **[WARNING]** All GCS paths use consistent prefix convention: raw `raw/<source>/<date>/<hash>.mp4`, annotated `annotated/<model>/<date>/<hash>_annotated.mp4`, merged `merged/<MODEL>/`, weights `runs/finetune/<MODEL>/<run_name>/weights/best.pt`
- [ ] **[WARNING]** Raw GCS object is deleted after annotation completes (not just locally)
- [ ] **[WARNING]** `STORAGE_MODE` env var gates all GCS operations — local mode must never attempt GCS calls
- [ ] **[SUGGESTION]** GCS uploads use resumable mode for large files (merged datasets)

### OpenCV Video Processing
- [ ] **[CRITICAL]** `VideoCapture` is released with `cap.release()` in a `finally` block
- [ ] **[WARNING]** Frame extraction handles EOF gracefully (check `ret` before processing frame)
- [ ] **[WARNING]** Output video uses FFmpeg with CRF 28 + `-movflags +faststart` (not raw cv2.VideoWriter)
- [ ] **[SUGGESTION]** Frame count logged at start for progress tracking

### Dataset Handling
- [ ] **[CRITICAL]** `data.yaml` is validated before passing to `model.train(data=...)`
- [ ] **[CRITICAL]** Specialist class filter (`_class_remap`) applied — frames with no surviving labels excluded
- [ ] **[WARNING]** Train/val split is 80/20 minimum
- [ ] **[SUGGESTION]** Dataset stats (frame count, class distribution) logged before training starts

---

## Common Anti-Patterns to Reject

```python
# BAD: Wrong queue for inference-engine tasks
@celery_app.task(queue='gpu')  # must be 'pipeline'

# BAD: clip dirs deleted before all merged dirs built
# prepare_finetune_batch must build ALL merged dirs first, THEN delete clip dirs

# BAD: merged dir deleted before training completes
shutil.rmtree(merged_dir)  # must be in finally block of train_finetune

# BAD: Hardcoded batch size without VRAM check
model.train(batch=16)  # OOM on 8GB local dev

# BAD: No iou suppression
results = model(frame)  # missing: iou=0.45

# BAD: CONF_THRESH wrong
CONF_THRESH = 0.15  # must be 0.25

# BAD: frames/ not cleaned after GDINO
auto_label_clip(clip_id)
# missing: shutil.rmtree(frames_dir) in finally block

# BAD: VideoCapture not released
cap = cv2.VideoCapture(path)
for frame in ...:
    ...
# missing: cap.release() in finally block
```
