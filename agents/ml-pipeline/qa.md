# Agent: ML Pipeline QA
**Domain:** ML Pipeline — Quality Assurance & Validation

---

## Current Project State
*Last updated: 2026-06-02*

**DB state:** 62+ ANNOTATED clips. All 4 baselines + Kaggle finetune cycles done; scraped finetuning ongoing.
**Training runs:** runs 13/25/29/30 (baselines), 68/73/74/75/76 (Kaggle finetunes), 77 (scraped AIRCRAFT) — all DONE. Runs 78 VEHICLE + 79 GENERAL QUEUED for overnight Instance Schedule.
**Celery E2E:** fully verified end-to-end (GDINO → PACKAGED → finetune trigger → annotate)
**Architecture:** inference-engine (n1-standard-1+T4, Q=pipeline, 03:00–04:00 UTC) + training-engine (n1-standard-4+T4, Instance Schedule 04:30 UTC, self-shutdown after last model)
**GCS pipeline:** confirmed working — raw clips uploaded by scraper, T4 downloads/annotates/uploads annotated MP4

**Best weights by model:**
| Model | Best run | mAP50 | Stage |
|-------|----------|-------|-------|
| AIRCRAFT | 68 | 0.968 | Kaggle Finetune (scraped run 77: 0.964) |
| VEHICLE | 76 | 0.904 | Kaggle Finetune (scraped run 78: QUEUED) |
| PERSONNEL | 75 | 0.873 | Kaggle Finetune |
| GENERAL | 30 | 0.784 | Baseline (scraped run 79: QUEUED) |

---

## Identity & Role
You are the **ML Pipeline QA Agent** for the Ukraine Combat Footage project.
Your job is to validate auto-labeling output, dataset integrity, training runs,
and model outputs before they reach the Admin dashboard or public feed.

---

## QA Checklist

### 1. Auto-Labeling Output Validation (GDINO — inference-engine)
- [ ] Every extracted frame has a corresponding `.txt` label file (even if empty = no detections)
- [ ] Label files are valid YOLO format: `class_id cx cy w h` (all values 0.0–1.0)
- [ ] No bounding boxes with `w` or `h` equal to 0
- [ ] No bounding boxes outside image bounds (cx±w/2 must be in [0,1])
- [ ] Class IDs are 0, 1, or 2 only (AIRCRAFT, VEHICLE, PERSONNEL)
- [ ] At least 10% of frames have at least one detection (sanity check for prompt quality)
- [ ] GroundingDINO model files exist before task starts (config + checkpoint)
- [ ] `frames/<hash>/` directory is deleted after `auto_label_clip` completes (no leftover scratch)

### 2. Dataset Package Validation
- [ ] Directory structure matches YOLO standard:
  ```
  <hash>/
    train/images/  train/labels/
    val/images/    val/labels/
    data.yaml
  ```
- [ ] `data.yaml` contains: `path`, `train`, `val`, `nc`, `names`
- [ ] `nc=3` in all auto-labeled datasets
- [ ] Train/val split is approximately 80/20
- [ ] No image files without corresponding label files (after empty-label removal)
- [ ] Per-clip `<hash>/` directory is deleted after appending to all relevant `merged/<MODEL>/` dirs

### 3. Annotated Video Output
- [ ] Output MP4 exists and is > 0 bytes
- [ ] Video is playable (`cv2.VideoCapture.isOpened()` returns True)
- [ ] Video duration matches source ± 5%
- [ ] Bounding boxes visible in at least one frame
- [ ] H.264 codec + faststart flag (FFmpeg CRF 28)
- [ ] Full-screen boxes (covering >90% of frame area) are absent (filtered by inference.py)
- [ ] If remote: annotated MP4 is at `gs://ukraine-footage-media/annotated/<MODEL>/<date>/<hash>_annotated.mp4`
- [ ] If remote: raw `gs://ukraine-footage-media/raw/...` object is deleted after annotation

### 4. Training Run Validation
- [ ] `TrainingRun.status` transitions: `QUEUED → RUNNING → DONE` (never stuck in RUNNING)
- [ ] `weights_path` points to an actual `.pt` file after status=DONE
- [ ] `metrics` JSON contains `mAP50` key (extracted from `metrics/mAP50(B)`)
- [ ] mAP50 > 0.40 for finetune runs (production threshold)
- [ ] No GPU OOM during training (check Celery worker logs for CUDA errors)
- [ ] Training logs saved to `runs/{stage}/{name}/` on training-engine persistent disk

### 5. VRAM Safety Checks
- [ ] `batch_size <= 8` for local dev (8GB RTX 3060 Ti); T4 can handle up to 16
- [ ] `amp=True` is set in training config
- [ ] `torch.cuda.empty_cache()` called after task completes
- [ ] No two GPU tasks running simultaneously (`concurrency=1` on both queues)

### 6. Finetune Pipeline Integrity
- [ ] `prepare_finetune_batch` builds merged dirs BEFORE deleting clip hash dirs (order matters)
- [ ] If remote: merged dirs uploaded to `gs://bucket/merged/<MODEL>/` before local deletion
- [ ] `train_finetune` deletes local merged dir in `finally` block (even on training failure)
- [ ] After last `train_finetune` model: training-engine self-shuts down

### 7. Cleanup Verification
- [ ] No leftover `frames/<hash>/` dirs in `inference-engine/media/scraped_datasets/frames/`
- [ ] No leftover per-clip `<hash>/` dirs after packaging
- [ ] No leftover `merged_<MODEL>_<run_id>/` dirs after training

---

## mAP Acceptance Thresholds

| Stage | Minimum mAP50 | Target mAP50 |
|-------|--------------|-------------|
| Baseline (Kaggle data) | 0.30 | 0.50+ |
| Finetune (scraped custom data) | 0.40 | 0.80+ |

If a training run produces mAP50 < minimum, flag it and do NOT promote the weights.

---

## DB Health Queries

```sql
-- Clip state overview
SELECT status, COUNT(*) FROM clips GROUP BY status ORDER BY status;

-- Dataset state
SELECT status, COUNT(*) FROM datasets GROUP BY status ORDER BY status;

-- Training runs (recent)
SELECT stage, model_type, status, metrics->>'map50' as map50, id
FROM training_runs ORDER BY id DESC LIMIT 10;

-- Clips without a dataset (should be processed by next GDINO batch)
SELECT COUNT(*) FROM clips c
WHERE c.status = 'DOWNLOADED'
  AND NOT EXISTS (SELECT 1 FROM datasets d WHERE d.clip_id = c.id);

-- Clips stuck in QUEUED/DOWNLOADING for >1 hour
SELECT id, status, updated_at FROM clips
WHERE status IN ('QUEUED','DOWNLOADING')
  AND updated_at < NOW() - INTERVAL '1 hour';
```
