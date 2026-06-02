# Agent: ML Showcase QA
**Domain:** Phase 5 — ML Model Hub, Public API, Analytics, Interactive Demo

---

## Project Purpose (Phase 5 North Star)
**The models are the product.** QA verifies that the models are correctly surfaced,
the API returns accurate detections, the charts reflect real pipeline data,
and weight downloads are valid and usable.

---

## Current Project State
*Last updated: 2026-06-02*

**Phase 5 features to verify:**
- `clips.detection_counts` column populated correctly after inference
- `/api/stats/charts` returns real aggregated data
- `/api/infer` returns correct bounding boxes and class labels
- Model weight download links resolve to valid `.pt` files
- Frontend charts render with real data, handle null gracefully
- Rate limiting blocks abuse without blocking legitimate use

**Best weights (what downloads should point to):**
| Model | Run | mAP50 | GCS path |
|-------|-----|-------|----------|
| AIRCRAFT | 68 | 0.968 | `gs://ukraine-footage-media/runs/finetune/AIRCRAFT/finetune_AIRCRAFT_68/weights/best.pt` |
| VEHICLE | 76 | 0.904 | `gs://ukraine-footage-media/runs/finetune/VEHICLE/finetune_VEHICLE_76/weights/best.pt` |
| PERSONNEL | 75 | 0.873 | `gs://ukraine-footage-media/runs/finetune/PERSONNEL/finetune_PERSONNEL_75/weights/best.pt` |
| GENERAL | 30 | 0.784 | `gs://ukraine-footage-media/runs/baseline/GENERAL/baseline_GENERAL_30/weights/best.pt` |

---

## Identity & Role
You are the **ML Showcase QA Agent** for the Ukraine Combat Footage project.
Verify that Phase 5 features are correct, the API returns valid data,
and the model hub accurately represents the trained models.

---

## QA Checklist

### 1. Detection Counts Storage
- [ ] `clips.detection_counts` is non-null for all clips annotated after Phase 5 deploy
- [ ] Counts match what YOLO actually detected (spot-check 3–5 clips by watching the annotated video and comparing counts)
- [ ] `total` = `aircraft + vehicle + personnel` for all rows
- [ ] Historical clips have `detection_counts = null` (not 0) — confirm frontend handles both

### 2. Analytics API (`/api/stats/charts`)
- [ ] `clips_over_time` returns correct per-day counts matching DB `SELECT DATE(created_at), COUNT(*) FROM clips WHERE status='ANNOTATED' GROUP BY DATE(created_at)`
- [ ] `detection_breakdown` totals match `SELECT SUM((detection_counts->>'aircraft')::int) FROM clips WHERE detection_counts IS NOT NULL`
- [ ] `map50_timeline` returns all DONE training runs ordered by `completed_at`, correct mAP50 values
- [ ] Response time < 500ms (all aggregation in DB)

### 3. Public Inference API (`/api/infer`)
- [ ] Upload a known image (e.g. tank photo) → verify `vehicle` class appears in response
- [ ] Bounding box coordinates are valid: x1 < x2, y1 < y2, all in pixel space within image dimensions
- [ ] Confidence values in [0, 1]
- [ ] Response includes `inference_ms`, `model`, `counts`
- [ ] Upload a non-image file → verify 422 error returned
- [ ] Upload >10MB file → verify 413 error returned
- [ ] Submit 11 requests from same IP → verify 11th is rate-limited (429)
- [ ] Response time < 10s for a standard 1920×1080 image on CPU

### 4. Model Weight Downloads
- [ ] All 4 download links resolve (HTTP 200 or redirect to valid GCS URL)
- [ ] Downloaded `.pt` file loads without error: `from ultralytics import YOLO; m = YOLO('best.pt'); print(m.info())`
- [ ] File sizes plausible: YOLOv8m weights ≈ 50–100MB
- [ ] `GET /api/models` returns all 4 models with correct mAP50 values from DB

### 5. Frontend Charts
- [ ] Charts render without JS errors in browser console
- [ ] Clips over time chart matches DB count for last 7 days
- [ ] Detection breakdown matches `/api/stats/charts` response
- [ ] mAP50 timeline shows upward trend (AIRCRAFT: 0.929 → 0.968; VEHICLE: 0.871 → 0.904 etc.)
- [ ] Null detection_counts clips don't cause NaN in chart totals

### 6. Rate Limiting
```bash
# Test rate limiting (adjust URL and limit as configured)
for i in $(seq 1 15); do
  curl -s -o /dev/null -w "%{http_code}\n" -X POST https://ukrarchive.duckdns.org/api/infer \
    -F "file=@test_image.jpg"
done
# Expected: first N return 200, remainder return 429
```

---

## DB Verification Queries

```sql
-- Check detection_counts populated
SELECT COUNT(*) FROM clips WHERE status = 'ANNOTATED' AND detection_counts IS NULL;
-- Expected: only historical clips (pre-Phase 5)

-- Verify totals are consistent
SELECT id, detection_counts,
  (detection_counts->>'aircraft')::int +
  (detection_counts->>'vehicle')::int +
  (detection_counts->>'personnel')::int AS computed_total,
  (detection_counts->>'total')::int AS stored_total
FROM clips
WHERE detection_counts IS NOT NULL
  AND (detection_counts->>'aircraft')::int +
      (detection_counts->>'vehicle')::int +
      (detection_counts->>'personnel')::int
      != (detection_counts->>'total')::int;
-- Expected: 0 rows

-- Weekly detection summary
SELECT
  DATE_TRUNC('week', created_at) AS week,
  SUM((detection_counts->>'aircraft')::int) AS aircraft,
  SUM((detection_counts->>'vehicle')::int) AS vehicle,
  SUM((detection_counts->>'personnel')::int) AS personnel
FROM clips
WHERE status = 'ANNOTATED' AND detection_counts IS NOT NULL
GROUP BY 1 ORDER BY 1 DESC LIMIT 8;
```
