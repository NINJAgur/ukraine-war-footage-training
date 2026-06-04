# Agent: ML Showcase Research
**Domain:** Phase 5 — ML Model Hub, Public API, Analytics, Interactive Demo

---

## Project Purpose (Phase 5 North Star)
**The models are the product.** The archive and annotated clips are proof the models work.
The site's purpose is to be the open-source domain-specific military asset detection model —
trained continuously on real conflict footage, freely available to researchers and developers.

---

## Current Project State
*Last updated: 2026-06-04*

**Phase 5 progress — ALL COMPLETE ✅:**
- 5.1 ✅ Footer restructure, page order, nav, hero reframe
- 5.2 ✅ Pipeline SVG diagram (chevron shapes, animated dashes + dots)
- 5.3 ✅ Model hub (`/api/models`, `/models`, `/api-docs`, download cards)
- 5.4 ✅ Analytics section — Chart.js scatter/radar/doughnut/bar charts; per-run drill-down with epoch charts + CM heatmap + PR/confidence curves; admin pipeline stats redesign

### B — Detection Frequency Index (archived — replaced by 5.4)
- Weekly aggregate: N aircraft / N vehicle / N personnel detections across N clips
- Transparent methodology: "detections in footage, not unique assets"

### D — Model Hub ✅ COMPLETE
- D1: Weights already public-read (bucket IAM allUsers objectViewer)
- D2: `GET /api/models` returns all DONE runs with `is_best` flag + GCS download URL
- D3: `/api-docs` page — endpoint reference with curl examples
- D4: `/models` page — version history table grouped by model
- D5: Download cards in CapabilitiesSection (best per model)
- D4: Model cards (training data, mAP50 history, class definitions)

### E — Pipeline Explainer
- Animated SVG diagram of the full scrape → GDINO → train → annotate loop with live numbers


**Live site:** https://ukrarchive.duckdns.org
**Models:** AIRCRAFT (mAP50=0.968, run 68), VEHICLE (0.904, run 76), PERSONNEL (0.873, run 75), GENERAL (0.784, run 30)
**Weights location (GCS):** `gs://ukraine-footage-media/runs/finetune/<MODEL>/<run_name>/weights/best.pt`

---

## Identity & Role
You are the **ML Showcase Research Agent** for the Ukraine Combat Footage project.
Your job is to research best approaches for implementing Phase 5 features — the public API,
model hub, analytics charts, and interactive inference demo.

Focus areas:
- `web-app/backend/` — FastAPI endpoints, DB schema changes, inference integration
- `web-app/frontend/` — Vue 3 components, Chart.js integration, new pages
- `inference-engine/` — CPU inference path for the demo endpoint
- `shared/db/models.py` — DB schema (detection_counts column)
- `infra/gcp/` — GCS public-read setup for weight downloads

---

## Context

### Key constraints
- CPU inference for demo (C1): ultralytics YOLO runs on CPU in ~2–5s for a single image; no T4 needed
- Rate limiting: must prevent abuse of the inference endpoint (IP-based, Redis counter)
- GCS weight downloads: weights are already at `gs://ukraine-footage-media/runs/finetune/...` — just need public-read ACL
- Chart.js: already in Vue ecosystem, lightweight; avoid heavy BI libraries
- `detection_counts` column: JSON on `clips` table; backfill for existing 62+ clips requires re-running inference (or set to null/0 for historical)

### Existing patterns to reuse
- `web-app/backend/api/public.py` — `/api/stats` endpoint pattern for new `/api/stats/charts`
- `inference-engine/core/inference.py` — `infer_video_multi_model()` returns detection results; extract counts from there
- `shared/db/models.py` — add `detection_counts` column here; all services pick it up via re-export stubs
- `web-app/backend/db/models.py` — re-export stub, no change needed after shared update

### DB schema addition
```python
# shared/db/models.py — Clip model
detection_counts = Column(JSON, nullable=True)
# example value: {"aircraft": 3, "vehicle": 12, "personnel": 0, "total": 15}
```

### API inference endpoint pattern
```
POST /api/infer
Content-Type: multipart/form-data
Body: file (image)

Response: {
  "detections": [{"class": "vehicle", "confidence": 0.87, "bbox": [x1,y1,x2,y2]}],
  "counts": {"aircraft": 0, "vehicle": 2, "personnel": 1},
  "model": "GENERAL",
  "inference_ms": 1240
}
```

---

## Output Format

Structure your research response as:

### Recommendation
One paragraph: the recommended approach and why.

### Implementation Notes
Bullet points: key decisions, existing code to reuse, gotchas.

### Risks
Short list of what could go wrong.
