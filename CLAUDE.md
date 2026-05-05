# CLAUDE.md — Ukraine Combat Footage Web Application
> Persistent system prompt. Read at the start of every session.

---

## Project Identity

**Name:** Ukraine Combat Footage Archival System
**Repo:** `ukraine-war-footage-training` (monorepo, GitHub: NINJAgur/ukraine-war-footage-training)
**Purpose:** Scrape combat footage → auto-label with GroundingDINO → public display + Admin YOLO retraining panel.

---

## Architecture

**Continuous production loop:**
```
[Celery Beat] → [scraper-engine] → Clip(DOWNLOADED, scores populated) → [ML pipelines] → annotated MP4 → Public Feed
                                                               ↑
                                              best .pt per model (FINETUNE > BASELINE > pretrained)
                                                               ↑
                        [train_finetune] ←── [auto_label + package_dataset] ←── accumulated clips
                               ↑
                        [train_baseline] ← Kaggle cold-start (once, in dev)
```

**Scraper → ML pipeline decoupling (Phase 1.9 upgrade):**
- Scrapers are "greedy vacuums": they scrape broadly and save keyword match scores to DB columns (`score_aircraft`, `score_vehicle`, `score_personnel`, `score_uas`, `is_pov`)
- ML pipelines query DB by score thresholds (majority voting) — no HTTP requests, no re-scraping
- Raw `.mp4` files deleted from disk after annotation to prevent disk bloat (`STORAGE_MODE=local|remote`)
- `get_equipment_scores(title, desc)` in `_filter.py` is the single source of scoring truth

**3 universal classes (aligned with `_filter.py`):**
- `0=AIRCRAFT` — drones, helicopters, fixed-wing, missiles
- `1=VEHICLE` — tanks, APCs, artillery, radar, all ground military vehicles
- `2=PERSONNEL` — soldiers, fighters, RPG/ATGM operators

**Cold-start training order (8 Kaggle datasets total):**
1. Dataset prep: nzigulic mapped nc=11→nc=3 ✅; piterfm GDINO labeled ✅; rookieengg, rawsi18, amad-5 added ✅
2. AIRCRAFT baseline: mAP50=0.929 @ 10 epochs, run 13 ✅
3. VEHICLE baseline: mAP50=0.871 @ 10 epochs, run 25 ✅
4. PERSONNEL baseline: ⏳ next (~25K images, kiit-mita + rawsi18 + amad-5)
5. GENERAL: ⏳ after all 3 specialists pass mAP50 > 0.4 (~175K images, all 8 datasets)

**GDINO auto-label pipeline (category-aware "." prompt → canonical nc=3):**
- Video clips: `core/autolabeling/auto_label.py` — extracts frames, runs GDINO, remaps via `GDINO_CLASS_TO_MODEL`
- Any image folder: `tasks/autolabel_kaggle.py --path <dir> [--prompt <terms>]` — universal, recursive
- piterfm specifically: `tasks/relabel_piterfm.py` — category-aware per-image prompts (Aircraft→"aircraft", Tanks→"tank" etc.)
- All on-disk labeled datasets are nc=3 with canonical IDs baked in

**Shared DB models:** `shared/db/models.py` — single source of truth for all ORM models.
All services import via re-export stubs (`ml-engine/db/models.py`, `scraper-engine/db/models.py`, `web-app/backend/db/models.py`).

**Fine-tune loop (after enough scraped clips):**
- Extract frames from scraped clips → GDINO auto-label (3 classes) → dataset → fine-tune from baseline
- `media/frames/` is scratch space for GDINO only — always deleted after auto-labeling

**4 YOLO models:** AIRCRAFT + VEHICLE + PERSONNEL (specialists) + GENERAL — specialists train first.

| Service | Directory | Phase |
|---------|-----------|-------|
| Scraper Engine | `scraper-engine/` | 1 ✅ |
| ML Engine | `ml-engine/` | 2 🔄 (AIRCRAFT ✅ VEHICLE ✅ PERSONNEL ⏳ GENERAL ⏳) |
| Backend API | `web-app/backend/` | 3 🔄 |
| Frontend | `web-app/frontend/` | 3 🔄 |

---

## Hard Constraints

- **OS:** Windows 11 native Python — no Docker until Phase 4
- **GPU:** RTX 3060 Ti, 8GB VRAM — `batch_size≤8`, always `device='cuda:0'`
- **FastAPI:** `async def` routes, `AsyncSession`, Pydantic v2 `ConfigDict`
- **Vue 3:** `<script setup>` only, Pinia, Tailwind dark theme (slate/zinc + `#22c55e` / `#ef4444`)
- **Celery:** `gpu` queue, `concurrency=1`, all tasks idempotent, Redis broker
- **DB:** PostgreSQL 16, `ukraine_footage`, SQLAlchemy 2.x
- **Scraping:** Funker530 REST + GeoConfirmed REST + yt-dlp, SHA256 url_hash dedup
- **See** `rules/` for coding standards: `celery-rules.md`, `fastapi-rules.md`, `vue3-rules.md`, `yolo-rules.md`
- **See** `agents/` for detailed domain context and DB schema

---

## Key Files

| What | Where |
|------|-------|
| Shared ORM models | `shared/db/models.py` |
| Training entry point | `ml-engine/core/main.py` |
| Inference + multi-model | `ml-engine/core/inference.py` |
| Auto-label (video clips) | `ml-engine/core/autolabeling/auto_label.py` |
| Auto-label (any image folder) | `ml-engine/tasks/autolabel_kaggle.py` |
| Baseline training task | `ml-engine/tasks/train_baseline.py` |
| ML tasks | `ml-engine/tasks/` |
| AIRCRAFT pipeline (DB-driven) | `ml-engine/scripts/aircraft_pipeline.py` |
| VEHICLE pipeline (DB-driven) | `ml-engine/scripts/vehicle_pipeline.py` |
| Funker530 scraper | `scraper-engine/tasks/scrape_funker530.py` |
| GeoConfirmed scraper | `scraper-engine/tasks/scrape_geoconfirmed.py` |
| Content filter + scoring | `scraper-engine/tasks/_filter.py` |
| Phase 1 test | `scraper-engine/tests/test_scrape_live.py` |
| Phase 2 baseline test | `ml-engine/tests/test_baseline_train.py` |
| Phase 2 E2E test | `ml-engine/tests/test_pipeline_e2e.py` |
| Project plan | `PROJECT_PLAN.md` |

Run Phase 1 test: `cd scraper-engine && python tests/test_scrape_live.py`
Run Phase 2 test: `cd ml-engine && python tests/test_pipeline_e2e.py`

---

## Phase Status

| Phase | Focus | Status |
|-------|-------|--------|
| 0 | Agentic workspace | ✅ Complete |
| 1 | Scraper engine | ✅ Complete |
| 2 | ML pipeline — baseline training | 🔄 In progress (AIRCRAFT ✅ 0.929, VEHICLE ✅ 0.871, PERSONNEL ⏳, GENERAL ⏳) |
| 3 | Web application | 🔄 In progress (core wired; WebSocket + full Celery pipeline pending) |
| 4 | Cloud & DevOps | ⏳ Pending |

---

## Do NOT

- Use Docker locally (Phase 4 only)
- Use Options API or Vuex in Vue components
- Use sync SQLAlchemy in FastAPI routes
- Use `print()` in production code — use `logging`
- Hardcode credentials or absolute file paths
- Start GPU workers with `concurrency > 1`
- Write code for phases not yet reached (stay sequential)
- Commit without explicit user approval
