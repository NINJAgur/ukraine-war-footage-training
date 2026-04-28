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
[Celery Beat] → [scraper-engine] → Clip(DOWNLOADED) → [render_annotated] → annotated MP4 → Public Feed
                                                               ↑
                                              best .pt per model (FINETUNE > BASELINE > pretrained)
                                                               ↑
                        [train_finetune] ←── [auto_label + package_dataset] ←── accumulated clips
                               ↑
                        [train_baseline] ← Kaggle cold-start (once, in dev)
```

**3 universal classes (aligned with `_filter.py`):**
- `0=AIRCRAFT` — drones, helicopters, fixed-wing, missiles
- `1=VEHICLE` — tanks, APCs, artillery, radar, all ground military vehicles
- `2=PERSONNEL` — soldiers, fighters, RPG/ATGM operators

**Cold-start training order:**
1. nzigulic: human-labeled nc=11 → visually mapped → remapped to nc=3 (done ✅)
2. piterfm: GDINO category-aware labels → nc=3, 26k images at `kaggle_datasets/piterfm/2022-ukraine-russia-war-equipment-losses-oryx/versions/1/` (done ✅)
3. Train AIRCRAFT, VEHICLE, PERSONNEL specialists on all 5 Kaggle datasets (tasks 2.36–2.38, in progress 🔄)
4. Train GENERAL only after all 3 specialists pass mAP50 > 0.4 (task 2.40)

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
| ML Engine | `ml-engine/` | 2 🔄 |
| Backend API | `web-app/backend/` | 3 ⏳ |
| Frontend | `web-app/frontend/` | 3 ⏳ |

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
| piterfm category-aware labeler | `ml-engine/tasks/relabel_piterfm.py` |
| Baseline training task | `ml-engine/tasks/train_baseline.py` |
| ML tasks | `ml-engine/tasks/` |
| Funker530 scraper | `scraper-engine/tasks/scrape_funker530.py` |
| GeoConfirmed scraper | `scraper-engine/tasks/scrape_geoconfirmed.py` |
| Content filter | `scraper-engine/tasks/_filter.py` |
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
| 2 | ML pipeline — baseline training | 🔄 In progress (2.36 AIRCRAFT training running, next: 2.37 VEHICLE, 2.38 PERSONNEL) |
| 3 | Web application | ⏳ Pending |
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
