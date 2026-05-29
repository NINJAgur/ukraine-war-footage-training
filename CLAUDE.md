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
- Raw `.mp4` files deleted from disk after annotation AND after rejection (failed validation or zero-detection inference)
- `get_equipment_scores(title, desc)` in `_filter.py` is the single source of scoring truth
- `is_pov_noise(scores)` in `_filter.py` blocks pure FPV clips with zero class scores from entering DB
- Scrapers have two function variants: `_since(since_date)` for Celery/daily runs, `_sample(max_count/max_incidents)` for tests
- Annotated output: `media/annotated/<model>/<publish_date>/<hash>_annotated.mp4` — date is `clip.published_at`, falling back to today; temp files written to same dir, renamed on completion
- Pipeline conf threshold: `CONF_THRESH=0.25` for both `validate_clip` and `infer_video_multi_model`; `iou=0.45` passed to all `model()` calls in `inference.py` to suppress overlapping boxes
- **Container path resolution (local dev):** scraper-worker (Docker) writes `/app/scraper-engine/media/...` paths to DB; pipeline scripts running natively call `_resolve_path()` to map these to Windows paths via `REPO_ROOT / rel`
- **GCS media (production):** scraper uploads raw `.mp4` to `gs://ukraine-footage-media/raw/<source>/<date>/<hash>.mp4` → `clip.file_path = gs://...`; T4 `annotate_clips` downloads raw from GCS via `_download_from_gcs()`, annotates, uploads to `gs://ukraine-footage-media/annotated/...` → `clip.mp4_path = https://storage.googleapis.com/...`; raw GCS object deleted after annotation

**3 universal classes (aligned with `_filter.py`):**
- `0=AIRCRAFT` — drones, helicopters, fixed-wing, missiles
- `1=VEHICLE` — tanks, APCs, artillery, radar, all ground military vehicles
- `2=PERSONNEL` — soldiers, fighters, RPG/ATGM operators

**Cold-start training order (8 Kaggle datasets total):**
1. Dataset prep: all 8 datasets downloaded fresh; class remapping applied in build script (source files never modified); nzigulic + rookieengg reorganized to standard train/val layout ✅
2. Merged folders rebuilt clean via `scripts/build_specialist_datasets.py` ✅ — in-memory class remapping + specialist class filter; verified 0 bad class IDs across all models
3. AIRCRAFT baseline: mAP50=0.929 @ 10 epochs, run 13 ✅ (stale — retraining needed on clean merged/, 65,557 train images)
4. VEHICLE: baseline run 25 (0.871) → finetune run 73 (0.901, 10 epochs on clean merged/, 56,440 train) ✅
5. PERSONNEL: baseline run 29 (0.780) → finetune run 74 (0.872, 20 epochs on clean merged/, 10,962 train) ✅
6. GENERAL: mAP50=0.784 @ 10 epochs, run 30 ✅ (finetune pending)

**Scraped dataset pipeline (GDINO → fine-tune):**

One `Dataset` DB record per clip. Disk layout:
```
inference-engine/media/scraped_datasets/
    frames/<url_hash>/    ← transient scratch; deleted immediately after GDINO runs
    <url_hash>/           ← per-clip YOLO dataset (LABELED → merged into merged/ → deleted by package_dataset)
        train/images/     ← frames with detections (empty-label frames removed post-remap)
        train/labels/     ← canonical nc=3 labels
        data.yaml
    merged/
        AIRCRAFT/         ← persistent accumulation dir; grows per-clip; cleared after training dispatched
        VEHICLE/
        PERSONNEL/
        GENERAL/
```

Pipeline per scrape batch (Beat schedule on inference-engine VM: GDINO @03:05, annotate @03:35 UTC — VM runs 03:00–04:00 UTC):

**Phase 1 — GDINO** (`auto_label_batch` → `auto_label_clip × N`, Q=pipeline):
1. `auto_label_batch` → finds all DOWNLOADED clips without a Dataset → dispatches `auto_label_clip × N`
2. `auto_label_clip` per clip:
   - Extract frames → `frames/<hash>/`
   - Run GDINO → remap to canonical nc=3
   - Delete `frames/<hash>/` immediately
   - Remove frames where label is empty after remap
   - Create `Dataset(LABELED)` with `detected_model_types`
   - Dispatch `package_dataset`

**Phase 2 — Packaging + Merging** (`package_dataset × N`, Q=pipeline, sequential after each auto_label_clip):
3. `package_dataset` per clip:
   - 80/20 train/val split
   - Filter + append clip's YOLO data into `merged/<MODEL>/` for each detected model (specialist class filter applied)
   - Immediately upload updated `merged/<MODEL>/` → `gs://bucket/merged/<MODEL>/` (remote mode) — survives VM recreation
   - Delete `scraped_datasets/<hash>/` immediately after appending to all relevant merged dirs
   - Mark `Dataset(PACKAGED)`

**Phase 3 — Trigger check** (chord callback, fires once after ALL `package_dataset` tasks complete):
4. Count PACKAGED datasets per model type (per-model filter: only datasets where that model was detected; GENERAL counts all)
5. Any model with ≥5 → `TrainingRun(QUEUED)` → dispatch ONE `prepare_finetune_batch` with all qualifying run IDs
6. Mark consumed PACKAGED datasets as TRAINED (so they don't count toward next cycle's threshold)

**Phase 4 — Finetune dispatch** (`prepare_finetune_batch`, Q=pipeline):
7. Remote: upload `merged/<MODEL>/` → `gs://bucket/merged/<MODEL>/` for each qualifying model → delete local merged dir
8. Local: leave merged dirs on disk (train_finetune reads directly)
9. Dispatch `train_finetune × N qualifying models` → Q=training (NO VM start — training-engine has its own Instance Schedule)

**Phase 5 — Training** (`train_finetune`, Q=training, on training-engine — starts at 04:30 UTC):
10. Startup script: query DB for QUEUED TrainingRuns → if none, `sudo shutdown -h now` immediately
11. Remote: download `gs://bucket/merged/<MODEL>/` → local temp dir; Local: read from local merged dir directly
12. Build `combined_data.yaml`: Kaggle merged (persistent disk) + scraped merged dir
13. Train → save weights → remote: upload `best.pt` to GCS
14. `finally`: delete local merged dir (both modes)
15. After last model: `sudo shutdown -h now` (non-Windows only)

**Phase 6 — Annotation** (`annotate_clips`, Q=pipeline, @03:35 UTC — waits behind GDINO in queue):
16. YOLO inference on raw .mp4 → annotated video → delete raw file → `_shutdown_if_no_training`

**Specialist filtering (CRITICAL):** `_class_remap` in `package_dataset.py` returns `{0:0}` for AIRCRAFT, `{1:1}` for VEHICLE, `{2:2}` for PERSONNEL, `{0:0,1:1,2:2}` for GENERAL. Frames where no label survives the filter are excluded. Matches `build_specialist_datasets.py` exactly.

**GCS storage (remote mode):** Raw clips never go to GCS individually — only `merged/<MODEL>/` dirs are uploaded to `gs://bucket/merged/<MODEL>/` when training is dispatched. Trained weights go to `gs://bucket/runs/finetune/<MODEL>/<run_name>/weights/best.pt`.

**Scraper media structure:**
```
scraper-engine/media/
    funker530/          ← downloaded videos (wiped after annotation in production)
    geoconfirmed/       ← downloaded videos (wiped after annotation in production)
    combined/           ← transient packaging folder, created + deleted per run
```

**Kaggle baseline datasets — pre-built merged folders:**
- `scripts/build_specialist_datasets.py` — ONE-TIME script, run once to build `media/kaggle_datasets/merged/`
- All training reads from `media/kaggle_datasets/merged/<MODEL>/dataset.yaml` — never merged on-the-fly again
- `media/kaggle_datasets/combined/` — OLD on-the-fly merge output, deleted
- **amad-5 note:** dataset was cleaned in-place — original class map had 2↔3 swapped (civilians labeled as soldiers). 13,862 images deleted, labels rewritten to canonical nc=3. Class map in `build_specialist_datasets.py` is now a pass-through `{0:0, 1:1, 2:2}`.

**GDINO auto-label:**
- Video clips: `core/autolabeling/auto_label.py` — extracts frames, runs GDINO, remaps via `GDINO_CLASS_TO_MODEL`
- Any image folder: `tasks/autolabel_kaggle.py --path <dir> [--prompt <terms>]` — universal, recursive

**Shared DB models:** `shared/db/models.py` — single source of truth for all ORM models.
All services import via re-export stubs (`inference-engine/db/models.py`, `training-engine/db/models.py`, `scraper-engine/db/models.py`, `web-app/backend/db/models.py`).

**4 YOLO models:** AIRCRAFT + VEHICLE + PERSONNEL (specialists) + GENERAL — specialists train first.

**2-VM production split (GCP) — 1 T4 quota constraint:**
- **inference-engine VM** (n1-standard-1 + T4, Instance Schedule 03:00–04:00 UTC, Q=pipeline): GDINO auto-labeling, dataset packaging, merged dataset creation, YOLO annotation of raw clips. Does NOT start training VM.
- **training-engine VM** (n1-standard-4 + T4, Instance Schedule 04:30 UTC start, self-shutdown): boots at 04:30, checks DB for QUEUED TrainingRuns → if none, `sudo shutdown -h now` immediately; if runs exist, trains all models → self-shuts down when done.
- **1 T4 constraint:** inference-engine stops at 04:00 (Instance Schedule). Training-engine starts at 04:30 (30-min buffer). They never overlap.

| Service | Directory | Phase |
|---------|-----------|-------|
| Scraper Engine | `scraper-engine/` | 1 ✅ |
| Inference Engine | `inference-engine/` | 2 ✅ (GDINO + YOLO annotation + dataset packaging + finetune dispatch) |
| Training Engine | `training-engine/` | 2 ✅ (YOLO baseline + finetune training only) |
| Backend API | `web-app/backend/` | 3 ✅ |
| Frontend | `web-app/frontend/` | 3 ✅ |

---

## Hard Constraints

- **OS:** Windows 11 native Python for ML engine; Docker on GCP e2-micro for production; Docker Desktop for local dev
- **ML engine always runs natively** — no Docker GPU passthrough on Windows without NVIDIA Container Toolkit (Linux-only)
- **GPU:** RTX 3060 Ti, 8GB VRAM — `batch_size≤8`, always `device='cuda:0'`
- **FastAPI:** `async def` routes, `AsyncSession`, Pydantic v2 `ConfigDict`
- **Vue 3:** `<script setup>` only, Pinia, Tailwind dark theme (slate/zinc + `#22c55e` / `#ef4444`)
- **Celery:** Q=`pipeline` (inference-engine, concurrency=1), Q=`training` (training-engine, concurrency=1); all tasks idempotent, Redis broker; GPU worker on Windows requires `--pool=solo` (billiard prefork fails with WinError 5/6)
- **DB:** PostgreSQL 16, `ukraine_footage`, SQLAlchemy 2.x
- **Scraping:** Funker530 REST + GeoConfirmed REST + yt-dlp, SHA256 url_hash dedup
- **See** `rules/` for coding standards: `celery-rules.md`, `fastapi-rules.md`, `vue3-rules.md`, `yolo-rules.md`
- **See** `agents/` for detailed domain context and DB schema

---

## Key Files

| What | Where |
|------|-------|
| Shared ORM models | `shared/db/models.py` |
| GDINO auto-label (video clips) | `inference-engine/core/autolabeling/auto_label.py` |
| YOLO inference + multi-model | `inference-engine/core/inference.py` |
| Auto-label (any image folder) | `inference-engine/tasks/autolabel_kaggle.py` |
| Auto-label Celery task | `inference-engine/tasks/auto_label.py` |
| Package dataset + finetune trigger | `inference-engine/tasks/package_dataset.py` |
| Annotate clips Celery task | `inference-engine/tasks/annotate_clips.py` |
| Weight resolution helpers | `inference-engine/tasks/weights.py` |
| Inference-engine config | `inference-engine/config.py` |
| GCS / local storage finalize | `inference-engine/core/storage.py` |
| Training entry point | `training-engine/core/main.py` |
| Finetune training task | `training-engine/tasks/train_finetune.py` |
| Baseline training task | `training-engine/tasks/train_baseline.py` |
| Build merged datasets (run once) | `training-engine/scripts/build_specialist_datasets.py` |
| AIRCRAFT pipeline (DB-driven) | `training-engine/scripts/aircraft_pipeline.py` |
| VEHICLE pipeline (DB-driven) | `training-engine/scripts/vehicle_pipeline.py` |
| PERSONNEL pipeline (DB-driven) | `training-engine/scripts/personnel_pipeline.py` |
| GENERAL pipeline (DB-driven) | `training-engine/scripts/general_pipeline.py` |
| One-time dataset setup | `training-engine/scripts/setup_datasets.sh` |
| Start all Celery workers (dev) | `start_workers.sh` |
| Funker530 scraper | `scraper-engine/tasks/scrape_funker530.py` |
| GeoConfirmed scraper | `scraper-engine/tasks/scrape_geoconfirmed.py` |
| Content filter + scoring | `scraper-engine/utils/_filter.py` |
| Phase 1 quick sample test | `scraper-engine/tests/test_scrape_sample.py` |
| Phase 1 24h window test | `scraper-engine/tests/test_scrape_24h.py` |
| Daily scrape orchestration | `scraper-engine/scripts/scrape_daily.py` |
| Inference-engine unit tests | `inference-engine/tests/unit/` |
| Inference-engine integration tests | `inference-engine/tests/integration/` |
| Training-engine unit tests | `training-engine/tests/unit/` |
| Project plan | `PROJECT_PLAN.md` |

Run Phase 1 sample test: `cd scraper-engine && python tests/test_scrape_sample.py`
Run Phase 1 24h test: `cd scraper-engine && python tests/test_scrape_24h.py`
Run inference-engine tests: `cd inference-engine && python -m pytest tests/`
Run training-engine tests: `cd training-engine && python -m pytest tests/`

---

## Phase Status

| Phase | Focus | Status |
|-------|-------|--------|
| 0 | Agentic workspace | ✅ Complete |
| 1 | Scraper engine | ✅ Complete |
| 2 | ML pipeline — baseline training | ✅ Complete (AIRCRAFT 0.929, VEHICLE 0.871, PERSONNEL 0.780, GENERAL 0.784) |
| 3 | Web application | ✅ Complete (Celery E2E, hero video, WebSocket progress bar, integration smoke test — 58 annotated clips) |
| 4 | Cloud & DevOps | ✅ Complete (GCP e2-micro ✅; T4 Spot VM ✅; GCS pipeline ✅; HTTPS ✅ ukrarchive.duckdns.org Let's Encrypt; CI/CD ✅ GitHub Actions deploy-on-push; mobile-responsive frontend ✅) |

---

## Agent Usage — Modus Operandi

Slash commands in `.claude/commands/` wire up the `agents/` domain docs. Invoke them via the `Skill` tool.

### Available commands
| Command | Domain | Agent doc |
|---------|--------|-----------|
| `/review-webapp` | `web-app/` code review | `agents/web-app/review.md` |
| `/review-ml` | `inference-engine/` + `training-engine/` code review | `agents/ml-pipeline/review.md` |
| `/review-scraper` | `scraper-engine/` code review | `agents/ingestion/review.md` |
| `/qa-webapp` | API contract + frontend QA | `agents/web-app/qa.md` |
| `/qa-pipeline` | End-to-end DB state + pipeline health | `agents/ml-pipeline/qa.md` + `agents/ingestion/qa.md` |
| `/qa-scraper` | Ingestion integrity | `agents/ingestion/qa.md` |
| `/research-webapp` | Before new web-app features | `agents/web-app/research.md` |
| `/research-ml` | Before new ML features | `agents/ml-pipeline/research.md` |
| `/research-scraper` | Before new scraper features | `agents/ingestion/research.md` |
| `/research-deploy` | Before new infra/cloud changes | `agents/cloud-deploy/research.md` |
| `/review-deploy` | Docker Compose + Dockerfile review | `agents/cloud-deploy/review.md` |
| `/qa-deploy` | Production deployment health check | `agents/cloud-deploy/qa.md` |

### When to spawn (demand criteria)

**Review agents** — spawn after a commit when ANY of:
- New admin endpoint or auth-touching code added
- Non-trivial architectural decision made during implementation
- A bug in this area was already hit this session
- Something felt hacky or required a workaround

**QA agents** — spawn when:
- New API endpoint added (verify contract)
- DB enum or column type changed (not just nullable column additions)
- A bug fix touches a path with no tests

**Research agents** — spawn BEFORE implementing when:
- New technology or pattern not yet used in this codebase (first WebSocket, first migration, etc.)
- Getting the approach wrong means a rewrite later

**Skip entirely for:** CSS/styling changes, text fixes, simple prop additions, anything obviously correct and self-contained.

### Rule of thumb
If a senior engineer would want to glance at it before merging — spawn. Roughly 1 in 3 commits warrants a review agent.

---

## Test Rules (Non-Negotiable)

1. **Never touch existing DB rows** — a test only operates on rows it inserted itself, tracked by ID from the moment of insertion
2. **Clean up everything in `finally`** — every DB row inserted, every file written to disk, unconditionally, even on failure
3. **Log all relevant output** — every URL scraped, every file downloaded with name/size/duration, every DB row inserted/deleted, every model result with detection counts
4. **Never bulk-query by source/status to find "your" rows** — always filter by the IDs you tracked at insert time

---

## Do NOT

- Run inference-engine or training-engine workers in Docker locally — GPU passthrough requires NVIDIA Container Toolkit (Linux only)
- Use Options API or Vuex in Vue components
- Use sync SQLAlchemy in FastAPI routes
- Use `print()` in production code — use `logging`
- Hardcode credentials or absolute file paths
- Start GPU workers with `concurrency > 1`
- Write code for phases not yet reached (stay sequential)
- Commit without explicit user approval
