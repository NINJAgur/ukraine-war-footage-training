# PROJECT_PLAN.md — Ukraine Combat Footage Web Application
> **Source of Truth** — All phases, structure, and decisions are tracked here.
> Last updated: 2026-05-31

---

## Table of Contents
1. [Architecture Overview](#1-architecture-overview)
2. [Host Machine Setup Guide](#2-host-machine-setup-guide)
3. [Directory Structure](#3-directory-structure)
4. [Master To-Do List](#4-master-to-do-list)
5. [Docker Desktop Quick-Start](#5-docker-desktop-quick-start)
6. [Next Steps](#6-next-steps)

---

## 1. Architecture Overview

### 1.1 Project Goal
An automated, full-stack web application that:
- Scrapes combat footage from open-source sites (Funker530, GeoConfirmed) on a schedule
- Runs YOLOv8 auto-labeling on every downloaded clip
- Packages labeled frames into YOLO/Kaggle-compatible datasets
- Renders annotated MP4 previews for a public media dashboard
- Provides a secure Admin command center to trigger two-stage model retraining

### 1.2 Engine Lifecycle — Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│         INGESTION LAYER  (scraper-engine — GCP e2-micro)            │
│                                                                     │
│  [Celery Beat — daily 00:00 UTC]                                    │
│       ├──► scrape_funker530   (REST API → score → yt-dlp download)  │
│       └──► scrape_geoconfirmed (REST API → score → yt-dlp download) │
│                    │                                                │
│   Raw .mp4 → uploaded to GCS: raw/<source>/<date>/<hash>.mp4       │
│   clip.file_path = gs://ukraine-footage-media/raw/...              │
│   Clip row written to PostgreSQL, status=DOWNLOADED                 │
└─────────────────────────────────────────────────────────────────────┘
                     │ (Redis Celery queue via GCP internal IP)
                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│   INFERENCE LAYER  (inference-engine — n1-standard-1 + T4, Q=pipeline) │
│   Instance Schedule: 03:00 start / 04:00 stop UTC  (1-hour window) │
│   [1 T4 quota — inference MUST stop before training starts]        │
│                                                                     │
│  [Beat @03:05] auto_label_batch → auto_label_clip × N              │
│       └──► Phase 1 GDINO: frames → canonical labels → Dataset(LABELED)  │
│            Phase 2 package_dataset (per clip):                      │
│              → filter+append into merged/<MODEL>/ (4 persistent dirs)   │
│              → delete clip hash dir immediately                     │
│              → Dataset(PACKAGED)                                    │
│            Phase 3 chord callback (once, after ALL clips done):     │
│              → count scraped imgs/model; threshold met → TrainingRun(QUEUED)│
│              → ONE prepare_finetune_batch dispatch (all run IDs)    │
│              → mark consumed datasets TRAINED                       │
│            Phase 4 prepare_finetune_batch (Q=pipeline):             │
│              → remote: upload merged/<MODEL>/ to GCS → delete local │
│              → local: leave merged dirs on disk                     │
│              → dispatch train_finetune × N (NO VM start — training  │
│                engine has its own Instance Schedule at 04:30)       │
│                                                                     │
│  [Beat @03:35] annotate_clips (Q=pipeline, waits behind GDINO):    │
│       └──► YOLO inference (AIRCRAFT→VEHICLE→PERSONNEL→GENERAL)     │
│            _download_from_gcs(clip.file_path) → /tmp/<hash>.mp4    │
│            validate_clip (≥10% frames detected at conf=0.25)       │
│                PASS → infer_video_multi_model → annotated MP4       │
│                     → upload to GCS: annotated/<model>/<date>/...  │
│                     → clip.mp4_path = https://storage.googleapis.com/...
│                     → delete raw GCS object                        │
│                FAIL → delete raw GCS object, status=PENDING        │
│            _shutdown_if_no_training (shuts VM if no active runs)   │
└─────────────────────────────────────────────────────────────────────┘
                     │ (QUEUED TrainingRuns in DB — no direct dispatch)
                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│   TRAINING LAYER  (training-engine — n1-standard-4 + T4, Q=training) │
│   Instance Schedule: 04:30 UTC start (30-min buffer after inference) │
│   Self-shutdown: after last model trains, or immediately if no work │
│                                                                     │
│  Startup script (before Celery):                                    │
│       → query DB for QUEUED TrainingRuns                            │
│       → if none → sudo shutdown -h now  (avoids idle GPU cost)     │
│                                                                     │
│  train_finetune × 4 (AIRCRAFT, VEHICLE, PERSONNEL, GENERAL):        │
│       download gs://bucket/merged/<MODEL>_<run_id>/ → local         │
│       combined_data.yaml: kaggle_merged/ + merged_dir               │
│       YOLO training → best.pt                                       │
│       upload best.pt → gs://bucket/runs/finetune/...               │
│       finally: delete local merged dir                              │
│  After last model: sudo shutdown -h now                             │
└─────────────────────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│         PUBLIC DASHBOARD  (web-app/frontend — GCP e2-micro)         │
│                                                                     │
│   GET /api/annotated-clips → ArchiveSection / Archive.vue          │
│   GET /api/stats           → TickerBar, MLCard HUD                 │
│   <video src="https://storage.googleapis.com/..."> (direct GCS)    │
└─────────────────────────────────────────────────────────────────────┘
                     │
                     ▼ (future: fine-tune loop)
┌─────────────────────────────────────────────────────────────────────┐
│                   ADMIN TRAINING LAYER                              │
│                                                                     │
│  POST /api/admin/train → TrainingRun(QUEUED) → [train_baseline]    │
│                                              → [train_finetune]    │
│                                                                     │
│  Stage 1 (baseline):  8 Kaggle datasets → specialist best.pt       │
│  Stage 2 (finetune):  accumulated annotated clips → fine_tuned.pt  │
│                                                                     │
│  TrainingRun metrics persisted to DB; AdminPanel shows live status  │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.3 Two-Stage Training Strategy

| Stage | Task | Data Source | Output |
|-------|------|-------------|--------|
| **Stage 1 — Baseline** | `train_baseline.py` | Kaggle military datasets (`sudipchakrabarty/kiit-mita` + others) | `runs/baseline/weights/best.pt` |
| **Stage 2 — Fine-Tune** | `train_finetune.py` | Auto-labeled custom datasets from the pipeline | `runs/finetune/weights/best.pt` |

- Stage 1 builds general military-object vocabulary (vehicles, personnel, weapons)
- Stage 2 specializes on the exact visual style of scraped footage
- Admin can trigger either stage independently; Stage 2 loads Stage 1's `.pt` as initial weights
- Celery GPU worker: `concurrency=1` to prevent VRAM contention on RTX 3060 Ti (8GB)

### 1.4 ML Foundation (Existing Repo Migration)

| Original File | Migrates To | Role |
|---------------|-------------|------|
| `scripts/main.py` | `ml-engine/core/main.py` | YOLO training entry point |
| `scripts/inference.py` | `ml-engine/core/inference.py` | Video/image inference → annotated output |
| `autolabeling/auto-label.py` | `ml-engine/core/autolabeling/auto_label.py` | GroundingDINO zero-shot auto-labeling |
| `scripts/preprocessing.py` | `ml-engine/core/preprocessing.py` | Data cleaning + augmentation |
| `scripts/dataset_explorer.py` | `ml-engine/core/dataset_explorer.py` | Dataset stats/visualization |

**Deleted (legacy):** `streamlit_app.py`, `scripts/face_blurring.py`, `scripts/select_blurring.py`

### 1.5 Tech Stack

| Layer | Technology |
|-------|-----------|
| **Hardware** | Windows 11, i5-13600KF, RTX 3060 Ti 8GB, CUDA 12.1 via pip |
| **Backend API** | FastAPI + SQLAlchemy + PostgreSQL |
| **Frontend** | Vue 3 (Composition API) + Vite + Tailwind CSS + Pinia |
| **Scraping** | `yt-dlp` + Funker530 REST API + GeoConfirmed REST API + Kaggle API |
| **Async Queue** | Celery + Redis (broker + result backend) |
| **ML** | Ultralytics YOLOv8 + PyTorch (`torch+cu121`) + OpenCV |
| **Containers** | Docker + Docker Compose (scraper/backend/frontend in Docker Desktop; ml-engine runs natively) |
| **Cloud** | GCP e2-micro free tier (CPU, $0/mo) + 2× T4 Spot VMs: inference-engine (n1-std-1) + training-engine (n1-std-4, ~$20/mo total) |
| **IaC** | Terraform (GCS bucket, e2-micro, inference-engine + training-engine VMs + Instance Schedules) |

---

## 2. Host Machine Setup Guide

> **Dev model:** Native Windows 11 + VSCode. Training via `torch+cu121` pip package —
> no standalone CUDA Toolkit required. Docker + NVIDIA Container Toolkit deferred to **Phase 4**.

### Step 1 — Python 3.11
Download the Python 3.11 installer from python.org and check **"Add Python to PATH"**.
```powershell
python --version   # expected: Python 3.11.x
```

### Step 2 — PyTorch with CUDA 12.1 (GPU Training Support)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
Verify GPU:
```python
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# expected: True  NVIDIA GeForce RTX 3060 Ti
```

### Step 3 — Git
Already present. Verify: `git --version`

### Step 4 — Node.js 20 LTS
Download from nodejs.org.
```bash
node --version   # expected: v20.x.x
```

### Step 5 — Redis (Local Dev)
```bash
wsl --install   # enable WSL2 if not active
# inside Ubuntu:
sudo apt update && sudo apt install -y redis-server
redis-server --daemonize yes && redis-cli ping   # PONG
```

### Step 6 — PostgreSQL
Download PostgreSQL 16 Windows installer from postgresql.org (port 5432).
Create DB: `createdb ukraine_footage`

### Step 7 — Kaggle API Credentials
1. kaggle.com → Account → API → "Create New API Token"
2. Place `kaggle.json` at `%USERPROFILE%\.kaggle\kaggle.json`

### Step 8 — yt-dlp
```bash
pip install yt-dlp
```

### Step 9 — GCP SDK *(Phase 4 only)*
Install `gcloud` CLI from cloud.google.com/sdk

### Step 11 — Docker Desktop + NVIDIA Container Toolkit *(Phase 4 only)*
- Docker Desktop with WSL2 backend
- NVIDIA Container Toolkit inside WSL2 Ubuntu

---

## 3. Directory Structure

```
yolo-training-template/                  ← monorepo root
│
├── PROJECT_PLAN.md                      ← THIS FILE — source of truth
├── CLAUDE.md                            ← Claude Code persistent system prompt
├── .env                                 ← environment variables (gitignored)
├── docker-compose.yml                   ← local dev stack (scraper + backend + frontend; no ml-worker)
├── docker-compose.prod.yml              ← GCP e2-micro prod deploy (CPU services only)
│
├── .claude/                             ← Claude Code agentic workspace
│   └── settings.json                    ← permissions, hooks, MCP config (gitignored)
│
├── agents/                              ← multi-agent swarm definitions
│   ├── ingestion/
│   │   ├── research.md
│   │   ├── qa.md
│   │   └── review.md
│   ├── ml-pipeline/
│   │   ├── research.md
│   │   ├── qa.md
│   │   └── review.md
│   ├── web-app/
│   │   ├── research.md
│   │   ├── qa.md
│   │   └── review.md
│   └── cloud-deploy/
│       ├── research.md                  ← GCP e2-micro+T4 Spot architecture, env vars
│       ├── review.md                    ← Docker Compose + Dockerfile review checklist
│       └── qa.md                        ← production health verification commands
│
├── rules/                               ← enforced coding standards per domain
│   ├── vue3-rules.md
│   ├── fastapi-rules.md
│   ├── yolo-rules.md
│   └── celery-rules.md
│
├── commands/                            ← custom Claude Code slash-commands
│   ├── scrape.md
│   ├── train.md
│   └── annotate.md
│
│
├── scraper-engine/                      ← PHASE 1: Data Ingestion ✅ Complete
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── celery_app.py
│   ├── beat_schedule.py
│   ├── config.py
│   ├── tasks/
│   │   ├── _filter.py                   ← shared content filter (equipment + negative gate)
│   │   ├── scrape_funker530.py          ← Funker530 REST API + filter + yt-dlp
│   │   ├── scrape_geoconfirmed.py       ← GeoConfirmed REST API + parallel fetch + filter + yt-dlp
│   │   └── download_kaggle.py
│   ├── db/
│   │   ├── session.py
│   │   └── models.py
│   ├── scripts/
│   │   └── scrape_daily.py              ← daily orchestration (calls _since functions)
│   └── tests/
│       ├── test_scrape_sample.py        ← Phase 1 sample test (max_count/max_incidents)
│       └── test_scrape_24h.py           ← Phase 1 24h window test (calls _since functions)
│
├── inference-engine/                    ← GDINO + YOLO annotation + dataset pipeline (Q=pipeline)
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── celery_app.py
│   ├── config.py                        ← includes GCP_PROJECT_ID, GCP_TRAINING_VM_* settings
│   ├── media/                           ← scraped_datasets/, annotated/ (gitignored)
│   ├── tasks/
│   │   ├── auto_label.py                ← GDINO on video clips (Celery, Q=pipeline)
│   │   ├── package_dataset.py           ← 80/20 split + _maybe_trigger_finetune + prepare_finetune_batch
│   │   ├── annotate_clips.py            ← YOLO annotation (Q=pipeline, @03:35 UTC)
│   │   └── weights.py                   ← _latest_weights, _resolve_weights_path (shared helpers)
│   ├── core/
│   │   ├── inference.py                 ← multi-model video inference → annotated MP4
│   │   ├── storage.py                   ← GCS / local finalize_clip
│   │   └── autolabeling/
│   │       └── auto_label.py            ← GroundingDINO labeling (video clips → frames)
│   ├── db/
│   │   ├── session.py
│   │   └── models.py
│   └── tests/
│
├── training-engine/                     ← YOLO training only (Q=training, on-demand VM)
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── celery_app.py
│   ├── config.py
│   ├── runs/                            ← YOLO training output (gitignored)
│   ├── tasks/
│   │   ├── train_baseline.py            ← Stage 1: Kaggle datasets → specialist .pt files
│   │   └── train_finetune.py            ← Stage 2: download merged from GCS → fine_tuned.pt
│   ├── core/
│   │   └── main.py                      ← YOLO training entry point
│   ├── scripts/
│   │   ├── build_specialist_datasets.py ← ONE-TIME: builds kaggle_datasets/merged/
│   │   ├── setup_datasets.sh            ← VM startup: Kaggle download + persistent disk setup
│   │   ├── aircraft_pipeline.py         ← DB-driven AIRCRAFT annotation pipeline
│   │   ├── vehicle_pipeline.py
│   │   ├── personnel_pipeline.py
│   │   └── general_pipeline.py
│   ├── db/
│   │   ├── session.py
│   │   └── models.py
│   └── tests/
│
├── web-app/                             ← PHASE 3: Web Application ✅ Complete
│   ├── backend/
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   ├── main.py
│   │   ├── api/
│   │   │   ├── public.py
│   │   │   └── admin.py
│   │   ├── db/
│   │   │   ├── session.py
│   │   │   └── models.py
│   │   └── schemas/
│   │       ├── clip.py
│   │       ├── dataset.py
│   │       └── training.py
│   └── frontend/
│       ├── Dockerfile
│       ├── package.json
│       ├── vite.config.js
│       ├── tailwind.config.js
│       └── src/
│           ├── main.js
│           ├── App.vue
│           ├── router/index.js
│           ├── stores/
│           │   ├── feed.js
│           │   └── admin.js
│           ├── views/
│           │   ├── PublicFeed.vue
│           │   ├── Archive.vue
│           │   ├── Submit.vue
│           │   └── admin/
│           │       ├── AdminLogin.vue
│           │       └── AdminPanel.vue
│           └── components/
│               ├── AppNav.vue
│               ├── HeroSection.vue
│               ├── TickerBar.vue
│               ├── MLDetectionSection.vue
│               ├── MLCard.vue
│               ├── ArchiveSection.vue
│               ├── SiteFooter.vue
│               └── ... (see 3.9 for full list)
│
├── infra/                               ← GCP Terraform (task 4.19 ✅)
│   ├── gcp/
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   └── upload_weights.py            ← uploads runs/ weights to GCS bucket
│   └── nginx/
│       └── nginx.conf
│
├── .github/                             ← GitHub Actions CI/CD (task 4.21 ✅)
│   └── workflows/
│       ├── ci.yml                       ← frontend build + ruff lint on push/PR
│       ├── deploy-e2-micro.yml          ← SSH deploy after CI passes (workflow_run)
│       ├── deploy-inference-engine.yml  ← manual: SSH deploy to inference-engine VM
│       └── deploy-training-engine.yml   ← manual: SSH deploy to training-engine VM
│
```

> Tests live inside each service directory (`scraper-engine/tests/`, `inference-engine/tests/`, `training-engine/tests/`, `web-app/tests/`) — not at the repo root.

---

## 4. Master To-Do List

### Phase 0 — Claude Code Agentic Workspace Init

- [x] **0.1** Delete legacy files: `streamlit_app.py`, `scripts/face_blurring.py`, `scripts/select_blurring.py`
- [x] **0.2** Create `CLAUDE.md` — project architecture, tech stack constraints, goals, phase map
- [x] **0.3** Create `.claude/settings.json` — permissions, hooks, MCP stubs
- [x] **0.4** Create `agents/ingestion/research.md`
- [x] **0.5** Create `agents/ingestion/qa.md`
- [x] **0.6** Create `agents/ingestion/review.md`
- [x] **0.7** Create `agents/ml-pipeline/research.md`
- [x] **0.8** Create `agents/ml-pipeline/qa.md`
- [x] **0.9** Create `agents/ml-pipeline/review.md`
- [x] **0.10** Create `agents/web-app/research.md`
- [x] **0.11** Create `agents/web-app/qa.md`
- [x] **0.12** Create `agents/web-app/review.md`
- [x] **0.13** Create `rules/vue3-rules.md`
- [x] **0.14** Create `rules/fastapi-rules.md`
- [x] **0.15** Create `rules/yolo-rules.md`
- [x] **0.16** Create `rules/celery-rules.md`
- [x] **0.17** Create `commands/scrape.md`
- [x] **0.18** Create `commands/train.md`
- [x] **0.19** Create `commands/annotate.md`
- [x] **0.20** Create `.env.example`
- [x] **0.21** Create `docker-compose.yml` skeleton (postgres + redis)
- [x] **0.22** Commit: `git commit -m "chore(phase-0): init agentic workspace"`

---

### Phase 1 — Data Ingestion ✅ Complete

- [x] **1.1** Scaffold `scraper-engine/` + `requirements.txt`
- [x] **1.2** Create `celery_app.py` with Redis broker config
- [x] **1.3** Create `db/session.py` + `models.py` (`Clip` ORM)
- [x] **1.4** Implement `tasks/scrape_funker530.py` (Funker530 REST API, multi-field description fallback, yt-dlp)
- [x] **1.5** Implement `tasks/scrape_geoconfirmed.py` (GeoConfirmed REST API, parallel detail fetch with ThreadPoolExecutor, gear+units metadata, yt-dlp)
- [x] **1.6** Implement `tasks/download_kaggle.py` (Kaggle API — legacy; Kaggle downloads are now manual via `scripts/download_new_datasets.py`)
- [x] **1.7** Configure `beat_schedule.py` (daily scrape: funker530 00:00 UTC, geoconfirmed 00:15 UTC)
- [x] **1.8** Write `scraper-engine/Dockerfile`
- [x] **1.9** Implement `utils/_filter.py` — shared content filter: equipment keyword gate (regex + word boundaries, specific hardware names) + negative rejection gate (fire, smoke, ruins, wreckage — blocks aftermath footage, not target type)
- [x] **1.10** Live scrape + download test passed (2026-04-18): 4/4 tests pass. Funker530 (5 clips, all valid Ukraine drone/infantry footage), GeoConfirmed (5 clips, all valid FPV/UAV strike footage). Impact filter correctly rejects refinery smoke plumes and aftermath videos.

---

### Phase 2 — ML Pipeline

#### 2a — Core Tasks (complete)
- [x] **2.1** Scaffold `ml-engine/` + `requirements.txt`
- [x] **2.2** Migrate `core/` scripts from existing repo (`main.py`, `inference.py`, `preprocessing.py`, `dataset_explorer.py`, `autolabeling/auto_label.py`)
- [x] **2.3** Create `celery_app.py` (concurrency=1, GPU queue, Beat schedule every 5 min)
- [x] **2.4** Implement `tasks/poll_clips.py` — Beat bridge: DOWNLOADED → QUEUED → dispatches auto_label_clip
- [x] **2.5** Implement `tasks/auto_label.py` — extract frames + GroundingDINO → YOLO .txt labels → Dataset(LABELED) → dispatch package_dataset
- [x] **2.6** Implement `tasks/package_dataset.py` — 80/20 train/val split + data.yaml → Dataset(PACKAGED) → dispatch render_annotated
- [x] **2.7** Implement `tasks/render_annotated.py` — YOLO inference on raw clip → H.264 MP4 → Clip(ANNOTATED)
- [x] **2.8** Implement `tasks/train_baseline.py` — Stage 1: Kaggle datasets → best.pt
- [x] **2.9** Implement `tasks/train_finetune.py` — Stage 2: merge custom datasets → fine-tune from baseline → best.pt
- [x] **2.10** Write `ml-engine/Dockerfile` (NVIDIA PyTorch base, GroundingDINO from source)
- [x] **2.11** Add `kagglehub` to `ml-engine/requirements.txt`
- [x] **2.12** Add `ClipStatus.QUEUED` to both `scraper-engine/db/models.py` and `ml-engine/db/models.py`

#### 2b — Multi-Model Architecture ✅
- [x] **2.13** `ModelType` enum (GENERAL, AIRCRAFT, VEHICLE, PERSONNEL) + `model_type` column on `TrainingRun`
- [x] **2.14** `config.py` multi-model setup — superseded by 2.26 (3-class redesign)
- [x] **2.15** `train_baseline.py` per-model datasets + `_merge_datasets()` — superseded by 2.27
- [x] **2.16** `train_finetune.py` per-model class filtering — superseded by 2.30
- [x] **2.17** `infer_video_multi_model()` in `core/inference.py` — 4-model sequential rendering, colour-coded bboxes
- [x] **2.18** `render_annotated.py` `_best_weights_per_model()` — superseded by 2.29
- [x] **2.19** Initial 5 Kaggle datasets on disk and verified: kiit-mita (1700, nc=7→remap), mihprofi (37900, nc=2→remap), shakedlevnat (19958, nc=83→remap), nzigulic (16809, nc=11→GDINO), piterfm (31041, no labels→GDINO). 3 additional datasets added later: rookieengg (11788, nc=43→AIRCRAFT), rawsi18 (26315, nc=12→remap), amad-5 (34960, nc=5→remap)

#### 2c — Infrastructure + First Test ✅
- [x] **2.20** `test_pipeline_e2e.py` — real clip required; render → annotated MP4; `--keep`/`--purge-outputs`
- [x] **2.21** `test_baseline_train.py` — smoke test with `--model-type`, `--epochs`, `--keep`, `--purge-outputs`
- [x] **2.22** DB tables bootstrapped via `Base.metadata.create_all()`
- [x] **2.23** E2E render test passed: 22.8MB annotated MP4, 1838 frames, pretrained weights
- [x] **2.24** VEHICLE baseline smoke test passed: nc=3, mAP50=0.405 @ 2 epochs

#### 2e — Taxonomy Redesign ✅
**Decision:** 3 universal classes aligned with `_filter.py`.

| ID | Class | Covers |
|----|-------|--------|
| 0 | AIRCRAFT | drones, helicopters, fixed-wing, missiles, glide bombs |
| 1 | VEHICLE | tanks, APCs, artillery, radar, MLRS, all ground military vehicles |
| 2 | PERSONNEL | soldiers, fighters, RPG/ATGM operators |

| Dataset | Pipeline role |
|---------|--------------|
| kiit-mita | YOLO labels remapped → nc=3 (baseline) |
| mihprofi/drone-detect | YOLO labels remapped → AIRCRAFT (baseline) |
| shakedlevnat | YOLO labels remapped → AIRCRAFT (baseline) |
| nzigulic | GDINO auto-label → nc=3 (fine-tune corpus) |
| piterfm | GDINO auto-label → nc=3 (fine-tune corpus) |

- [x] **2.25** `ModelType` SOLDIER→PERSONNEL; DB migration applied
- [x] **2.26** `config.py` — 15-term "." GDINO prompt; `YOLO_EPOCHS_BASELINE=10` (incremental)
- [x] **2.27** `train_baseline.py` — nc=3 canonical remap; nzigulic/piterfm removed from baseline datasets
- [x] **2.28** `auto_label.py` — "." separator; post-GDINO 15→3 canonical remap; data.yaml nc=3
- [x] **2.29** `render_annotated.py` + `inference.py` — 3-class colour map (PERSONNEL replaces SOLDIER)
- [x] **2.30** `train_finetune.py` — identity `_class_remap` (nc=3 pre-remapped on disk); SOLDIER→PERSONNEL

#### Step 1 — Install GDINO + auto-label nzigulic + piterfm

> nzigulic already has YOLO-format labels on disk (kagglehub reorganized `images_test/labels_test/` → standard `test/images/test/labels/`). Class names are anonymous (`class_0`–`class_10`, nc=11) — identify via visual inspection, add remap to `train_baseline.py`.  
> piterfm has zero labels — GDINO required.

- [x] **2.31** Install GroundingDINO: `pip install groundingdino-py` + checkpoint `groundingdino_swint_ogc.pth` (661MB, gitignored)
- [x] **2.32** Implement `tasks/autolabel_kaggle.py` — GDINO batch labeling on image folders; canonical nc=3 remap; substring fallback for merged GDINO phrases; outputs to `media/kaggle_datasets/labeled/<name>/`
- [x] **2.33** nzigulic: identify nc=11 anonymous class mapping via bbox visualization on sample images → add `"nzigulic/military-equipment"` entry to `DATASET_CLASS_MAPS` + `BASELINE_DATASETS` in `train_baseline.py`
- [x] **2.34** piterfm: GDINO auto-label all ~27k images → nc=3 YOLO dataset
  - Initial run (5k, generic prompt): 4168/5000 labeled (84%); 796 no-detections
  - Targeted re-label of 796 no-detections (category-aware prompts): +683 recovered → 97% coverage
  - Full run: `tasks/relabel_piterfm.py` on all 27,714 images → 26,226 labeled / 1,488 empty / 0 failed (94.6%)
  - Empty-label images moved to `kaggle_datasets/to_annotate_manually/` (destroyed/satellite imagery)
  - Labeled dataset at `piterfm/2022-ukraine-russia-war-equipment-losses-oryx/versions/1/train/` (26,197 images, 26,118 labels); raw source deleted
  - `"piterfm/2022-ukraine-russia-war-equipment-losses-oryx"` identity map added to `DATASET_CLASS_MAPS` + `BASELINE_DATASETS` (AIRCRAFT, VEHICLE, GENERAL)
- [x] **2.35** Spot-check label quality: nzigulic validated via contact sheets (all 11 classes identified); piterfm validated via 94.6% GDINO detection rate with category-aware prompts

#### Dataset Inventory (8 datasets, all on disk)

| Kaggle handle | nc | Images | Notes |
|---|---|---|---|
| `mihprofi/drone-detect` | 2 | 36,013 | Both classes → AIRCRAFT; fresh download 2026-05-14 |
| `shakedlevnat/military-aircraft-database` | 83 | 17,962 | All 83 → AIRCRAFT; fresh download 2026-05-14 |
| `nzigulic/military-equipment` | 11 | 13,448 | Anonymous nc=11: 4-7→AIRCRAFT, 0-3/8-10→VEHICLE; reorganized to train/val layout |
| `piterfm/2022-ukraine-russia-war-equipment-losses-oryx` | 3 | 26,197 | Canonical nc=3 pass-through; GDINO labels |
| `sudipchakrabarty/kiit-mita` | 7 | 1,530 | 7-class remap → nc=3; fresh download 2026-05-14 |
| `rookieengg/military-aircraft-detection` | 43 | 11,788 | All 43 → AIRCRAFT; reorganized to train/val layout |
| `rawsi18/military-assets-dataset-12-classes` | 12 | 24,919 | 12-class remap → nc=3 (4 classes skipped); fresh download 2026-05-14 |
| `rupankarmajumdar/amad-5` | 5 | 32,529 | 5-class remap → nc=3 (civilians skipped); fresh download 2026-05-14 |
| **TOTAL** | | **164,386** | Source files never modified — remapping in build script only |

#### Per-Model Merged Dataset (post-filter counts, 2026-05-17 rebuild — specialist label filter applied)

> **Critical fix (2026-05-17):** `build_specialist_datasets.py` had a bug — specialist datasets included annotation lines from all remapped classes, not just the target class. Fixed: label text is filtered to target class only before writing. All 4 datasets rebuilt clean.

| Model | Source Datasets | Train | Val |
|---|---|---|---|
| **AIRCRAFT** | mihprofi, shakedlevnat, nzigulic, piterfm, rookieengg, rawsi18 | 65,557 | 9,382 |
| **VEHICLE** | kiit-mita, nzigulic, piterfm, rawsi18, amad-5 | 56,440 | 6,638 |
| **PERSONNEL** | kiit-mita, rawsi18, amad-5 | 10,962 | 1,302 |
| **GENERAL** | all 8 | 144,466 | ~19,920 |

#### Step 2 — Train specialists (8 Kaggle datasets as corpus)

- [x] **2.36** Run `test_baseline_train.py --model-type AIRCRAFT --epochs 10 --keep` — mAP50=0.929 @ epoch 10 (run 13, 65,557 train / 9,382 val) ✅
- [x] **2.36b** AIRCRAFT finetune cycle 1 — mAP50=0.968 @ epoch 8 (run 68, 65,557 train / 9,382 val) ✅
- [x] **2.36c** `ml-engine/scripts/aircraft_pipeline.py` — scrape→validate→annotate pipeline; `validate_clip()` in `core/inference.py` (generic, any model); detection-rate gate (≥10% frames at conf=0.25, 30 samples)
- [x] **2.37** Run `test_baseline_train.py --model-type VEHICLE --epochs 10 --keep` — mAP50=0.871 @ epoch 10 (run 25, 56,440 train / 6,638 val) ✅
- [x] **2.38** Run `test_baseline_train.py --model-type PERSONNEL --epochs 10 --keep` — mAP50=0.780 @ epoch 10 (run 29, contaminated dataset — kept as reference); clean rerun pending after 2026-05-17 dataset rebuild
- [x] **2.39** All 3 specialists evaluated: all mAP50 > 0.4 ✅

#### Step 3 — Train generalist

- [x] **2.40** Run `test_baseline_train.py --model-type GENERAL --epochs 10 --keep` — mAP50=0.784 @ epoch 10 (run 30, 144,466 train / 19,920 val) ✅

#### Step 4 — Tests

- [ ] **2.41** `test_pipeline_e2e.py` with trained weights → verify annotated MP4 quality improved
- [ ] **2.42** `test_scrape_sample.py` → full Phase 1→2 flow (scrape → download → render → annotated MP4)

---

### Phase 3 — Web Application

- [x] **3.1** Scaffold `web-app/backend/` + `requirements.txt`
- [x] **3.2** ORM models → `shared/db/models.py` (single source of truth); re-export stubs in ml-engine, scraper-engine, web-app
- [x] **3.3** Pydantic v2 schemas
- [x] **3.4** Public API endpoints
- [x] **3.5** Admin API endpoints (WebSocket TBD)
- [x] **3.6** JWT authentication
- [x] **3.7** Scaffold Vue 3 + Vite + Tailwind + Pinia + vue-router frontend
- [x] **3.8** Full dark tactical design — Space Grotesk + IBM Plex Mono; `#080a0b` base; amber `oklch(0.65 0.18 55deg)` accent; scanline + noise overlays; crosshair cursor
- [x] **3.9** `PublicFeed.vue` — full public homepage (assembled from components below)
  - `AppNav.vue` — fixed nav; logo mark; scroll-spy active section; "Admin Login" CTA
  - `HeroSection.vue` — full-bleed generalist ML canvas bg + `hero.mp4` video; hero headline + stats
  - `TickerBar.vue` — scrolling LIVE ticker with live status items
  - `MissionSection.vue` — 3-col mission statement grid
  - `DataStrip.vue` — 4-stat number strip (147K clips, 38+ countries, 2.4TB, 12K events)
  - `MLDetectionSection.vue` — 3 specialist ML detection cards with video backgrounds
  - `MLCard.vue` — expanding parallelogram/trapezoid card; IntersectionObserver scroll trigger; animated canvas bounding-box overlay (different style per category: generalist=boxes, aircraft=radar sweep+diamonds, personnel=skeleton, vehicles=tank detail)
  - `RadarCanvas.vue` — reusable animated SIGINT-node radar background
  - `ArchiveSection.vue` — footage grid; filter by detection class (Aircraft/Vehicle/Personnel) + source (Funker530/GeoConfirmed); search; click-to-open modal with Teleport
  - `FootageCard.vue` — card with meta overlay, play button hover, status tag
  - `FootageModal.vue` — fixed modal via `<Teleport to="body">`; video placeholder; metadata grid
  - `CapabilitiesSection.vue` — 2×2 grid: Automated Ingestion / YOLO Detection / GDINO Labeling / Open Archive
  - `AboutSection.vue` — project info + tech stack list
  - `SiteFooter.vue` — 4-col footer
- [x] **3.10** `Archive.vue` — dedicated `/archive` page
  - Paginated grid (20/page) with page controls
  - Filters: detection class pill buttons + source buttons + search input (mirrors `ArchiveSection` controls)
  - Folder/category sidebar: group by `source` (Funker530 / GeoConfirmed) and `det_class` (AIRCRAFT / VEHICLE / PERSONNEL); click to filter
  - URL query params (`?class=AIRCRAFT&source=funker530&q=drone`) so links are shareable
  - Data from `GET /api/annotated-clips`; scroll-to-top fixed with `nextTick` + `behavior:instant`
  - `ArchiveSection.vue` "View All" button routes to `/archive`; "Browse Archive" hero button routes to `/archive`
- [x] **3.11** `Submit.vue` — footage submission form (URL input → `POST /api/submit`); submitted clips land as `status=REVIEW`, admin approves via `POST /api/admin/clips/{id}/approve`

#### Frontend ↔ Backend Integration

- [x] **3.16** Replace hardcoded `FOOTAGE_DATA` in `ArchiveSection.vue` + `Archive.vue` with live `GET /api/annotated-clips`; constants.js stripped to visual-only
- [x] **3.17** `AdminPanel.vue` — clips table wired to `GET /api/admin/clips`; paginated; REVIEW filter + APPROVE button for submitted clips; panel scrollable (`overflow-y: auto`)
- [x] **3.18** `AdminPanel.vue` — training runs table wired to `GET /api/admin/training-runs`; `map50` normalized via `model_validator` in `TrainingRunOut`
- [x] **3.19** `AdminPanel.vue` — train buttons wired to `POST /api/admin/train`; dispatches Celery task on gpu queue; 202 Accepted
- [x] **3.20** End-to-end auth flow — JWT login/logout; router guard; `.env` credentials; `admin` / `admin123`
- [x] **3.23** `TickerBar.vue` + `MLCard` + `DataStrip` + `CapabilitiesSection` — all pull live data from `GET /api/stats`; `GET /api/stats` reads from `TrainingRun.metrics` DB column
- [x] **3.24** `FootageCard.vue` — inline video player with controls when `videoUrl` present; modal not opened on video interaction
- [x] **3.25** `FootageModal.vue` — title truncation with ellipsis; close button flex-shrink fix (no overflow on long titles)
- [x] **3.26** `MLCard.vue` — canvas random-box overlay hidden (`v-if="!cat.videoSrc"`) when real annotated video is present
- [x] **3.27** Pipeline weights: all 3 pipelines use `_latest_weights(model_name)` — auto-selects highest-numbered run dir with `best.pt`; `ClipOut` det_class reads directly from DB (no regex override); `video_url` correctly includes full subdir path
- [x] **3.28** `_RAW_DIR` in `public.py` updated to `scraper-engine/media/` (old `raw/` subdir was removed in Phase 1.9)
- [x] **3.29** Pipeline conf threshold fix: all 3 pipelines pass `conf_thresh=CONF_THRESH` (0.15) to `infer_video_multi_model`; added zero-detection guard (clip rejected + raw deleted if full inference produces 0 boxes)
- [x] **3.30** Annotated output path: `media/annotated/<model>/<date>/<hash>_annotated.mp4`; temp files written to same dir during inference, renamed on completion; `ClipOut.video_url` extracts relative subpath from `annotated/` segment; `public.py` uses `mp4.relative_to(_ANNOTATED_DIR)` for correct URL construction
- [x] **3.31** Raw file cleanup on reject: all 3 pipelines now delete raw `.mp4` on both reject paths (failed validation + zero-detection inference)
- [x] **3.33** GENERAL pipeline: `_run_general()` added to `annotate_clips` task as 4th step — catch-all for remaining DOWNLOADED clips after specialists; sets `det_class='GENERAL'`; standalone script `scripts/general_pipeline.py` also added
- [x] **3.34** Admin DECLINE: `DELETE /api/admin/clips/{clip_id}` endpoint; AdminPanel DECLINE button (REVIEW clips only); clip preview modal (video for ANNOTATED, external URL link for REVIEW)
- [x] **3.35** `FootageCard.vue` hover-play: `ref="videoEl"` + `pointer-events: none` on `.card-overlay` (was blocking mouseenter); `@click.stop` on video now also emits `open`
- [x] **3.36** `GET /api/stats` `images_labeled` fixed: uses GENERAL image count only (175,627) instead of summing all 4 models (was 354,005)
- [x] **3.37** `HeroSection.vue` shows blank when no GENERAL annotated clip exists (no fallback to placeholder)

#### Backend ↔ ML Engine Integration

- [x] **3.21** `POST /api/admin/train` → create `TrainingRun(QUEUED)` in DB → dispatch Celery task `train_baseline` with `model_type` + `run_id` on gpu queue; task updates status/metrics/weights in DB on finish
- [x] **3.22a** Pipeline reorganization: `annotate_clips` task (sequential AIRCRAFT→VEHICLE→PERSONNEL, Beat daily 04:00 UTC) replaces old GDINO chain. `_filter.py` moved to `scraper-engine/utils/`. Old GDINO tasks quarantined to `ml-engine/tasks/legacy/`. Kaggle download removed from Celery — CLI-only via `scripts/download_new_datasets.py`. Scrape Beat changed to daily 00:00 UTC.
- [x] **3.22b** Fine-tune auto-trigger: `trigger_finetune_check` chord callback in `package_dataset.py` — counts scraped train images per model in GCS merged dirs (AIRCRAFT 1000, VEHICLE 1000, PERSONNEL 500, GENERAL 2500), queues `train_finetune` Celery task; raw file deletion happens after DB commit to prevent orphaned clips
- [x] **3.23** `TickerBar.vue` items pulled from DB: total clip count, scrape status, model mAP50 scores — `GET /api/stats` endpoint returning live counts

#### Training Progress (WebSocket)

- [x] **3.14** FastAPI WebSocket endpoint `ws://localhost:8000/ws/training/{run_id}` — polls DB every 3s, sends `{status, metrics}` JSON, closes on DONE/ERROR; Vite proxy `/ws` with `ws:true`
- [x] **3.14b** `AdminPanel.vue` — WebSocket progress bar: INITIALIZING→EPOCH 0/N (0%)→EPOCH N/N (100%); auto-reconnects to RUNNING jobs on page load; `on_train_epoch_start` callback writes epoch progress at start of each epoch (null metrics); `on_fit_epoch_end` writes final metrics

- [x] **3.32** Test suite implemented across all 4 services:
  - `scraper-engine/tests/unit/` — `_filter.py` unit tests (21 tests: equipment scoring, negative gate, POV detection)
  - `ml-engine/tests/unit/` — epoch callback DB writes, fine-tune trigger logic, GENERAL pipeline (10 tests)
  - `web-app/backend/tests/unit/` + `tests/integration/` — public + admin API endpoints via FastAPI TestClient; decline endpoint (401/404/409)
  - `web-app/frontend/tests/unit/` — Vitest + `@vue/test-utils` component tests (HeroSection, MLCard, TickerBar)
  - pytest marks: `unit`, `integration`, `network`, `gpu`, `smoke`, `slow`; default run skips gpu/slow/network/integration
- [x] **3.38** Agent slash commands wired: `/review-webapp`, `/review-ml`, `/review-scraper`, `/qa-webapp`, `/qa-pipeline`, `/qa-scraper`, `/research-webapp`, `/research-ml`, `/research-scraper` — all in `.claude/commands/`; modus operandi documented in `CLAUDE.md`
- [x] **3.15** Integration smoke test — scraped 20 clips (72h window, Funker530 + GeoConfirmed), 19 downloaded, 17 annotated across VEHICLE/PERSONNEL/GENERAL pipelines; 25 ANNOTATED total in DB; clips appear in archive with playable video

---

### Phase 4 — Cloud & DevOps

#### 4a — Docker Local Stack ✅
- [x] **4.1** `docker-compose.yml` restructured — local dev only (no ml-worker); scraper/backend/frontend + postgres/redis; bind-mounts for scraper + ml media
- [x] **4.1b** Write production Dockerfiles for all services (scraper, ml-engine, backend, frontend); `entrypoint.sh` downloads GDINO + YOLO base weights on cold start; `setup_datasets.sh` for fresh-machine Kaggle download
- [x] **4.1c** `core/storage.py` — GCS upload stub (replaces dead stub); `STORAGE_MODE=remote` flag
- [x] **4.1d** Fine-tune pipeline: `_maybe_trigger_finetune` triggers all 4 models; `YOLO_FINETUNE_MAX_CYCLES=4`; each cycle loads best existing weights (cumulative: 10→20→30→40→50 epochs)
- [x] **4.1e** Inference box labels: `infer_video_multi_model` uses `model.names[cls_id]` per box (was hardcoded model name)
- [x] **4.1f** Docker pre-flight fixes: `JWT_SECRET` alignment, `init_db()` on startup, GDINO config path via installed package, Playwright removed from scraper-engine
- [x] **4.2** All 5 Docker services healthy in Docker Desktop (scraper-worker, scraper-beat, backend, frontend, postgres, redis); scraper-beat scheduling daily scrapes via Celery Beat

#### 4b — Pipeline Hardening ✅
- [x] **4.3** Container path resolution: `_resolve_path()` in all 4 pipeline scripts + `annotate_clips.py`; maps `/app/scraper-engine/media/` → Windows host path via `REPO_ROOT / rel`
- [x] **4.4** Annotated output date fix: all pipelines use `clip.published_at` for folder date (not annotation date)
- [x] **4.5** NMS overdraw fix: `iou=0.45` in all `model()` calls in `inference.py` (was defaulting to 0.7)
- [x] **4.6** Content filter updates: "police" → civilian negative keyword in `_filter.py`
- [x] **4.7** ArchiveSection limit: main page archive capped at 10 most-recent clips (full `/archive` unaffected)
- [x] **4.8** May 12–15 range scrape + annotation: 38 clips scraped, 35 downloaded, all annotated; 58 ANNOTATED total in DB
- [x] **4.9** Video compression + faststart: FFmpeg CRF 28 + `-movflags +faststart` in `inference.py`; all 67 existing annotated files re-encoded (~5× size reduction)
- [x] **4.10** Full-screen box filter: boxes covering >90% of frame area discarded in `infer_video_multi_model`
- [x] **4.11** `_latest_weights` finetune preference: all 4 pipeline scripts now check `runs/finetune/` before `runs/baseline/`; uses highest-numbered run dir with `best.pt`
- [x] **4.12** `_filter.py` updates: cruise missile → AIRCRAFT scoring; hovercraft/naval drone/aircraft carrier → NAVAL_MARINE category; "vehicle"/"vehicles" added to logistics keywords
- [x] **4.12b** `docker-compose.yml` REDIS_URL fix: scraper tasks called `redis.from_url(settings.REDIS_URL)` directly; docker-compose was overriding `CELERY_BROKER_URL` but not `REDIS_URL`; both scraper services now set `REDIS_URL: redis://redis:6379/0`
- [x] **4.12c** Backend stats query fix: two separate `await db.execute()` calls in `get_stats` caused idle-in-transaction pool exhaustion → 504; merged into single query with FILTER label
- [x] **4.12d** Frontend: hero section canvas (detection box overlay) suppressed when video is present; ML card expansion animation slowed 50% (1.3s → 1.95s/2.1s)

#### 4c — Training Cycle ✅/🔄
- [x] **4.13a** AIRCRAFT finetune cycle 1: mAP50=0.968 (run 68, 10 epochs from run 13, best @ epoch 8) ✅
- [x] **4.13b** PERSONNEL baseline cleanup: bad runs 57, 58, 59, 69, 70 deleted from DB + disk; run 29 (mAP50=0.780) intact as reference
- [x] **4.13c** `build_specialist_datasets.py` specialist label filter bug fix: label text now filtered to target class only before writing; PERSONNEL dataset verified clean (train: {2: 22244}, val: {2: 2160})
- [x] **4.13d** All 4 merged datasets rebuilt with fix (2026-05-17): AIRCRAFT (65,557/9,382), VEHICLE (56,440/6,638), GENERAL (~144K/~20K), PERSONNEL (10,962/1,302)
- [x] **4.13e** VEHICLE finetune cycle 1 — mAP50=0.901 (run 73, 10 epochs from run 25, clean merged dataset, 56,440 train) ✅
- [x] **4.13f** PERSONNEL finetune cycle 1 — mAP50=0.872 (run 74, 20 epochs from run 29, clean merged dataset, 10,962 train) ✅
- [x] **4.13g** `annotate_clips.py` raw file deletion bug fix: success path was missing `raw_path.unlink()` (rejection paths had it; acceptance path did not) ✅
- [x] **4.13h-i** `train_baseline.py` hardened: removed `--weights` option entirely; baseline always cold-starts from `yolov8m.pt`; `train_finetune.py __main__` gets `--epochs` arg
- [x] **4.13j** VEHICLE finetune cycle 2 — mAP50=0.904 (run 76, 10 epochs from run 73, 56,440 train) ✅
- [x] **4.13k** PERSONNEL finetune cycle 2 — mAP50=0.873 (run 75, 10 epochs from run 74, 10,962 train) ✅
- [x] **4.13l** GENERAL finetune cycle 1 (scraped) — run 79 DONE mAP50=0.851 ✅
- [x] **4.13m** VEHICLE finetune cycle 2 (scraped) — run 78 DONE mAP50=0.902 ✅
- [x] **4.13m** Scraped pipeline end-to-end (2026-05-21): 10 clips auto-labeled → 5 datasets PACKAGED → 6 clips annotated (3 VEHICLE, 2 PERSONNEL, 1 GENERAL); 80 ANNOTATED total in DB ✅
- [x] **4.13n** Pipeline cleanup fixes (2026-05-21): cv2 corrupt-frame skip in `auto_label.py`; clip dataset dirs deleted immediately after `_merge_datasets()` (not post-training); `merged_dir` cleanup moved to `finally` block; `_cleanup_zero_score_clips()` added to `annotate_clips` end-of-run sweep ✅

#### 4d — Cloud Deployment Architecture ✅
- [x] **4.14** Deployment architecture decided: GCP e2-micro free tier (CPU, $0/mo) + GCP T4 Spot VM (GPU, ~$10/mo via Instance Scheduling 3hr/night)
- [x] **4.15** Cloud deploy agent files: `agents/cloud-deploy/{research,review,qa}.md` + `.claude/commands/{research,review,qa}-deploy.md`; commands wired in `CLAUDE.md`
- [x] **4.16** `docker-compose.prod.yml` created — GCP prod config (no ml-worker/ml-beat; named volumes; nginx direct media serving)
- [x] **4.17** GCP project setup + e2-micro instance provisioned (us-central1, billing upgraded, static IP reserved)
- [x] **4.18** All 6 CPU services deployed to GCP e2-micro (postgres, redis, scraper-worker, scraper-beat, backend, frontend); DB seeded from local dump; 80 annotated clips seeded to ml_media volume; nginx direct media serving
- [x] **4.19** GCP T4 Spot VM fully operational — n1-standard-1 + T4, Instance Scheduling (02:00–05:00 UTC daily), startup script fully automated: NVIDIA drivers (first-boot reboot), ffmpeg, sparse repo clone, venv + deps (torch==2.5.1+cu121 pinned — 2.6+ broke ultralytics weights_only), weights downloaded from GCS, Celery GPU worker + Beat (`--beat` flag). GCS annotation pipeline confirmed end-to-end: scraper uploads raw → T4 downloads + annotates + uploads annotated → `clip.mp4_path = https://storage.googleapis.com/...`. 7 stale pre-GCS clips marked ERROR in DB (Docker paths no longer resolvable). ✅
- [x] **4.20** HTTPS — DuckDNS (`ukrarchive.duckdns.org`) + Let's Encrypt via Certbot standalone; nginx.conf redirects HTTP→HTTPS; cert auto-renews via certbot systemd timer; `/etc/letsencrypt` mounted into frontend container
- [x] **4.21** CI/CD — GitHub Actions workflows:
  - [x] **4.21a** `ci.yml` — frontend build (`npm run build`) + ruff lint on push/PR to main
  - [x] **4.21b** `deploy-e2-micro.yml` — SSH deploy via `appleboy/ssh-action` on push to main after CI passes (`workflow_run` trigger)
  - [x] **4.21c** `deploy-weights.yml` — DELETED; weights now auto-uploaded to GCS at end of each `train_finetune` run (no manual step needed)
  - [x] **4.21d** GitHub secrets set: `E2_MICRO_HOST`, `E2_MICRO_SSH_KEY`, `T4_SSH_KEY`; e2-micro uses sparse checkout; CI/CD pipeline confirmed passing end-to-end (2026-05-26)
- [x] **4.22** Admin panel `latestRun()` fix: prefer DONE runs over ERROR for model status cards — was showing ERROR for all models despite successful DONE runs
- [x] **4.23** Test suite hardening: production DB guard added to all 3 conftests (`web-app/backend`, `scraper-engine`, `ml-engine`) — refuses to run if `DATABASE_URL` points to non-local host
- [x] **4.24** Security: `.claude/settings.json` purged from all 140 git commits via `git filter-repo --invert-paths`; added to `.gitignore`; force-pushed to GitHub
- [x] **4.25** `web-app/backend/config.py` hardened: removed insecure defaults for `JWT_SECRET`, `ADMIN_USERNAME`, `ADMIN_PASSWORD` — app fails fast at startup if not set in `.env`
- [x] **4.26** `web-app/backend/api/public.py` `_model_stats` performance fix: replaced 8 sequential DB queries (2 per model × 4 models) with a single query fetching all DONE/RUNNING TrainingRuns at once; reduces API response time from 5–10s to <1s ✅
- [x] **4.27** Mobile-responsive frontend: all pages and components made responsive across 3 breakpoints (≤900px tablet, ≤768px mobile, ≤480px small phone) — nav links hidden on mobile; hero stats pulled from absolute positioning into flow; archive sidebar collapses to horizontal scroll strip; AdminPanel tables wrapped in `overflow-x: auto` scroll containers; ML cards flatten trapezoid/parallelogram shapes at narrow widths; modal, data strip, footer all adapt ✅

---

## 5. Docker Desktop Quick-Start

### First run (local dev, you already have the data)

```bash
# 1. Create .env — only 3 values required
cp .env.example .env
# Edit .env: set POSTGRES_PASSWORD, JWT_SECRET, ADMIN_PASSWORD

# 2. Build and start all 7 services
#    docker-compose.override.yml is auto-merged — bind-mounts your existing
#    ml-engine/media/, ml-engine/runs/, scraper-engine/media/ into containers
docker compose up --build
```

- Frontend: http://localhost
- Backend API: http://localhost:8000
- The `ml_checkpoints` volume is populated automatically by `entrypoint.sh`
  (downloads GDINO .pth + yolov8m.pt on first start, ~750 MB)

### Fresh machine (no local datasets)

```bash
cp .env.example .env
# Also set KAGGLE_USERNAME and KAGGLE_KEY in .env
docker compose up --build -d postgres redis
docker compose run --rm ml-worker bash scripts/setup_datasets.sh   # ~10 GB, 30-60 min
docker compose up
```

### Running the pipeline manually

```bash
# Trigger a scrape now (instead of waiting for 00:00 UTC Beat schedule)
docker compose exec scraper-worker celery -A celery_app call tasks.scrape_funker530.scrape_funker530

# Trigger YOLO annotation now
docker compose exec ml-worker celery -A celery_app call tasks.annotate_clips.annotate_clips
```

---

### Phase 5 — ML Showcase (models as the product)

> **North star:** The trained models are the primary deliverable. The archive is proof they work.
> Every Phase 5 feature either surfaces the models directly or proves their value.

**Page layout (final section order):**
```
Hero → TickerBar → DataStrip → MLDetectionSection → AnalyticsSection → ArchiveSection → CapabilitiesSection → Footer
```

**Section restructuring:**
- `AboutSection` removed — right-area content (links, tech stack, project info) moves into Footer
- `MissionSection` removed
- `Footer` rewritten: project name + tagline + submit button (left); about right-area content (right); empty link columns dropped
- `CapabilitiesSection` upgraded in-place: gains pipeline diagram (5.2) + model download cards (5.3)
- `AnalyticsSection` is a new component (5.4) between ArchiveSection and CapabilitiesSection
- `DataStrip` moved up: after TickerBar, before MLDetectionSection

#### 5.1 — Footer & About Restructure ✅
- [x] **5.1a** Rewrite `SiteFooter.vue` — 3 cols: Site links, About project stats, Models placeholder; same footer grid design
- [x] **5.1b** Remove `AboutSection` and `MissionSection` from `PublicFeed.vue`
- [x] **5.1c** Update `PublicFeed.vue` section order: Hero → TickerBar → DataStrip → MLDetection → Archive → Capabilities → Footer
- [x] **5.1d** Nav links: removed `about`, order = `detection → archive → capabilities`
- [x] **5.1e** Hero reframed: tag = "Open Military Asset Detection Models", CTA = "Explore Models", stats = per-model mAP50
- [x] **5.1f** FootageModal shows real clip description; `/api/annotated-clips` returns `description` field

#### 5.2 — Pipeline Explainer (in CapabilitiesSection) ✅
- [x] **5.2a** Animated SVG chevron pipeline diagram above cap-grid — chevron arrow shapes, animated grey dashes at junctions, traveling amber dot, live stats from `/api/stats`
- [x] **5.2b** Mobile: vertical list layout ≤640px (SVG hidden)

#### 5.3 — Model Hub ✅
- [x] **5.3a** GCS weights already public-read (bucket `allUsers` objectViewer IAM)
- [x] **5.3b** `GET /api/models` — all DONE runs per model with `is_best` flag + GCS download URL
- [x] **5.3c** Model download cards in `CapabilitiesSection.vue` — 4-col responsive grid (2-col ≤900px, 1-col ≤480px); badge colors from `--cat-color-*` vars
- [x] **5.3d** `/api-docs` page — endpoint reference, curl examples, model spec
- [x] **5.3e** `/models` page — version history table grouped by model (all runs, date/mAP50/download, best highlighted)
- [x] **5.3f** AppNav updated: Models + API route links (desktop + mobile menu); scroll-spy disabled on non-home routes
- [x] **5.3g** SiteFooter Models column links to `/models` and `/api-docs`
- [x] **5.3h** `/api/annotated-clips` returns `description` field; FootageModal shows real clip description

#### 5.4 — Analytics & Detection Index (new AnalyticsSection) ✅
- [x] **5.4a** `GET /api/stats/charts` — clips/day, detection class split, mAP50 timeline, training scatter ✅
- [x] **5.4b** `GET /api/training/epoch-data` — per-epoch metrics + CM + curves per run; dev CSV fallback ✅
- [x] **5.4c** `AnalyticsSection.vue` — Chart.js: bar (clips/day), doughnut (class split), scatter+line (mAP50), radar (model performance); per-run drill-down with 4 epoch charts + CM heatmap + 3 curve charts ✅
- [x] **5.4d** Admin panel pipeline stats redesign — 3-col layout: scraper stats, inference progress bars, packaged datasets list ✅
- [x] **5.4e** DB backfill for all runs (13/25/29/30/68/73/74/75/76/77/78/79) — epochs + CM + curves ✅

---

## 6. Next Steps

Phase 0 ✅, Phase 1 ✅, Phase 2 ✅, Phase 3 ✅, Phase 4 ✅, Phase 5 ✅

**Training status (2026-06-04):**
- AIRCRAFT: 0.929 (baseline run 13) → 0.968 (Kaggle finetune run 68) → 0.964 (scraped finetune run 77) ✅ — run 68 still best weights
- VEHICLE: 0.871 (baseline run 25) → 0.904 (Kaggle finetune run 76) → 0.902 (scraped finetune run 78) ✅
- PERSONNEL: 0.780 (baseline run 29) → 0.873 (Kaggle finetune run 75, cycle 2) ✅
- GENERAL: 0.784 (baseline run 30) → 0.851 (scraped finetune run 79) ✅
- 72 ANNOTATED clips in DB; all 12 training runs backfilled with epochs + CM + curves

**Web app — complete ✅:**
- Public feed, archive, submit, hero, ticker, ML cards — all wired to live DB/API
- Admin panel: clips table (APPROVE + DECLINE + preview modal), training runs table, train buttons, live WebSocket progress bar
- Video pipeline: FFmpeg CRF 28 + faststart; 90% full-screen box filter; multi-model inference
- Mobile-responsive across all pages (≤900px/768px/480px breakpoints)
- Gzip compression enabled on nginx (94KB → 38KB main JS bundle)

**Cloud deployment — complete ✅ (2026-05-31):**
- Architecture: GCP e2-micro free tier (CPU, $0/mo) + 2× T4 Spot VM (GPU, ~$20/mo)
- e2-micro live ✅ — all 6 CPU services deployed; HTTPS via ukrarchive.duckdns.org + Let's Encrypt
- inference-engine VM ✅ — n1-standard-1 + T4 Spot; Instance Schedule 03:00–04:00 UTC; Q=pipeline: auto_label_batch (GDINO @03:05) + annotate_clips (YOLO @03:35) + package_dataset + prepare_finetune_batch; merged dirs backed up to GCS after every append
- training-engine VM ✅ — n1-standard-4 + T4 Spot; Instance Schedule 04:30 UTC start; startup script checks DB for QUEUED TrainingRuns → shuts down immediately if none; Q=training: train_finetune × 4 models; downloads merged from GCS, trains, uploads weights, self-shuts
- CI/CD live ✅ — GitHub Actions: frontend build + ruff lint → auto-deploy to e2-micro on push to main; deploy-inference-engine manual workflow
- 1-T4 quota constraint enforced: inference stops at 04:00, training starts at 04:30 (30-min buffer)

**Bug fixes applied (2026-06-01):**
- [x] `train_finetune`: `acks_late=True + reject_on_worker_lost=True` — tasks survive VM preemption/SIGKILL; no task loss on spot preemption
- [x] `train_finetune`: shutdown now checks remaining QUEUED/RUNNING runs — was shutting down after first model completed
- [x] `train_finetune`: `_delete_gcs_merged()` cleans up `merged/<MODEL>/` from GCS after training (was accumulating indefinitely)
- [x] `train_finetune`: `total_train_images` saved to run metrics (fixes zero on training stats page)
- [x] `infra/gcp/main.tf`: `--pool=solo` added to celery-training ExecStart (YOLO spawns child processes; billiard daemon restriction)
- [x] `web-app/backend/api/public.py`: `/api/stats` shows latest completed run mAP, not historical best
- [x] `web-app/frontend`: video autoplay race condition fixed — `@canplay` handler in FootageModal; `.catch()` on FootageCard hover-play
- [x] `web-app/frontend`: nav logo mark replaced with favicon SVG across AppNav, AdminPanel, AdminLogin

**Bug fixes applied (2026-05-29 → 2026-05-31):**
- [x] `package_dataset`: merged dirs backed up to GCS after every append (survives VM recreation)
- [x] `package_dataset`: PostgreSQL sequence synced before TrainingRun insert (prevents UniqueViolation crash)
- [x] `annotate_clips`: GCS 404 caught per-clip (missing raw → ERROR, continues instead of crashing worker)
- [x] `scrape_funker530`: ffprobe fallback for duration when yt-dlp returns 0
- [x] nginx: gzip compression enabled (94KB → 38KB JS)
- [x] AboutSection: "Self-hosted" → "Google Cloud Platform"
- [x] Terraform fixed: `inference_engine` scheduling block — removed `preemptible = true` (incompatible with Instance Scheduling resource policies); `training_engine` — added `provisioning_model = "SPOT"` + `instance_termination_action = "STOP"`
- [x] `start_workers.sh` updated: removed stale "Phase 4 Docker" comment; added local runner hint

**Phase 4 additions (2026-05-29):**
- [x] `_model_stats` performance fix: 8 sequential DB queries → 1 query; API response time 5–10s → <1s
- [x] Mobile-responsive frontend: 3-breakpoint CSS (≤900px/768px/480px); archive sidebar collapses to horizontal scroll; admin tables get horizontal scroll containers; hero stats pulled from absolute to flow; all pages verified

---

*This document is the single source of truth. Update it as phases complete or decisions change.*
