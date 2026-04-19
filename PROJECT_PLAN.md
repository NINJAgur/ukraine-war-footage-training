# PROJECT_PLAN.md — Ukraine Combat Footage Web Application
> **Source of Truth** — All phases, structure, and decisions are tracked here.
> Last updated: 2026-04-18

---

## Table of Contents
1. [Architecture Overview](#1-architecture-overview)
2. [Host Machine Setup Guide](#2-host-machine-setup-guide)
3. [Directory Structure](#3-directory-structure)
4. [Master To-Do List](#4-master-to-do-list)
5. [Next Steps](#5-next-steps)

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
│                         INGESTION LAYER                             │
│                                                                     │
│  [Celery Beat]                                                      │
│       │                                                             │
│       ├──► scrape_funker530 task  (REST API + yt-dlp download)      │
│       ├──► scrape_geoconfirmed task (GeoConfirmed REST API + yt-dlp) │
│       └──► download_kaggle task   (Kaggle API)                      │
│                    │                                                │
│                    ▼                                                │
│          raw video/frames saved to /media/raw/                      │
│          Clip record written to PostgreSQL (status=PENDING)         │
└─────────────────────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          ML LAYER                                   │
│                                                                     │
│  [auto_label task]  ──────────────────────────────────────────────► │
│  GroundingDINO zero-shot inference on extracted frames              │
│  Outputs: bounding-box .txt files (YOLO format)                     │
│                     │                                               │
│       ┌─────────────┴──────────────┐                               │
│       ▼                            ▼                               │
│  [package_dataset task]    [render_annotated task]                 │
│  Build YOLO dir structure   Run inference.py on raw video          │
│  + data.yaml                Outputs annotated H.264 MP4            │
│       │                            │                               │
│       ▼                            ▼                               │
│  Dataset record in DB        Clip record updated                   │
│  (status=LABELED)            (status=ANNOTATED, mp4_path set)      │
└─────────────────────────────────────────────────────────────────────┘
                     │                          │
                     │                          ▼
                     │               ┌──────────────────────┐
                     │               │   PUBLIC DASHBOARD   │
                     │               │  "Daily Feed" card   │
                     │               │  visible to users    │
                     │               └──────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       ADMIN TRAINING LAYER                          │
│                                                                     │
│  Admin sees: "5 New Auto-Labeled Datasets" badge in inbox          │
│  Admin selects datasets → clicks "Train Model"                      │
│                     │                                               │
│          ┌──────────┴──────────┐                                   │
│          ▼                     ▼                                   │
│  [train_baseline task]  [train_finetune task]                      │
│  Stage 1: Kaggle data   Stage 2: custom labeled data               │
│  sudipchakrabarty/      load baseline.pt as starting weights       │
│  kiit-mita + others     train on auto-labeled custom datasets      │
│  → baseline.pt          → fine_tuned.pt                            │
│                                                                     │
│  TrainingRun record logged to DB; WebSocket pushes                 │
│  live epoch/loss metrics to Admin → TrainModel.vue                 │
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
| **Containers** | Docker + Docker Compose w/ NVIDIA runtime **(Phase 4 only)** |
| **DevOps** | GitHub Actions + GCP (GCS, Compute Engine) |

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
├── docker-compose.yml                   ← orchestrates all services
│
├── .claude/                             ← Claude Code agentic workspace
│   └── settings.json                    ← permissions, hooks, MCP config
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
│   └── web-app/
│       ├── research.md
│       ├── qa.md
│       └── review.md
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
│   │   ├── _filter.py                   ← shared content filter (equipment + impact/aftermath gate)
│   │   ├── scrape_funker530.py          ← Funker530 REST API + filter + yt-dlp
│   │   ├── scrape_geoconfirmed.py       ← GeoConfirmed REST API + parallel fetch + filter + yt-dlp
│   │   └── download_kaggle.py
│   ├── db/
│   │   ├── session.py
│   │   └── models.py
│   └── tests/
│       └── test_scrape_live.py          ← Phase 1 end-to-end test
│
├── ml-engine/                           ← PHASE 2: ML Pipeline 🔄 Next
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── celery_app.py
│   ├── config.py
│   ├── runs/                            ← YOLO training output (gitignored)
│   ├── media/                           ← frames, annotated, datasets (gitignored)
│   ├── tasks/                           ← Phase 2 (not yet implemented)
│   │   ├── auto_label.py
│   │   ├── package_dataset.py
│   │   ├── render_annotated.py
│   │   ├── train_baseline.py
│   │   └── train_finetune.py
│   ├── tests/
│   ├── core/                            ← migrated from original repo
│   │   ├── main.py                      ← from scripts/main.py
│   │   ├── inference.py                 ← from scripts/inference.py
│   │   ├── preprocessing.py             ← from scripts/preprocessing.py
│   │   ├── dataset_explorer.py          ← from scripts/dataset_explorer.py
│   │   └── autolabeling/
│   │       └── auto_label.py            ← from autolabeling/auto-label.py
│   └── db/
│       ├── session.py
│       └── models.py
│
├── web-app/                             ← PHASE 3: Web Application ⏳ Pending
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
│           │       ├── AdminInbox.vue
│           │       └── TrainModel.vue
│           └── components/
│               ├── VideoCard.vue
│               ├── DatasetRow.vue
│               └── TrainingProgress.vue
│
├── infra/                               ← PHASE 4: Cloud & DevOps ⏳ Pending
│   ├── gcp/
│   │   ├── main.tf
│   │   └── variables.tf
│   └── nginx/
│       └── nginx.conf
│
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── deploy.yml
│
```

> Tests live inside each service directory (`scraper-engine/tests/`, `ml-engine/tests/`, `web-app/tests/`) — not at the repo root.

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
- [x] **1.6** Implement `tasks/download_kaggle.py` (Kaggle API)
- [x] **1.7** Configure `beat_schedule.py` (hourly scrape, nightly Kaggle)
- [x] **1.8** Write `scraper-engine/Dockerfile`
- [x] **1.9** Implement `tasks/_filter.py` — shared content filter: equipment keyword gate (regex + word boundaries, specific hardware names) + impact/aftermath rejection gate (fire, smoke, ruins, wreckage — blocks aftermath footage, not target type)
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
- [x] **2.13** Add `ModelType` enum (GENERAL, SOLDIER, VEHICLE, AIRCRAFT) to `ml-engine/db/models.py`; add `model_type` column to `TrainingRun`
- [x] **2.14** Update `ml-engine/config.py` — `MODEL_CLASSES`, `GDINO_CLASS_TO_MODEL`, `MODEL_COLORS` dicts; clean 8-class GDINO prompt
- [x] **2.15** Update `ml-engine/tasks/train_baseline.py` — per-model Kaggle dataset config; auto-detects nc/names from dataset YAML when not supplied
- [x] **2.16** Update `ml-engine/tasks/train_finetune.py` — per-model class filtering + ID remapping when building merged dataset
- [x] **2.17** Add `infer_video_multi_model()` to `ml-engine/core/inference.py` — sequential multi-model rendering with per-type colour-coded bboxes
- [x] **2.18** Update `ml-engine/tasks/render_annotated.py` — `_best_weights_per_model()` + 4-model rendering via `infer_video_multi_model()`
- [x] **2.19** Verified Kaggle baseline datasets:
  - GENERAL/SOLDIER/VEHICLE: `sudipchakrabarty/kiit-mita` — 1700 imgs, nc=7 (Artilary, Missile, Radar, M. Rocket Launcher, Soldier, Tank, Vehicle) ✅
  - AIRCRAFT: `mihprofi/drone-detect` — 37900 imgs, nc=2 (Dron, Dron2) ✅
  - Dropped: `rookieengg` (9.5GB impractical), `nzigulic` (broken cache), `hillsworld` (pedestrian not soldier)
  - Future: `piterfm/2022-ukraine-russia-war-equipment-losses-oryx` — Oryx Ukraine war equipment photos → run GroundingDINO auto-labeling → fine-tuning corpus (task 2.21)

#### 2c — Testing ✅
- [x] **2.20** `ml-engine/tests/test_pipeline_e2e.py` — E2E: fixture → auto_label → package_dataset → verify → self-cleaning teardown; `--skip-gdino` flag
- [x] **2.21** `ml-engine/tests/test_baseline_train.py` — smoke test: creates TrainingRun → train_baseline() → verifies best.pt; `--model-type`, `--epochs`, `--keep` flags
- [x] **2.22** DB tables created via `Base.metadata.create_all()` (clips, datasets, training_runs)
- [x] **2.23** Smoke test passed: VEHICLE model, 2 epochs, kiit-mita — mAP50=0.405, best.pt=52MB in ~2 min

---

### Phase 3 — Web Application

- [ ] **3.1** Scaffold `web-app/backend/` + `requirements.txt`
- [ ] **3.2** ORM models + Alembic migration
- [ ] **3.3** Pydantic v2 schemas
- [ ] **3.4** Public API endpoints
- [ ] **3.5** Admin API endpoints + WebSocket
- [ ] **3.6** JWT authentication
- [ ] **3.7** Scaffold Vue 3 frontend
- [ ] **3.8** Dark tactical Tailwind theme
- [ ] **3.9** `PublicFeed.vue`
- [ ] **3.10** `Archive.vue`
- [ ] **3.11** `Submit.vue`
- [ ] **3.12** `AdminLogin.vue`
- [ ] **3.13** `AdminInbox.vue`
- [ ] **3.14** `TrainModel.vue`
- [ ] **3.15** Integration test

---

### Phase 4 — Cloud & DevOps

- [ ] **4.1** Install Docker Desktop + NVIDIA Container Toolkit
- [ ] **4.2** Write production Dockerfiles for all services
- [ ] **4.3** Write production `docker-compose.yml`
- [ ] **4.4** Write `infra/gcp/main.tf`
- [ ] **4.5** Write `infra/nginx/nginx.conf`
- [ ] **4.6** Write `.github/workflows/ci.yml`
- [ ] **4.7** Write `.github/workflows/deploy.yml`
- [ ] **4.8** Configure GCS CORS
- [ ] **4.9** End-to-end smoke test on GCP

---

## 5. Next Steps

Phase 0 ✅, Phase 1 ✅, Phase 2a ✅ (core ML tasks + Dockerfile + Beat polling bridge) are complete.

**Phase 2 complete.** Next: Phase 3 — Web Application (FastAPI backend + Vue 3 frontend).

```bash
# Phase 3 kick-off:
# 3.1  scaffold web-app/backend/ — FastAPI + SQLAlchemy async + Alembic
# 3.2  ORM models (Clip, Dataset, TrainingRun) + first Alembic migration
# 3.3  Pydantic v2 schemas
# 3.4  Public API: GET /api/feed, GET /api/archive, POST /api/submit
# 3.5  Admin API: GET /api/admin/datasets, POST /api/admin/train + JWT auth
# 3.6  WebSocket training progress endpoint
# 3.7  Vue 3 frontend scaffold (Vite + Tailwind dark theme)

# Future Phase 2 additions (non-blocking for Phase 3):
# 2.21  tasks/label_oryx_dataset.py — run GroundingDINO on Oryx Ukraine war photos
#        → fine-tuning corpus (piterfm/2022-ukraine-russia-war-equipment-losses-oryx)
```

---

*This document is the single source of truth. Update it as phases complete or decisions change.*
