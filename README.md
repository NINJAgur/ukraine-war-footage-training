# Ukraine Combat Footage Archival System

[![CI](https://github.com/NINJAgur/ukraine-war-footage-training/actions/workflows/ci.yml/badge.svg)](https://github.com/NINJAgur/ukraine-war-footage-training/actions/workflows/ci.yml)
[![Deploy e2-micro](https://github.com/NINJAgur/ukraine-war-footage-training/actions/workflows/deploy-e2-micro.yml/badge.svg)](https://github.com/NINJAgur/ukraine-war-footage-training/actions/workflows/deploy-e2-micro.yml)
[![Deploy inference-engine](https://github.com/NINJAgur/ukraine-war-footage-training/actions/workflows/deploy-inference-engine.yml/badge.svg)](https://github.com/NINJAgur/ukraine-war-footage-training/actions/workflows/deploy-inference-engine.yml)
[![Deploy training-engine](https://github.com/NINJAgur/ukraine-war-footage-training/actions/workflows/deploy-training-engine.yml/badge.svg)](https://github.com/NINJAgur/ukraine-war-footage-training/actions/workflows/deploy-training-engine.yml)

> Built on top of [YOLO Training Template](https://github.com/computer-vision-with-marco/yolo-training-template) by [Marco Franzon](https://github.com/computer-vision-with-marco) — a clean starting point for YOLO training workflows with GroundingDINO auto-labeling.

Automated full-stack application that scrapes, auto-labels, and publicly displays
archival combat footage from the war in Ukraine, with a secure admin panel for
YOLOv8 model retraining.

**Live:** https://ukrarchive.duckdns.org

## Architecture

```
e2-micro (Docker, always-on):
  scraper-beat  ──00:00 UTC──→ scrape_funker530
                ──00:15 UTC──→ scrape_geoconfirmed
                                      ↓
                               Clip(DOWNLOADED, scores in DB) → GCS raw/

inference-engine VM (n1-standard-1 + T4 Spot, Instance Schedule 03:00–04:00 UTC):
  beat @03:05 → auto_label_batch → auto_label_clip × N (GDINO)
                                → package_dataset (Dataset PACKAGED)
                                    → upload merged/<MODEL>/ to GCS after every append
                                → [image threshold met] trigger_finetune_check
                                    → TrainingRun(QUEUED) per model
                                    → prepare_finetune_batch → dispatch train_finetune × N → Q=training
  beat @03:35 → annotate_clips (YOLO → annotated MP4 → GCS annotated/)
               → _shutdown_if_no_training (self-shuts VM if no active runs)

training-engine VM (n1-standard-4 + T4 Spot, Instance Schedule 04:30 UTC start):
  startup: query DB for QUEUED TrainingRuns → shutdown immediately if none
  train_finetune × N models → download merged/ from GCS → YOLO → upload best.pt
  → self-shutdown after last model

1-T4 quota: inference stops at 04:00, training starts at 04:30 (30-min buffer)
```

**Scraper → ML decoupling:** scrapers write keyword scores to DB (`score_aircraft`, `score_vehicle`, `score_personnel`, `score_uas`, `is_pov`). Pipelines query by score thresholds — no re-scraping at inference time.

## Services

| Service | Directory | Runs on | Status |
|---------|-----------|---------|--------|
| Scraper Engine | `scraper-engine/` | e2-micro (Docker) | ✅ Complete |
| Inference Engine | `inference-engine/` | inference-engine VM (T4) | ✅ Complete |
| Training Engine | `training-engine/` | training-engine VM (T4) | ✅ Complete |
| Backend API | `web-app/backend/` | e2-micro (Docker) | ✅ Complete |
| Frontend | `web-app/frontend/` | e2-micro (Docker) | ✅ Complete (mobile-responsive) |

## ML Training — Best Runs

| Model | mAP50 | Images | Stage | Run | Status |
|-------|-------|--------|-------|-----|--------|
| AIRCRAFT | 0.968 | 65,553 | Finetune (Kaggle) | 68 | Scraped cycle done (run 77: 0.964) ✅ |
| VEHICLE | 0.904 | 56,440 | Finetune (Kaggle) | 76 | Scraped cycle done (run 78: 0.902) ✅ |
| PERSONNEL | 0.873 | 10,962 | Finetune (Kaggle) | 75 | Scraped cycles continuing (image threshold) ✅ |
| GENERAL | 0.851 | 144,466 | Finetune (scraped) | 79 | All cycles complete ✅ |

#### Dataset Inventory (8 datasets)

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

## Local Running Guide

### Prerequisites

1. **PostgreSQL** running (local install or `docker compose up -d postgres`)
2. **Redis** running (local install or `docker compose up -d redis`)
3. **Python venv** set up for each engine you need:
   ```bash
   cd inference-engine && python -m venv venv && venv/Scripts/pip install -r requirements.txt
   cd scraper-engine   && python -m venv venv && venv/Scripts/pip install -r requirements.txt
   ```
4. **`.env`** files in each engine directory (copy `.env.example` if present; at minimum set `DATABASE_SYNC_URL` and `CELERY_BROKER_URL`)

### Start all workers at once

```bash
bash start_workers.sh
```

This starts: scraper-worker (Q=default), scraper-beat, inference-engine worker (Q=pipeline, --pool=solo), inference-engine beat.

### Start workers individually

```bash
# Scraper workers (needed for 'scrape' command)
cd scraper-engine
celery -A celery_app worker -Q default --loglevel=info --concurrency=4
celery -A celery_app beat --loglevel=info   # separate terminal

# Inference-engine worker (needed for 'gdino' and 'annotate' commands)
cd inference-engine
celery -A celery_app worker -Q pipeline --pool=solo --concurrency=1 --loglevel=info
```

### Run pipeline steps with `run_local.py`

```bash
# Show DB state (clips, datasets, recent training runs)
python run_local.py status

# Trigger scrapers (dispatches to scraper-worker, polls for new DOWNLOADED clips)
python run_local.py scrape                   # both sources
python run_local.py scrape funker530
python run_local.py scrape geoconfirmed

# Run GDINO auto-labeling batch (dispatches to inference-engine, polls for PACKAGED datasets)
python run_local.py gdino

# Run YOLO annotation batch (dispatches to inference-engine, polls for ANNOTATED clips)
python run_local.py annotate

# Full pipeline: scrape → gdino → annotate
python run_local.py all
python run_local.py all funker530            # scrape one source then full pipeline
```

`run_local.py` connects to Redis at `CELERY_BROKER_URL` (default: `redis://localhost:6379/0`) and polls PostgreSQL via `DATABASE_SYNC_URL`.

### Web app (local dev)

```bash
# Backend (port 8000)
cd web-app/backend && python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Frontend (port 5173, auto-opens browser)
cd web-app/frontend && npm run dev
```

Admin panel: `http://localhost:5173/admin` — credentials in `web-app/backend/.env`

## Running Tests

Unit tests run locally without Docker. Integration tests require `docker compose up -d` first.

```bash
# Scraper unit tests (no DB needed)
cd scraper-engine && pytest tests/unit -v

# Inference-engine unit tests (no GPU needed)
cd inference-engine && pytest tests/unit -v

# Inference-engine integration tests (requires Docker Compose running)
cd inference-engine && pytest tests/integration -v

# Training-engine unit tests (no GPU needed)
cd training-engine && pytest tests/unit -v

# Backend unit tests (no DB needed)
cd web-app/backend && pytest tests/unit -v -m unit

# Frontend component tests
cd web-app/frontend && npm run test
```

## Agent Slash Commands

Domain-specific review, QA, and research agents in `.claude/commands/`:

| Command | Use when |
|---------|----------|
| `/review-webapp` | After web-app changes touching auth, endpoints, or architecture |
| `/review-ml` | After inference-engine or training-engine pipeline changes |
| `/review-scraper` | After scraper changes |
| `/qa-webapp` | New API endpoint added or DB enum changed |
| `/qa-pipeline` | End-to-end DB state health check |
| `/research-webapp` | Before implementing new web-app patterns |
| `/research-ml` | Before new ML pipeline patterns |

## Tech Stack

- **Scraping:** Funker530 REST API + GeoConfirmed REST API + yt-dlp + ffprobe (duration)
- **Queue:** Celery 5 + Redis (broker + result backend)
- **Database:** PostgreSQL 16 (`ukraine_footage`) + SQLAlchemy 2.x
- **ML:** Ultralytics YOLOv8m + GroundingDINO (SwinT) + PyTorch 2.5.1 (CUDA 12.1)
- **Storage:** Google Cloud Storage — raw clips, annotated videos, merged training datasets, model weights
- **Backend:** FastAPI (async) + Pydantic v2 + WebSocket (live training progress)
- **Frontend:** Vue 3 (`<script setup>`) + Vite + Tailwind CSS + Pinia + Vue Router 4; nginx with gzip
- **Infra:** GCP e2-micro (free tier) + 2× n1-standard T4 Spot VMs; GCP Instance Schedules; Terraform; GitHub Actions CI/CD; Let's Encrypt HTTPS

See [PROJECT_PLAN.md](PROJECT_PLAN.md) for full architecture and implementation plan.
