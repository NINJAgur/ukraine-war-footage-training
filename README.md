# Ukraine Combat Footage Archival System

[![CI](https://github.com/NINJAgur/ukraine-war-footage-training/actions/workflows/ci.yml/badge.svg)](https://github.com/NINJAgur/ukraine-war-footage-training/actions/workflows/ci.yml)
[![Deploy](https://github.com/NINJAgur/ukraine-war-footage-training/actions/workflows/deploy-e2-micro.yml/badge.svg)](https://github.com/NINJAgur/ukraine-war-footage-training/actions/workflows/deploy-e2-micro.yml)

Automated full-stack application that scrapes, auto-labels, and publicly displays
archival combat footage from the war in Ukraine, with a secure admin panel for
YOLOv8 model retraining.

**Live:** https://ukrarchive.duckdns.org

## Architecture

```
[Celery Beat 00:00 UTC] → scraper-engine → Clip(DOWNLOADED, scores in DB)
                                                    ↓
                          [Celery Beat 04:00 UTC] annotate_clips task
                                  ↓           ↓          ↓         ↓
                             AIRCRAFT    VEHICLE    PERSONNEL   GENERAL
                             pipeline    pipeline   pipeline    pipeline
                                  └──────────┴──────────┴─────────┘
                                             ↓
                                     annotated MP4
                                  ↓                   ↓
                           Public Feed           Admin Panel
                                                      ↓
                                          train_baseline / train_finetune
                                                      ↓
                                             best.pt weights
                                          (auto-selected next run)
```

**Scraper → ML decoupling:** scrapers write keyword scores to DB (`score_aircraft`, `score_vehicle`, `score_personnel`, `score_uas`, `is_pov`). Pipelines query by score thresholds — no re-scraping at inference time.

## Services

| Service | Directory | Phase | Status |
|---------|-----------|-------|--------|
| Scraper Engine | `scraper-engine/` | 1 | ✅ Complete |
| ML Engine | `ml-engine/` | 2 | ✅ Complete |
| Backend API | `web-app/backend/` | 3 | ✅ Complete |
| Frontend | `web-app/frontend/` | 3 | ✅ Complete |
| Cloud & DevOps | GCP e2-micro + T4 Spot | 4 | ✅ Complete |

## ML Training — Best Runs

| Model | mAP50 | Images | Stage | Run |
|-------|-------|--------|-------|-----|
| AIRCRAFT | 0.968 | 64,916 | Finetune | 68 |
| VEHICLE | 0.904 | 56,440 | Finetune | 76 |
| PERSONNEL | 0.873 | 10,962 | Finetune | 75 |
| GENERAL | 0.784 | 144,466 | Baseline | 30 |

## Dataset Inventory (8 Kaggle datasets)

| Dataset | nc | Images | Per-model role |
|---------|-----|--------|----------------|
| `mihprofi/drone-detect` | 2 | 37,900 | AIRCRAFT, GENERAL |
| `shakedlevnat/military-aircraft-database` | 83 | 19,958 | AIRCRAFT, GENERAL |
| `nzigulic/military-equipment` | 11 | 16,809 | AIRCRAFT, VEHICLE, GENERAL |
| `piterfm/oryx-equipment-losses` | 3 | 26,197 | AIRCRAFT, VEHICLE, GENERAL |
| `sudipchakrabarty/kiit-mita` | 7 | 1,700 | VEHICLE, PERSONNEL, GENERAL |
| `rookieengg/military-aircraft-detection` | 43 | 11,788 | AIRCRAFT, GENERAL |
| `rawsi18/military-assets-12-classes` | 12 | 26,315 | AIRCRAFT, VEHICLE, PERSONNEL, GENERAL |
| `rupankarmajumdar/amad-5` | 5 | 34,960 | VEHICLE, PERSONNEL, GENERAL |
| **TOTAL** | | **175,627** | |

## Quick Start

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
# Backend unit tests (no DB needed)
cd web-app/backend && pytest tests/unit -v -m unit

# Backend integration tests (requires Docker Compose running)
cd web-app/backend && pytest tests/integration -v

# Scraper unit tests (no DB needed)
cd scraper-engine && pytest tests/unit -v

# ML engine unit tests (no GPU needed)
cd ml-engine && pytest tests/unit -v

# Frontend component tests
cd web-app/frontend && npm run test
```

## Agent Slash Commands

Domain-specific review, QA, and research agents in `.claude/commands/`:

| Command | Use when |
|---------|----------|
| `/review-webapp` | After web-app changes touching auth, endpoints, or architecture |
| `/review-ml` | After ml-engine pipeline changes |
| `/review-scraper` | After scraper changes |
| `/qa-webapp` | New API endpoint added or DB enum changed |
| `/qa-pipeline` | End-to-end DB state health check |
| `/research-webapp` | Before implementing new web-app patterns |
| `/research-ml` | Before new ML pipeline patterns |

## Tech Stack

- **Scraping:** Funker530 REST API + GeoConfirmed REST API + yt-dlp
- **Queue:** Celery + Redis
- **Database:** PostgreSQL 16 (`ukraine_footage`)
- **ML:** Ultralytics YOLOv8 + GroundingDINO + PyTorch (CUDA 12.1, RTX 3060 Ti)
- **Backend:** FastAPI + SQLAlchemy 2.x (async) + Pydantic v2
- **Frontend:** Vue 3 (`<script setup>`) + Vite + Tailwind CSS + Pinia + Vue Router 4

See [PROJECT_PLAN.md](PROJECT_PLAN.md) for full architecture and implementation plan.
