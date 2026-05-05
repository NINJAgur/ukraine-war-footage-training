# Ukraine Combat Footage Archival System

Automated full-stack application that scrapes, auto-labels, and publicly displays
archival combat footage from the war in Ukraine, with a secure admin panel for
YOLOv8 model retraining.

## Architecture

```
[Celery Beat] → scraper-engine → PostgreSQL + media/raw/
                                       ↓
                            ml-engine: auto_label (Phase 2)
                                       ↓
                    package_dataset + render_annotated
                           ↓                    ↓
                    Admin Inbox             Public Feed (Phase 3)
                           ↓
              train_baseline → train_finetune
```

## Services

| Service | Directory | Phase | Status |
|---------|-----------|-------|--------|
| Scraper Engine | `scraper-engine/` | 1 | ✅ Complete |
| ML Engine | `ml-engine/` | 2 | 🔄 In progress |
| Backend API | `web-app/backend/` | 3 | 🔄 In progress |
| Frontend | `web-app/frontend/` | 3 | 🔄 In progress |

## ML Training Progress

| Model | mAP50 | Images | Status |
|-------|-------|--------|--------|
| AIRCRAFT | 0.929 | 83K | ✅ run 13 |
| VEHICLE | 0.871 | 87K | ✅ run 25 |
| PERSONNEL | — | ~25K | ⏳ next |
| GENERAL | — | ~175K | ⏳ after specialists |

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
# Backend
cd web-app/backend && uvicorn main:app --reload --port 8001

# Frontend
cd web-app/frontend && npm run dev

# PERSONNEL baseline training
cd ml-engine && python tests/test_baseline_train.py --model-type PERSONNEL --epochs 10 --keep
```

## Tech Stack

- **Scraping:** Funker530 REST API + GeoConfirmed REST API + yt-dlp
- **Queue:** Celery + Redis
- **Database:** PostgreSQL 16
- **ML:** Ultralytics YOLOv8 + GroundingDINO + PyTorch (CUDA 12.1)
- **Frontend:** Vue 3 + Vite + Tailwind CSS
- **API:** FastAPI + SQLAlchemy 2.x

See [PROJECT_PLAN.md](PROJECT_PLAN.md) for full architecture and implementation plan.
