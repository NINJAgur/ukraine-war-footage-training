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
| ML Engine | `ml-engine/` | 2 | 🔄 Next |
| Backend API | `web-app/backend/` | 3 | ⏳ Pending |
| Frontend | `web-app/frontend/` | 3 | ⏳ Pending |

## Quick Start

```bash
# 1. Install scraper-engine deps
cd scraper-engine && pip install -r requirements.txt

# 2. Copy and fill environment
cp .env.example .env   # edit DATABASE_SYNC_URL, REDIS_URL

# 3. Run Phase 1 live test (auto-cleans DB + media/raw before running)
cd scraper-engine && python ../tests/scraper_engine/test_scrape_live.py
```

## Tech Stack

- **Scraping:** Funker530 REST API + GeoConfirmed REST API + yt-dlp
- **Queue:** Celery + Redis
- **Database:** PostgreSQL 16
- **ML:** Ultralytics YOLOv8 + GroundingDINO + PyTorch (CUDA 12.1)
- **Frontend:** Vue 3 + Vite + Tailwind CSS
- **API:** FastAPI + SQLAlchemy 2.x

See [PROJECT_PLAN.md](PROJECT_PLAN.md) for full architecture and implementation plan.
