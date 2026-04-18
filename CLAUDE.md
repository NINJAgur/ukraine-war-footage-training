# CLAUDE.md — Ukraine Combat Footage Web Application
> This file is the persistent system prompt for Claude Code. Read it at the start of every session.

---

## Response Format Rules

Every single response MUST end with BOTH of these lines (no exceptions):

1. **What changed:** One sentence summarizing what was implemented or fixed this response.
2. **Token status:** `[session: ~Xk tokens used / Yk remaining]` — estimate from conversation length. Warn explicitly if < 20k remaining.

---

## Project Identity

**Name:** Ukraine Combat Footage Archival System  
**Repo:** `yolo-training-template` (monorepo)  
**Purpose:** Automated full-stack application that scrapes, auto-labels, and publicly displays
archival combat footage from the war in Ukraine, with a secure Admin panel for YOLOv8 model retraining.

---

## Architecture at a Glance

```
[Celery Beat] → [scraper-engine] → PostgreSQL + /media/raw/
                                         ↓
                              [ml-engine: auto_label]
                                         ↓
                    [package_dataset] + [render_annotated]
                           ↓                    ↓
                    Admin Inbox             Public Feed
                           ↓
              [train_baseline] → [train_finetune]
              Stage 1: Kaggle    Stage 2: custom data
```

**Four services, four phases:**

| Service | Directory | Phase |
|---------|-----------|-------|
| Scraper Engine | `scraper-engine/` | Phase 1 |
| ML Engine | `ml-engine/` | Phase 2 |
| Backend API | `web-app/backend/` | Phase 3 |
| Frontend | `web-app/frontend/` | Phase 3 |

---

## Tech Stack Constraints

### Hardware
- **OS:** Windows 11, native Python development inside VSCode
- **GPU:** NVIDIA RTX 3060 Ti — **8GB VRAM hard limit**
- **CUDA:** Provided via `torch+cu121` pip package. No standalone CUDA Toolkit install.
- **Docker:** Only used in Phase 4 for GCP deployment. Do NOT use Docker locally.

### Backend
- **FastAPI** — all route handlers must be `async def`
- **SQLAlchemy 2.x** with async sessions (`AsyncSession`)
- **Pydantic v2** — use `model_config = ConfigDict(...)`, not `class Config`
- **Alembic** for DB migrations
- **PostgreSQL 16** — default port 5432, DB name `ukraine_footage`

### Frontend
- **Vue 3 Composition API ONLY** — `<script setup>` syntax everywhere
- **NO Options API**, no `this`, no Vuex
- **Pinia** for state management
- **Tailwind CSS** dark tactical theme: slate/zinc base, green `#22c55e` accent, red `#ef4444` danger
- **Vite** as build tool
- **Vue Router 4** for client-side routing

### ML Stack
- **Ultralytics YOLOv8** — use the Python API, not subprocess calls
- **GroundingDINO** for zero-shot auto-labeling
- **PyTorch** — always set `device='cuda:0'` explicitly; never default to CPU for training
- **VRAM budget:** YOLOv8m with batch_size=8 uses ~6GB. Max batch_size=8 for 8GB VRAM.
- **OpenCV** for video frame extraction and rendering

### Async Queue
- **Celery** with Redis broker (`redis://localhost:6379/0`)
- **Celery Beat** for scheduled tasks
- GPU tasks run on a dedicated `gpu` queue with `concurrency=1`
- All tasks must be idempotent (safe to retry)

### Scraping
- **Funker530 REST API** — `https://api.funker530.com/api/Get?categoryId=16` with `gettype: Video` header; no Playwright needed
- **GeoConfirmed REST API** — `GET /api/placemark/Ukraine` (list) + `GET /api/placemark/detail/{id}` (detail); filter `origin=="VID"`
- **yt-dlp** for all video downloads (Bunny.net HLS, Rumble, Telegram, Twitter/X) — not YouTube-specific
- **kagglehub** for Kaggle dataset downloads
- De-duplicate by `url_hash` (SHA256 of the canonical URL)
- 3-tier content filtering: tier-1-hc (equipment visible), tier-1 (equipment mentioned), tier-2 (geo only)

---

## Database Schema (Overview)

```
Clip
  id, url, url_hash (unique), source, title, description
  status: PENDING | DOWNLOADING | LABELED | ANNOTATED | ERROR
  file_path, mp4_path, created_at, updated_at

Dataset
  id, name, clip_id (FK), yolo_dir_path, yaml_path
  status: LABELED | QUEUED | TRAINED
  frame_count, class_count, created_at

TrainingRun
  id, stage (BASELINE | FINETUNE), status (QUEUED | RUNNING | DONE | ERROR)
  dataset_ids (JSON array), weights_path, metrics (JSON)
  started_at, completed_at, celery_task_id
```

---

## Key File Locations

| What | Where |
|------|-------|
| Training entry point | `ml-engine/core/main.py` |
| Inference script | `ml-engine/core/inference.py` |
| Auto-label script | `ml-engine/core/autolabeling/auto_label.py` |
| Preprocessing utils | `ml-engine/core/preprocessing.py` |
| Funker530 scraper task | `scraper-engine/tasks/scrape_funker530.py` |
| GeoConfirmed scraper task | `scraper-engine/tasks/scrape_geoconfirmed.py` |
| Phase 1 integration test | `scraper-engine/tests/test_scrape_live.py` |
| Project plan (source of truth) | `PROJECT_PLAN.md` |
| Agent personas | `agents/` |
| Coding rules | `rules/` |
| CLI commands | `commands/` |

**Run Phase 1 test:**
```bash
cd scraper-engine && python tests/test_scrape_live.py
```

---

## Coding Conventions

1. **Python:** Use f-strings, type hints everywhere, `pathlib.Path` not `os.path`
2. **Async:** Use `asyncio` patterns; never `time.sleep()` in async context — use `asyncio.sleep()`
3. **Logging:** Use Python `logging` module, not `print()`. Format: `%(asctime)s [%(levelname)s] %(name)s: %(message)s`
4. **Errors:** Raise specific exceptions; never `except Exception: pass`
5. **Secrets:** All credentials via environment variables; never hardcode
6. **Tests:** Integration tests preferred over unit tests for this project (real DB, real Redis)

---

## Phase Execution Order

| Phase | Focus | Status |
|-------|-------|--------|
| **0** | Agentic workspace init | ✅ Complete |
| **1** | Data Ingestion (scraper-engine) | ✅ Complete |
| **2** | ML Pipeline (ml-engine) | 🔄 Next |
| **3** | Web Application | ⏳ Pending |
| **4** | Cloud & DevOps | ⏳ Pending |

---

## Environment Variables (key ones)

```
DATABASE_SYNC_URL=postgresql://postgres:postgres@localhost:5432/ukraine_footage
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/ukraine_footage
REDIS_URL=redis://localhost:6379/0
KAGGLE_USERNAME=...
KAGGLE_KEY=...
JWT_SECRET=...
ADMIN_PASSWORD=...
```

Do NOT set `MEDIA_ROOT`, `RAW_VIDEO_DIR`, or `DATASETS_DIR` in `.env` — Python defaults (relative to `__file__`) are correct and CWD-independent.

---

## Do NOT

- Do NOT use Docker for local development (Windows native only until Phase 4)
- Do NOT use Options API in Vue components
- Do NOT use synchronous SQLAlchemy calls in FastAPI routes
- Do NOT use `print()` for logging in production code
- Do NOT hardcode credentials or file paths
- Do NOT start new Celery workers with `concurrency > 1` for GPU tasks
- Do NOT write code for phases not yet reached (stay sequential)
