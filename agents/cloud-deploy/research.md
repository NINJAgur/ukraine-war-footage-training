# Agent: Cloud Deployment Research
**Domain:** Cloud & DevOps — GCP + Docker Compose

---

## Current Project State
*Last updated: 2026-05-26*

**Architecture fully deployed:**
- CPU services (postgres, redis, backend, frontend, scraper-worker, scraper-beat) → GCP e2-micro free tier, us-central1
- GPU ML worker + Beat → GCP T4 Spot VM (n1-standard-1 + NVIDIA T4), Instance Scheduling 02:00–05:00 UTC daily
- Media storage → GCS bucket `ukraine-footage-media` (public-read for annotated/, private raw/)
- DNS + HTTPS → pending (Cloudflare or Certbot, task 4.20)
- Total cost: ~$0/mo CPU (free tier) + ~$10/mo GPU = **~$10/mo** (free during $300 trial)

**Production compose:** `docker-compose.prod.yml` — CPU services only (no ml-worker/ml-beat)

**T4 Startup script:** Fully automated — NVIDIA drivers (first-boot reboot via sentinel), ffmpeg, sparse repo clone, python venv + deps (torch==2.5.1+cu121 pinned — 2.6+ breaks ultralytics weights_only), model weights downloaded from GCS at `runs/`, Celery GPU worker + embedded Beat (`--beat` flag). systemd service `celery-gpu` starts automatically.

**GCS annotation pipeline confirmed:** scraper uploads `gs://ukraine-footage-media/raw/...` → T4 downloads, runs YOLO, uploads `gs://ukraine-footage-media/annotated/...` → frontend serves directly via `https://storage.googleapis.com/...`

---

## Identity & Role
You are the **Cloud Deployment Research Agent** for the Ukraine Combat Footage project.
Your job is to research infrastructure, deployment patterns, and DevOps tooling for this project.

---

## Architecture

### CPU Stack (GCP e2-micro — free tier)
```
GCP e2-micro (2 vCPU shared, 1GB RAM, us-central1)
└── Docker Compose (docker-compose.prod.yml, minus ml-worker/ml-beat)
    ├── postgres:16-alpine        (named volume: postgres_data)
    ├── redis:7-alpine            (named volume: redis_data)
    ├── scraper-worker            (named volume: scraper_media)
    ├── scraper-beat
    ├── backend (FastAPI)         (named volume: ml_media:ro)
    └── frontend (nginx)          (ports 80, 443)
```

### GPU Stack (GCP T4 Spot VM — Instance Scheduling 02:00–05:00 UTC)
```
GCP n1-standard-1 + T4 Spot (1 vCPU, 3.75GB RAM, 16GB VRAM)
└── Celery GPU worker + Beat (--beat, -Q gpu, --concurrency=1)
    ├── Connects to Redis on e2-micro via internal GCP IP (free, no egress)
    ├── Connects to PostgreSQL on e2-micro via internal GCP IP
    ├── Downloads raw videos from GCS bucket (clips from scraper)
    └── Uploads annotated videos to GCS bucket (served publicly)
```

### Key advantage of same-GCP architecture
Both VMs share the same VPC — ml-worker connects to Redis/Postgres via **internal IP**, no public port exposure needed, zero egress cost.

### Key env vars for GPU worker
```
CELERY_BROKER_URL=redis://<GCP_INTERNAL_IP>:6379/0
CELERY_RESULT_BACKEND=redis://<GCP_INTERNAL_IP>:6379/1
DATABASE_SYNC_URL=postgresql://postgres:<pw>@<GCP_INTERNAL_IP>:5432/ukraine_footage
```

---

## GCP Specifics

### Free tier requirements (e2-micro)
- Must be in `us-west1`, `us-central1`, or `us-east1` — any other region is billed
- 30GB standard persistent disk included free
- 1GB egress/month free

### Firewall — GCP VPC Firewall Rules
GCP uses a single firewall layer (VPC firewall rules) unlike Oracle's two layers.
Console → VPC Network → Firewall → Create Firewall Rule

### Required ports
| Port | Service | Source |
|------|---------|--------|
| 22 | SSH | Your IP |
| 80 | nginx | 0.0.0.0/0 |
| 443 | nginx HTTPS | 0.0.0.0/0 |
| 6379 | Redis | GCP internal only (10.0.0.0/8) |
| 5432 | PostgreSQL | GCP internal only (10.0.0.0/8) |

### T4 Spot VM notes
- Spot VMs can be preempted (terminated) by GCP with 30s notice when capacity is needed
- For training runs: checkpoint saves protect against preemption (YOLO saves best.pt incrementally)
- Price: ~$0.11/hr vs ~$0.35/hr on-demand — 3x cheaper
- Create in same region/zone as e2-micro for internal IP connectivity

---

## Solved Architecture Decisions (reference)

### 1. Weight Distribution → GCS (solved)
Weights uploaded from local Windows via `infra/gcp/upload_weights.py` → T4 startup script downloads from GCS on first boot. Only downloads `best.pt` files under `runs/` prefix. Skips blobs that already exist.

### 2. Media Sharing Between VMs → GCS bucket (solved)
Scraper uploads raw `.mp4` as `gs://ukraine-footage-media/raw/...` → T4 downloads for annotation, uploads result as `gs://ukraine-footage-media/annotated/...` → frontend references `https://storage.googleapis.com/...` directly. No shared filesystem needed.

### 3. T4 PyTorch Version → pin torch==2.5.1+cu121 (solved)
PyTorch 2.6+ changed `weights_only` default from `False` to `True`, breaking `ultralytics==8.2.34`. Pinned in `ml-engine/requirements.txt` with `--extra-index-url https://download.pytorch.org/whl/cu121`.

### 4. Beat Scheduler on T4 → embedded `--beat` (solved)
Single Celery worker process runs both worker and beat via `--beat` flag. Beat schedule defined in `ml-engine/celery_app.py`: `auto_label_batch` @ 02:00 UTC, `annotate_clips` @ 04:00 UTC.

## Open Research Topics

### 5. HTTPS Setup (pending task 4.20)
- Certbot (Let's Encrypt) on e2-micro directly — simplest
- Cloudflare Tunnel (no ports needed, proxies via Cloudflare edge)
- Cloudflare DNS proxy (orange cloud) + Certbot is the recommended combo

### 6. CI/CD (pending task 4.21)
- GitHub Actions `ci.yml` — run unit tests on every push
- `deploy-e2-micro.yml` — SSH in, `git pull`, `docker compose up -d`
- Secrets: `GCP_SSH_PRIVATE_KEY`, `GCP_E2_MICRO_HOST`, `POSTGRES_PASSWORD`, `JWT_SECRET`, `ADMIN_PASSWORD`
- Consider: auto-trigger `infra/gcp/upload_weights.py` after new training run completes

---

## Output Format

1. **Recommendation** — what to do and why
2. **Commands** — exact shell commands ready to run
3. **Gotchas** — what will break if you miss this step
