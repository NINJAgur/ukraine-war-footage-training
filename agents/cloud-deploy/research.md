# Agent: Cloud Deployment Research
**Domain:** Cloud & DevOps — GCP + Docker Compose

---

## Current Project State
*Last updated: 2026-05-28*

**Architecture fully deployed (3-VM split):**
- CPU services (postgres, redis, backend, frontend, scraper-worker, scraper-beat) → GCP e2-micro free tier, us-central1
- Inference-engine (GDINO auto-label + YOLO annotation + dataset packaging) → n1-standard-1 + T4 Spot, Instance Scheduling 03:00–04:00 UTC daily
- Training-engine (YOLO baseline + finetune training) → n1-standard-4 + T4 Spot, on-demand (started by inference-engine via GCP Compute API)
- Media storage → GCS bucket `ukraine-footage-media` (public-read for annotated/, private raw/)
- DNS + HTTPS → **ukrarchive.duckdns.org** (DuckDNS free subdomain + Let's Encrypt via Certbot, expires 2026-08-24, auto-renews via certbot systemd timer)
- Total cost: ~$0/mo CPU (free tier) + ~$10/mo GPU = **~$10/mo** (free during $300 trial)

**Production compose:** `docker-compose.prod.yml` — CPU services only (no GPU workers)

**Inference-engine VM startup:** Fully automated — NVIDIA drivers (first-boot reboot via sentinel), ffmpeg, sparse clone `inference-engine/ shared/`, python venv + deps (torch==2.5.1+cu121 pinned — 2.6+ breaks ultralytics weights_only), model weights downloaded from GCS at `runs/`, systemd service `celery-inference` (`celery -A celery_app worker -Q pipeline --pool=solo --concurrency=1 --beat`).

**Training-engine VM startup:** On-demand only, started by inference-engine via GCP Compute Engine API (`_start_training_vm`). Sparse clone `training-engine/ shared/`, mounts persistent datasets disk at `/mnt/datasets` (150GB Kaggle cache). systemd service `celery-training` (`celery -A celery_app worker -Q training --concurrency=1`). Self-shuts-down after last model trained (`sudo shutdown -h now`).

**GCS annotation pipeline confirmed:** scraper uploads `gs://ukraine-footage-media/raw/...` → inference-engine downloads, runs YOLO, uploads `gs://ukraine-footage-media/annotated/...` → frontend serves directly via `https://storage.googleapis.com/...`

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

### GPU Stack — inference-engine (daily 03:00–04:00 UTC)
```
GCP n1-standard-1 + T4 Spot — Instance Schedule (start 03:00 / stop 04:00 UTC)
└── Celery Q=pipeline worker + Beat (--beat, --pool=solo, --concurrency=1)
    Beat schedule:
      03:05 UTC: auto_label_batch   → GDINO + package_dataset chord
      03:35 UTC: annotate_clips     → YOLO annotation of raw clips
    Tasks:
      auto_label_clip     — GDINO frames → canonical nc=3 labels → Dataset(LABELED)
      package_dataset     — 80/20 split → merged/<MODEL>/ → Dataset(PACKAGED)
      trigger_finetune_check — chord callback: ≥5 PACKAGED → TrainingRun(QUEUED) + prepare_finetune_batch
      prepare_finetune_batch — upload merged dirs to GCS (remote) → start training VM → dispatch train_finetune
      annotate_clips      — YOLO inference → annotated MP4 → GCS → Clip(ANNOTATED)
    Connects to Redis/Postgres on e2-micro via GCP internal IP (free, no egress)
```

### GPU Stack — training-engine (on-demand, started by inference-engine)
```
GCP n1-standard-4 + T4 Spot — on-demand (started via GCP Compute Engine API)
└── Celery Q=training worker (--concurrency=1)
    Tasks:
      train_finetune — download merged dir from GCS → train YOLO → upload best.pt → shutdown
    Persistent disk: 150GB at /mnt/datasets (Kaggle merged datasets)
    Self-shuts-down after last train_finetune completes (sudo shutdown -h now)
    Connects to Redis/Postgres on e2-micro via GCP internal IP
```

### All VMs share the same VPC — internal IP connectivity, zero egress cost

### Key env vars for GPU VMs
```
CELERY_BROKER_URL=redis://<E2_MICRO_INTERNAL_IP>:6379/0
CELERY_RESULT_BACKEND=redis://<E2_MICRO_INTERNAL_IP>:6379/1
DATABASE_SYNC_URL=postgresql+psycopg2://postgres:<pw>@<E2_MICRO_INTERNAL_IP>:5432/ukraine_footage
REDIS_URL=redis://<E2_MICRO_INTERNAL_IP>:6379/0
STORAGE_MODE=remote
REMOTE_STORAGE_BUCKET=ukraine-footage-media
GCP_PROJECT_ID=<project_id>
GCP_TRAINING_VM_ZONE=us-central1-a
GCP_TRAINING_VM_NAME=ukraine-footage-training
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

### 4. Beat Scheduler on inference-engine → embedded `--beat` (solved)
Single Celery worker process runs both worker and beat via `--beat` flag. Beat schedule defined in `inference-engine/celery_app.py`: `auto_label_batch` @ 03:05 UTC (after VM starts at 03:00), `annotate_clips` @ 03:35 UTC (waits behind GDINO tasks in Q=pipeline).

## Solved Architecture Decisions (continued)

### 5. HTTPS → DuckDNS + Let's Encrypt Certbot (solved)
- Free subdomain `ukrarchive.duckdns.org` via duckdns.org
- Cert issued via `certbot certonly --standalone`, mounted into nginx container at `/etc/letsencrypt:ro`
- nginx.conf: HTTP → 301 redirect to HTTPS; TLSv1.2/1.3 on port 443
- Auto-renewal: certbot systemd timer (installed by default), pre-hook stops nginx container, post-hook starts it
- GCP firewall must allow port 80 from `0.0.0.0/0` for certbot standalone challenge

### 6. CI/CD → GitHub Actions (solved)
- `.github/workflows/ci.yml` — frontend build + ruff lint on all pushes/PRs
- `.github/workflows/deploy-e2-micro.yml` — SSH deploy via `appleboy/ssh-action` triggered by `workflow_run` after CI passes
- `.github/workflows/deploy-weights.yml` — manual `workflow_dispatch` to upload weights from T4 to GCS
- Required GitHub secrets: `E2_MICRO_HOST`, `E2_MICRO_SSH_KEY`, `T4_SSH_KEY` (weights only)
- e2-micro uses sparse checkout (`web-app/`, `scraper-engine/`, `shared/`, `.github/`) — no ml-engine or docs pulled

---

## Output Format

1. **Recommendation** — what to do and why
2. **Commands** — exact shell commands ready to run
3. **Gotchas** — what will break if you miss this step
