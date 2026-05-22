# Agent: Cloud Deployment Research
**Domain:** Cloud & DevOps — GCP + Docker Compose

---

## Current Project State
*Last updated: 2026-05-22*

**Target architecture decided:**
- CPU services (postgres, redis, backend, frontend, scraper) → GCP e2-micro (free tier, us-central1/us-west1/us-east1, $0/mo permanent)
- GPU ML worker + ML Beat → GCP T4 Spot VM (on-demand, ~$0.11/hr, ~$10/mo at 3hrs/day)
- DNS + HTTPS → Cloudflare (free, future)
- Total cost: ~$0/mo CPU (free tier) + ~$10/mo GPU = **~$10/mo** (free during $300 trial)

**Production compose:** `docker-compose.prod.yml` — exists, ml-worker/ml-beat run on separate GCP T4 Spot VM (not on e2-micro)

**Deployment status:** GCP account activated, $300 trial credit active — setup in progress

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

### GPU Stack (GCP T4 Spot VM — on-demand)
```
GCP n1-standard-4 + T4 Spot (4 vCPU, 15GB RAM, 16GB VRAM)
└── Celery ml-worker (-Q gpu, --concurrency=1)
    ├── Connects to Redis on e2-micro via internal GCP IP (free, no egress)
    ├── Connects to PostgreSQL on e2-micro via internal GCP IP
    └── Reads/writes ml_media (GCP persistent disk or GCS bucket)
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

## Research Goals

### 1. Volume Seeding Strategy
How to get model weights + annotated media from local Windows machine to GCP:
- SCP from local → GCP e2-micro (slow but simple)
- `gcloud compute scp` (easier than raw SCP, handles auth)
- GCS bucket as intermediary (upload from Windows, pull from GCP VM)

### 2. Media Sharing Between VMs
Annotated videos written by T4 Spot worker need to be accessible to backend on e2-micro:
- GCS bucket mounted on both VMs via gcsfuse
- NFS share from e2-micro (simple but adds latency)
- Periodic rsync from T4 → e2-micro after each annotation run

### 3. HTTPS Setup
- Certbot (Let's Encrypt) on e2-micro directly
- Cloudflare Tunnel (no ports needed, proxies via Cloudflare edge)

### 4. Monitoring & Alerting
- Docker stats + logs (basic, already available)
- Uptime monitoring (UptimeRobot free tier)
- GCP Cloud Monitoring (free tier available)

---

## Output Format

1. **Recommendation** — what to do and why
2. **Commands** — exact shell commands ready to run
3. **Gotchas** — what will break if you miss this step
