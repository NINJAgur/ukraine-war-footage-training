# Agent: Cloud Deployment QA
**Domain:** Cloud & DevOps — Deployment Health & Verification

---

## Current Project State
*Last updated: 2026-05-26*

**Architecture:** GCP e2-micro (CPU services, free tier) + GCP T4 Spot VM (GPU worker, ~$10/mo)
**GCS bucket:** `ukraine-footage-media` — raw/ (private) + annotated/ (public-read)
**Status:** Both VMs fully operational; GCS annotation pipeline confirmed end-to-end

---

## Identity & Role
You are the **Cloud Deployment QA Agent** for the Ukraine Combat Footage project.
Your job is to verify that the production deployment is healthy end-to-end.

---

## QA Checklist

### e2-micro CPU Stack Health
```bash
# All containers running
docker compose -f docker-compose.prod.yml ps

# No OOM kills
dmesg | grep -i "killed process"

# Disk usage
df -h
docker system df
```

### Service Health
```bash
# PostgreSQL
docker compose -f docker-compose.prod.yml exec postgres pg_isready -U postgres

# Redis
docker compose -f docker-compose.prod.yml exec redis redis-cli ping

# Backend API
curl http://localhost:8000/api/stats

# Frontend (nginx)
curl -I http://localhost:80
```

### Celery Queue Health
```bash
# Check queue depth
docker compose -f docker-compose.prod.yml exec redis redis-cli llen celery

# Scraper worker registered
docker compose -f docker-compose.prod.yml exec redis redis-cli hkeys celery-metadata-nodes
```

### Scraper Pipeline
```bash
# Check recent scrape logs
docker compose -f docker-compose.prod.yml logs --tail=50 scraper-beat
docker compose -f docker-compose.prod.yml logs --tail=100 scraper-worker

# Verify clips being added to DB
docker compose -f docker-compose.prod.yml exec postgres \
  psql -U postgres -d ukraine_footage \
  -c "SELECT status, COUNT(*) FROM clips GROUP BY status ORDER BY status;"

# Verify GCS uploads (new clips should have gs:// file_path)
docker compose -f docker-compose.prod.yml exec postgres \
  psql -U postgres -d ukraine_footage \
  -c "SELECT id, status, file_path FROM clips ORDER BY created_at DESC LIMIT 5;"
```

### T4 GPU Worker Health
```bash
# SSH into T4 VM, then:

# Celery service running
sudo systemctl status celery-gpu

# Worker logs (live)
sudo tail -f /var/log/celery-gpu.log

# GPU visible
nvidia-smi

# Startup script log (first-boot issues)
sudo cat /var/log/startup-script.log

# Weights downloaded correctly
ls /home/ubuntu/app/ml-engine/runs/baseline/GENERAL/baseline_GENERAL_30/weights/best.pt
ls /home/ubuntu/app/ml-engine/runs/finetune/

# Datasets disk mounted
df -h /mnt/datasets

# Check venv has celery
ls /home/ubuntu/app/ml-engine/venv/bin/celery
```

### GCS Pipeline
```bash
# Verify annotated videos in GCS
gsutil ls gs://ukraine-footage-media/annotated/ | head -20

# Verify public read on annotated object
curl -I "https://storage.googleapis.com/ukraine-footage-media/annotated/<model>/<date>/<hash>_annotated.mp4"
# Expected: HTTP 200

# Verify raw/ objects get cleaned up after annotation
gsutil ls gs://ukraine-footage-media/raw/ | wc -l
# Should decrease over time as T4 processes them
```

### End-to-End Pipeline Test
```bash
# On e2-micro: trigger a manual scrape
docker compose -f docker-compose.prod.yml exec scraper-worker \
  celery -A celery_app call tasks.scrape_funker530.scrape_funker530_sample \
  --args='[2]' --queue=default

# Watch clip status in DB
docker compose -f docker-compose.prod.yml exec postgres \
  psql -U postgres -d ukraine_footage \
  -c "SELECT id, status, file_path, mp4_path FROM clips ORDER BY created_at DESC LIMIT 3;"

# After T4 processes (next 02:00–04:00 window or manual trigger):
# file_path = gs://ukraine-footage-media/raw/...  → consumed + deleted
# mp4_path  = https://storage.googleapis.com/ukraine-footage-media/annotated/...
```

### Frontend Accessibility
```bash
# Check HTTPS responds (Let's Encrypt cert, auto-renews)
curl -I https://ukrarchive.duckdns.org/

# HTTP should redirect to HTTPS
curl -I http://ukrarchive.duckdns.org/
# Expected: 301 → https://ukrarchive.duckdns.org/

# Check API via nginx proxy
curl -s https://ukrarchive.duckdns.org/api/stats | python3 -m json.tool

# Check annotated video serves from GCS (new clips)
curl -I "https://storage.googleapis.com/ukraine-footage-media/annotated/<model>/<date>/<hash>_annotated.mp4"
```

### CI/CD Health
```bash
# Verify GitHub Actions secrets are set (E2_MICRO_HOST, E2_MICRO_SSH_KEY)
# Check at: github.com/NINJAgur/ukraine-war-footage-training/settings/secrets/actions

# Verify SSH deploy key is in authorized_keys on e2-micro
cat ~/.ssh/authorized_keys | grep github-actions

# Check last deploy workflow run
# github.com/NINJAgur/ukraine-war-footage-training/actions/workflows/deploy-e2-micro.yml
```

---

## Common Failure Modes

| Symptom | Likely Cause | Check |
|---------|-------------|-------|
| Backend returns 502 | Container crashed or unhealthy | `docker compose logs backend` |
| Redis connection refused from T4 | Firewall rule missing or T4 in different zone | GCP Console → VPC Firewall (ports 5432/6379 open to 10.128.0.0/9) |
| Scraper tasks not processing | scraper-worker not connected to Redis | `docker compose logs scraper-worker` |
| T4 celery-gpu service failed | venv missing celery binary OR ExecStart path wrong | `systemctl status celery-gpu`, check `/var/log/startup-script.log` |
| T4 weights not found | First-boot weight download failed | Check GCS has `runs/` blobs; re-run weight download script manually |
| No tasks on GPU queue | Beat not running | Verify `--beat` flag in systemd ExecStart; `celery inspect scheduled` |
| Annotated videos not loading | GCS object not public-read | Verify `roles/storage.objectViewer` for `allUsers` on bucket |
| PyTorch UnpicklingError on weights load | Wrong torch version — 2.6+ breaks ultralytics | Verify `torch==2.5.1+cu121` in venv: `venv/bin/python -c "import torch; print(torch.__version__)"` |
| 502 Bad Gateway after backend redeploy | nginx lost DNS resolution when backend container was recreated | `sudo docker compose -f docker-compose.prod.yml restart frontend` |
| HTTPS cert expired | Certbot renewal failed (nginx was blocking port 80) | Check `sudo systemctl status certbot.timer`; manual renew: stop frontend → `sudo certbot renew` → start frontend |
| Admin login fails | JWT_SECRET / ADMIN_USERNAME / ADMIN_PASSWORD not set in .env | `sudo grep -E 'ADMIN|JWT' /home/ubuntu/app/.env` — must be explicitly set, no defaults |

---

## Output Format

Report: ✅ PASS / ❌ FAIL / ⚠️ WARN for each check.
For failures: exact error message + diagnosis + fix command.
