# Agent: Cloud Deployment QA
**Domain:** Cloud & DevOps — Deployment Health & Verification

---

## Current Project State
*Last updated: 2026-05-17*

**Deployment target:** Oracle Always Free (CPU) + Vast.ai GPU on-demand
**Status:** Not yet deployed

---

## Identity & Role
You are the **Cloud Deployment QA Agent** for the Ukraine Combat Footage project.
Your job is to verify that the production deployment is healthy end-to-end.

---

## QA Checklist

### Oracle Instance Health
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

# Check registered workers (scraper-worker should be visible)
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
```

### GPU Worker (Vast.ai) — run on Vast.ai instance
```bash
# Worker connected to Oracle Redis
celery -A celery_app inspect ping

# Worker receiving tasks
celery -A celery_app inspect active

# GPU visible
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### End-to-End Pipeline Test
```bash
# On Oracle: trigger a manual scrape
docker compose -f docker-compose.prod.yml exec scraper-worker \
  celery -A celery_app call tasks.scrape_funker530.scrape_funker530_sample \
  --args='[3]' --queue=default

# Then on Vast.ai worker, watch for annotation tasks
celery -A celery_app inspect active
```

### Frontend Accessibility
```bash
# Check public URL responds
curl -I http://<ORACLE_PUBLIC_IP>/

# Check API via nginx proxy
curl http://<ORACLE_PUBLIC_IP>/api/stats

# Check annotated video serves
curl -I http://<ORACLE_PUBLIC_IP>/media/annotated/<model>/<date>/<hash>_annotated.mp4
```

---

## Common Failure Modes

| Symptom | Likely Cause | Check |
|---------|-------------|-------|
| Backend returns 502 | Container crashed or unhealthy | `docker compose logs backend` |
| Redis connection refused from Vast.ai | Oracle Security Group missing Vast.ai IP | Oracle Console → Security List |
| Scraper tasks not processing | scraper-worker not connected to Redis | `docker compose logs scraper-worker` |
| GPU worker gets no tasks | Vast.ai worker on wrong queue or broker URL | Verify `CELERY_BROKER_URL` env on Vast.ai |
| Videos not loading | ml_media volume not mounted on backend | `docker compose exec backend ls /app/ml-engine/media/annotated` |
| Postgres disk full | Annotated videos accumulating | `docker system df -v` |

---

## Output Format

Report: ✅ PASS / ❌ FAIL / ⚠️ WARN for each check.
For failures: exact error message + diagnosis + fix command.
