# /scrape — Manually Trigger a Scraping Run

## Description
Triggers a manual, on-demand scraping task for a specific source (or all sources).
Use this to test scraper functionality, backfill missing content, or re-run a failed scrape.

## Usage
```
/scrape [source] [--url URL]
```

### Arguments
| Argument | Values | Description |
|----------|--------|-------------|
| `source` | `funker530`, `geoconfirmed`, `kaggle`, `all` | Which source to scrape. Defaults to `all`. |
| `--url` | URL string | Optional: scrape a specific URL instead of running the full batch |

### Examples
```bash
/scrape funker530          # scrape latest Funker530 posts
/scrape geoconfirmed       # fetch latest GeoConfirmed video incidents
/scrape kaggle             # pull configured Kaggle datasets
/scrape all                # trigger all three scrapers
```

## What This Command Does

1. Verify Redis is reachable: `redis-cli ping`
2. Verify the scraper-engine Celery worker is running
3. Dispatch the appropriate Celery task:
   - `scrape_funker530.delay()` for funker530
   - `scrape_geoconfirmed.delay()` for geoconfirmed
   - `download_kaggle.delay()` for kaggle
4. Return the Celery task ID for monitoring
5. Poll task status and print result when complete

## Implementation Steps (for Claude to follow)

```python
# Check Redis
import redis
r = redis.from_url(os.environ["REDIS_URL"])
r.ping()

# Dispatch task
from scraper_engine.tasks.scrape_funker530 import scrape_funker530
task = scrape_funker530.delay()
print(f"Task dispatched: {task.id}")

# Monitor
result = task.get(timeout=300)
print(f"Result: {result}")
```

## Troubleshooting

- **"Redis connection refused"**: Start Redis with `redis-server --daemonize yes` in WSL2
- **"No worker ready"**: Start Celery worker: `celery -A scraper_engine.celery_app worker -Q default --loglevel=info`
- **"Task already running"**: Task has a Redis lock — wait for it to finish or delete the lock key manually
