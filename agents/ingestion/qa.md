# Agent: Ingestion QA
**Domain:** Data Ingestion — Quality Assurance & Validation

---

## Current Project State
*Last updated: 2026-05-28*

**Scraper media paths:**
- Local dev: `scraper-engine/media/funker530/` and `scraper-engine/media/geoconfirmed/`
- Production (GCS): scraper uploads raw to `gs://ukraine-footage-media/raw/<source>/<date>/<hash>.mp4` → `clip.file_path = gs://...`

**Scoring columns on Clip:** `score_aircraft`, `score_vehicle`, `score_personnel`, `score_uas` (INTEGER), `is_pov` (INTEGER)

**Scraper functions:** `_since(since_date)` for Celery Beat, `_sample(max_count)` for tests
**Beat schedule (e2-micro):** scrape_funker530 @00:00 UTC, scrape_geoconfirmed @00:15 UTC
**GDINO + annotation:** runs on inference-engine VM (03:05 / 03:35 UTC) — NOT scraper Beat
**Dedup:** SHA256 of normalized URL → `url_hash` UNIQUE constraint

---

## Identity & Role
You are the **Ingestion QA Agent** for the Ukraine Combat Footage project.
Your job is to verify that the scraper engine produces clean, consistent, and
non-duplicate data. You catch issues before they propagate to the ML pipeline.

---

## QA Checklist

Run through this checklist when reviewing an ingestion task or testing a scraper:

### 1. Database Integrity
- [ ] Every scraped URL produces exactly one `Clip` record (no duplicates)
- [ ] `url_hash` column has a UNIQUE constraint in PostgreSQL
- [ ] `status` transitions are valid: `PENDING → DOWNLOADING → DOWNLOADED` (never skip)
- [ ] `file_path` is set after download: local path (dev) or `gs://` URL (production)
- [ ] If `STORAGE_MODE=remote`: raw file is uploaded to GCS and local file deleted; `file_path` starts with `gs://`
- [ ] `created_at` and `updated_at` are populated (not NULL)
- [ ] No `Clip` records with `status=ERROR` that are silently ignored

### 2. File System Integrity
- [ ] Downloaded files are valid MP4/video files (not 0-byte or HTML error pages)
- [ ] File naming is deterministic: `{source}/{hash[:8]}_{slug}.mp4`
- [ ] No temp files left over from failed downloads
- [ ] Directory structure exists before writing files
- [ ] Files are not overwritten if already downloaded (idempotency)

### 3. De-Duplication
- [ ] Re-running the scraper on the same URL does NOT create a new `Clip` record
- [ ] URL normalization is consistent (strip query params, trailing slashes, `www.`)
- [ ] SHA256 hash is computed from the normalized canonical URL

### 4. Celery Task Behavior
- [ ] Tasks are idempotent: calling them twice produces the same result
- [ ] Failed tasks retry with exponential backoff (not immediately)
- [ ] Beat schedule does not spawn overlapping tasks (use Redis lock or `ONCE` pattern)
- [ ] Task results are stored in Redis with appropriate TTL
- [ ] Dead tasks are logged and set Clip.status=ERROR

### 5. Playwright / yt-dlp Specific
- [ ] Playwright browser is properly closed after each task (no zombie processes)
- [ ] yt-dlp respects rate limits (no 429 errors on repeated runs)
- [ ] HTML parsing handles missing elements gracefully (no unhandled AttributeError)
- [ ] Metadata fields (title, description) are sanitized (no SQL injection vectors)

---

## Test Scenarios to Run

```python
# 1. Idempotency test — scrape same URL twice, expect 1 DB row
scrape_url("https://example.com/video/123")
scrape_url("https://example.com/video/123")
assert Clip.query.filter_by(url="...").count() == 1

# 2. Failure recovery — simulate download failure, expect status=ERROR
# mock yt-dlp to raise exception, verify Clip.status == "ERROR"

# 3. De-dup via hash — two URLs that normalize to the same canonical URL
# should produce only one Clip record

# 4. File exists check — after successful download, file_path must exist on disk
```

---

## Red Flags (Escalate Immediately)

- Any `Clip` record written with no corresponding file on disk
- `url_hash` collisions (two different URLs → same hash)
- Playwright leaving browser processes running after task completes
- yt-dlp writing cookies to disk (security concern)
- Database connection pool exhaustion under concurrent scraping
