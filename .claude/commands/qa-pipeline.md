Load context from both `agents/ingestion/qa.md` and `agents/ml-pipeline/qa.md`.

You are running a full end-to-end pipeline QA check across scraper-engine, inference-engine, and training-engine.

Check in order:
1. **Scraper health** — apply the Ingestion QA checklist: DB integrity, file system / GCS, dedup, Celery task state
2. **ML pipeline health** — apply the ML QA checklist: GDINO label validity, dataset packaging, cleanup (frames/, hash dirs, merged dirs), training run statuses
3. **Cross-service state** — verify Clip status transitions are coherent (no clips stuck in QUEUED/DOWNLOADING for >1 hour)

$ARGUMENTS

Start with:
```sql
SELECT status, COUNT(*) FROM clips GROUP BY status ORDER BY status;
SELECT status, COUNT(*) FROM datasets GROUP BY status ORDER BY status;
SELECT stage, model_type, status, metrics->>'map50' as map50, id FROM training_runs ORDER BY id DESC LIMIT 10;
```
