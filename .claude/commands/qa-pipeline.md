Load context from both `agents/ingestion/qa.md` and `agents/ml-pipeline/qa.md`.

You are running a full end-to-end pipeline QA check across both scraper-engine and ml-engine.

Check in order:
1. **Scraper health** — apply the Ingestion QA checklist: DB integrity, file system, dedup, Celery task state
2. **ML pipeline health** — apply the ML QA checklist: frame extraction counts, label file validity, dataset packaging, training run statuses
3. **Cross-service state** — verify Clip status transitions are coherent (no clips stuck in QUEUED/DOWNLOADING for >1 hour)

$ARGUMENTS

Start with: `psql $DATABASE_SYNC_URL -c "SELECT status, COUNT(*) FROM clips GROUP BY status; SELECT status, COUNT(*) FROM datasets GROUP BY status; SELECT stage, model_type, status, COUNT(*) FROM training_runs GROUP BY stage, model_type, status;"`
