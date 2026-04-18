First, read the file `agents/ingestion/qa.md` to load the full QA checklist for the ingestion pipeline.

You are now acting as the Ingestion QA Agent.

Run a QA check against the live database and filesystem. For each checklist item:
1. Write and execute the SQL query or file check
2. Report PASS / FAIL / WARN with the actual count or value found
3. Flag anything that needs immediate attention

$ARGUMENTS

Start by connecting to PostgreSQL: `psql $DATABASE_SYNC_URL -c "SELECT status, COUNT(*) FROM clips GROUP BY status ORDER BY status;"`
