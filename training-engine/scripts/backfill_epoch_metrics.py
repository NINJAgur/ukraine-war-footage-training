"""
training-engine/scripts/backfill_epoch_metrics.py

One-time script:
  1. Reads results.csv from every local training run → stores epoch data in DB
  2. Uploads all run PNG artifacts (confusion_matrix, BoxPR, P/R/F1 curves,
     results.png, train_batch, val_batch) to GCS at:
       runs/{stage}/{MODEL}/{run_name}/{filename}
  3. Saves GCS image URLs dict to TrainingRun.metrics["run_images"]

Connects to prod DB via the e2-micro backend container over SSH.

Usage:
    python training-engine/scripts/backfill_epoch_metrics.py

Requirements: google-cloud-storage, psycopg2-binary
Run from repo root.
"""
import csv
import json
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("backfill")

REPO_ROOT   = Path(__file__).resolve().parents[2]
RUNS_DIR    = REPO_ROOT / "training-engine" / "runs"
GCS_BUCKET  = "ukraine-footage-media"
E2_IP       = "104.154.153.143"
SSH_KEY     = str(Path.home() / ".ssh" / "gcp-deploy")

PNG_NAMES = [
    "confusion_matrix.png",
    "confusion_matrix_normalized.png",
    "BoxPR_curve.png",
    "BoxP_curve.png",
    "BoxR_curve.png",
    "BoxF1_curve.png",
    "results.png",
]

BATCH_NAMES = [
    "train_batch0.jpg", "train_batch1.jpg", "train_batch2.jpg",
    "val_batch0_labels.jpg", "val_batch0_pred.jpg",
    "val_batch1_labels.jpg", "val_batch1_pred.jpg",
]


def parse_csv(csv_path: Path) -> list[dict]:
    rows = []
    try:
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    rows.append({k.strip(): float(v.strip()) for k, v in row.items() if v.strip()})
                except (ValueError, TypeError):
                    pass
    except Exception as e:
        log.warning(f"Could not parse {csv_path}: {e}")
    return rows


def upload_to_gcs(local_path: Path, gcs_key: str):
    try:
        from google.cloud import storage
        client = storage.Client()
        blob = client.bucket(GCS_BUCKET).blob(gcs_key)
        blob.upload_from_filename(str(local_path))
        return f"https://storage.googleapis.com/{GCS_BUCKET}/{gcs_key}"
    except Exception as e:
        log.warning(f"  GCS upload failed for {gcs_key}: {e}")
        return None


def update_db(run_id: int, updates: dict) -> None:
    """Write updates JSON to a temp file, SCP to e2-micro, run python update."""
    import tempfile

    tmp = Path(tempfile.mktemp(suffix=".json"))
    tmp.write_text(json.dumps({"run_id": run_id, "updates": updates}))

    # SCP temp file to e2-micro
    scp_cmd = ["scp", "-i", SSH_KEY, "-o", "StrictHostKeyChecking=no",
               str(tmp), f"ubuntu@{E2_IP}:/tmp/backfill_{run_id}.json"]
    subprocess.run(scp_cmd, capture_output=True, timeout=30)

    # Run python inside backend container to read file and update DB
    python_cmd = (
        "sudo docker exec $(sudo docker ps -qf name=backend) python -c \""
        "import json, os; "
        f"d=json.load(open('/tmp/backfill_{run_id}.json')); "
        "from sqlalchemy import create_engine, text; "
        "engine=create_engine(os.environ['DATABASE_SYNC_URL']); "
        "conn=engine.connect(); "
        f"row=conn.execute(text('SELECT metrics FROM training_runs WHERE id={run_id}')).fetchone(); "
        "m=dict(row[0] or {}); m.update(d['updates']); "
        f"conn.execute(text('UPDATE training_runs SET metrics=:m WHERE id={run_id}'), {{'m':json.dumps(m)}}); "
        "conn.commit(); print('OK')\""
    )
    # Copy file into container first
    cp_cmd = (
        f"sudo docker cp /tmp/backfill_{run_id}.json "
        f"$(sudo docker ps -qf name=backend):/tmp/backfill_{run_id}.json"
    )
    ssh_cmd = ["ssh", "-i", SSH_KEY, "-o", "StrictHostKeyChecking=no", f"ubuntu@{E2_IP}",
               f"{cp_cmd} && {python_cmd}"]
    result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=60)
    tmp.unlink(missing_ok=True)

    if "OK" in result.stdout:
        log.info(f"  DB updated for run {run_id}")
    else:
        log.warning(f"  DB update failed for run {run_id}: {result.stderr[:200]}")


def run() -> None:
    if not RUNS_DIR.exists():
        log.error(f"Runs directory not found: {RUNS_DIR}")
        sys.exit(1)

    csv_files = list(RUNS_DIR.rglob("results.csv"))
    log.info(f"Found {len(csv_files)} run directories")

    for csv_path in sorted(csv_files):
        run_dir  = csv_path.parent
        run_name = run_dir.name                        # e.g. finetune_AIRCRAFT_68
        parts    = run_name.rsplit("_", 1)
        if len(parts) != 2 or not parts[1].isdigit():
            log.warning(f"Cannot parse run_id from {run_name} — skipping")
            continue

        run_id = int(parts[1])
        stage_model = run_dir.parent.parent.name + "/" + run_dir.parent.name  # baseline/AIRCRAFT
        gcs_prefix = f"runs/{run_dir.parent.parent.name.lower()}/{run_dir.parent.name}/{run_name}"

        log.info(f"\n─── run {run_id}: {run_name} ───")
        updates: dict = {}

        # 1. Epoch data from CSV
        epochs = parse_csv(csv_path)
        if epochs:
            updates["epochs_data"] = epochs
            best = max(e.get("metrics/mAP50(B)", 0) for e in epochs)
            log.info(f"  CSV: {len(epochs)} epochs, best mAP50={best:.3f}")
        else:
            log.warning("  CSV: empty or unparseable")

        # 2. Upload PNGs to GCS
        images: dict = {}
        for fname in PNG_NAMES + BATCH_NAMES:
            fpath = run_dir / fname
            if not fpath.exists():
                continue
            gcs_key = f"{gcs_prefix}/{fname}"
            url = upload_to_gcs(fpath, gcs_key)
            if url:
                images[fname.replace(".png", "").replace(".jpg", "")] = url
                log.info(f"  Uploaded: {fname}")

        if images:
            updates["run_images"] = images

        # 3. Write to DB
        if updates:
            update_db(run_id, updates)
        else:
            log.info(f"  Nothing to update for run {run_id}")

    log.info("\nDone.")


if __name__ == "__main__":
    run()
