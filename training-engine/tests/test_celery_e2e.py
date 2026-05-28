"""
training-engine/tests/test_celery_e2e.py

Celery E2E test: fires POST /api/admin/train via the real HTTP API,
waits for the Celery worker to execute the task, and verifies
DB transitions QUEUED → RUNNING → DONE with weights + metrics.

Requires:
  - Backend running on localhost:8000
  - Celery worker running: python -m celery -A celery_app worker -Q gpu --pool=solo --concurrency=1
  - Redis running

Run from training-engine/:
    python tests/test_celery_e2e.py --model-type VEHICLE --epochs 1 --keep
"""
import argparse
import logging
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from db.models import TrainingRun, TrainingStatus
from db.session import get_session

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("celery-e2e")

BASE_URL = "http://localhost:8000"
POLL_INTERVAL = 10   # seconds between DB polls
TIMEOUT = 3600       # max seconds to wait for DONE (1 epoch PERSONNEL ~30 min)


def get_token() -> str:
    resp = requests.post(f"{BASE_URL}/api/auth/login", json={"username": "admin", "password": "admin123"}, timeout=10)
    resp.raise_for_status()
    return resp.json()["access_token"]


def dispatch_run(token: str, model_type: str) -> "int | None":
    """
    Returns run_id on 202, None on 409 (already active).
    Raises on any other error status.
    """
    resp = requests.post(
        f"{BASE_URL}/api/admin/train",
        json={"model_type": model_type, "stage": "BASELINE"},
        headers={"Authorization": f"Bearer {token}"},
        timeout=10,
    )
    if resp.status_code == 409:
        logger.info("POST /api/admin/train returned 409 — a run is already active, skipping test")
        return None
    resp.raise_for_status()
    run_id = resp.json()["training_run_id"]
    logger.info(f"Dispatched run_id={run_id} via POST /api/admin/train")
    return run_id


def poll_until_done(run_id: int) -> TrainingRun:
    deadline = time.time() + TIMEOUT
    poll_n = 0
    while time.time() < deadline:
        poll_n += 1
        elapsed = int(time.time() - (deadline - TIMEOUT))
        with get_session() as s:
            run = s.query(TrainingRun).filter(TrainingRun.id == run_id).first()
            status = run.status if run else None

        logger.info(
            f"poll {poll_n}/{TIMEOUT // POLL_INTERVAL}: status={status.value if status else 'NOT FOUND'} elapsed={elapsed}s"
        )

        if status == TrainingStatus.DONE:
            with get_session() as s:
                run = s.query(TrainingRun).filter(TrainingRun.id == run_id).first()
                s.expunge(run)
            return run
        if status == TrainingStatus.ERROR:
            with get_session() as s:
                run = s.query(TrainingRun).filter(TrainingRun.id == run_id).first()
                error_msg = run.error_message if run else "unknown"
            raise RuntimeError(f"Run {run_id} failed: {error_msg}")

        time.sleep(POLL_INTERVAL)

    raise TimeoutError(f"Run {run_id} did not complete within {TIMEOUT}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", default="VEHICLE")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()

    # Patch epochs in config before worker picks up the task
    from config import settings
    original = settings.YOLO_EPOCHS_BASELINE
    settings.YOLO_EPOCHS_BASELINE = args.epochs

    run_id = None
    passed = False
    weights_path = None
    try:
        token = get_token()
        run_id = dispatch_run(token, args.model_type)

        if run_id is None:
            logger.info("SKIP: a training run is already active — re-run after it completes")
            sys.exit(0)

        run = poll_until_done(run_id)

        assert run.status == TrainingStatus.DONE, f"Expected DONE, got {run.status}"
        assert run.weights_path, "weights_path is empty"
        assert Path(run.weights_path).exists(), f"best.pt missing: {run.weights_path}"
        assert run.metrics, "metrics is empty"
        map50_key = next((k for k in run.metrics if "map50" in k.lower() and "map50-95" not in k.lower()), None)
        assert map50_key, f"No mAP50 key in metrics: {run.metrics}"
        map50 = float(run.metrics[map50_key])
        assert map50 > 0, f"mAP50={map50} — model did not learn"

        weights_path = run.weights_path
        logger.info(
            f"CELERY E2E PASSED — run_id={run_id} status=DONE mAP50={map50:.4f} weights={weights_path}"
        )
        passed = True

    except Exception as exc:
        logger.error(f"CELERY E2E FAILED: {exc}", exc_info=True)
        sys.exit(1)
    finally:
        settings.YOLO_EPOCHS_BASELINE = original
        # Always delete the DB row — test runs don't belong in production DB.
        # --keep only preserves the weights directory on disk (for inspection).
        if run_id is not None:
            with get_session() as s:
                run = s.query(TrainingRun).filter(TrainingRun.id == run_id).first()
                if run:
                    weights_path = weights_path or run.weights_path
                    s.delete(run)
            logger.info(f"Cleanup: deleted training_run id={run_id}")
            if not (args.keep and passed) and weights_path:
                import shutil
                weights_dir = Path(weights_path).parent.parent
                if weights_dir.exists():
                    shutil.rmtree(weights_dir, ignore_errors=True)
                    logger.info(f"Cleanup: deleted weights dir {weights_dir}")


if __name__ == "__main__":
    main()
