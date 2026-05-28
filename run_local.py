"""
run_local.py — Local pipeline orchestrator

Dispatches tasks to running Celery workers via Redis and polls DB for progress.

Prerequisites:
    1. PostgreSQL running (local or via: docker compose up -d postgres)
    2. Redis running (local or via: docker compose up -d redis)
    3. inference-engine worker (needed for gdino / annotate):
           cd inference-engine
           celery -A celery_app worker -Q pipeline --pool=solo --concurrency=1 --loglevel=info
    4. scraper worker (needed for scrape only):
           cd scraper-engine
           celery -A celery_app worker -Q default --loglevel=info --concurrency=4
    OR start everything at once:
           bash start_workers.sh

Commands:
    status                              show DB clip/dataset/training counts
    scrape [funker530|geoconfirmed|both]  trigger scrapers (default: both)
    gdino                               run GDINO auto-labeling batch
    annotate                            run YOLO annotation batch
    all [funker530|geoconfirmed|both]   full pipeline: scrape -> gdino -> annotate

Usage (from repo root):
    python run_local.py status
    python run_local.py scrape funker530
    python run_local.py gdino
    python run_local.py annotate
    python run_local.py all
"""
import os
import sys
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("run_local")

REPO_ROOT = Path(__file__).resolve().parent
TRAINING_ENGINE_DIR = REPO_ROOT / "training-engine"
INFERENCE_ENGINE_DIR = REPO_ROOT / "inference-engine"
SCRAPER_ENGINE_DIR = REPO_ROOT / "scraper-engine"

sys.path.insert(0, str(INFERENCE_ENGINE_DIR))
sys.path.insert(0, str(TRAINING_ENGINE_DIR))
sys.path.insert(0, str(REPO_ROOT))

from db.session import get_session
from shared.db.models import Clip, ClipStatus, Dataset, DatasetStatus, TrainingRun, TrainingStatus


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_celery(broker_url: str = None):
    from celery import Celery
    url = broker_url or os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
    return Celery(broker=url)


def _count_clips(status: ClipStatus) -> int:
    with get_session() as session:
        return session.query(Clip).filter(Clip.status == status).count()


def _count_datasets(status: DatasetStatus) -> int:
    with get_session() as session:
        return session.query(Dataset).filter(Dataset.status == status).count()


def _count_downloaded_without_dataset() -> int:
    from sqlalchemy import exists as sa_exists
    with get_session() as session:
        return (
            session.query(Clip)
            .filter(
                Clip.status == ClipStatus.DOWNLOADED,
                Clip.file_path.isnot(None),
                ~sa_exists().where(Dataset.clip_id == Clip.id),
            )
            .count()
        )


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_status(_args: list) -> None:
    with get_session() as session:
        clip_counts = {}
        for s in ClipStatus:
            n = session.query(Clip).filter(Clip.status == s).count()
            if n:
                clip_counts[s.value] = n

        ds_counts = {}
        for s in DatasetStatus:
            n = session.query(Dataset).filter(Dataset.status == s).count()
            if n:
                ds_counts[s.value] = n

        recent_runs = (
            session.query(TrainingRun)
            .order_by(TrainingRun.id.desc())
            .limit(6)
            .all()
        )

    log.info("=" * 55)
    log.info("DB STATUS")
    log.info("=" * 55)
    log.info("Clips:")
    for status, count in clip_counts.items():
        log.info(f"  {status:<16} {count}")
    if ds_counts:
        log.info("Datasets:")
        for status, count in ds_counts.items():
            log.info(f"  {status:<16} {count}")
    if recent_runs:
        log.info("Recent TrainingRuns:")
        for r in recent_runs:
            map50 = None
            if r.metrics:
                for k, v in r.metrics.items():
                    if "map50" in k.lower() and "map50-95" not in k.lower():
                        try:
                            map50 = round(float(v), 3)
                        except (ValueError, TypeError):
                            pass
            map50_str = f"  mAP50={map50}" if map50 else ""
            log.info(
                f"  run_id={r.id}  {r.model_type.value:<12} {r.stage.value:<9} "
                f"{r.status.value:<8}{map50_str}"
            )
    log.info("=" * 55)


def cmd_scrape(args: list) -> None:
    target = args[0] if args else "both"
    if target not in ("funker530", "geoconfirmed", "both"):
        log.error(f"Unknown scrape target: {target!r}  (use funker530 | geoconfirmed | both)")
        sys.exit(1)

    tasks = []
    if target in ("funker530", "both"):
        tasks.append(("tasks.scrape_funker530.scrape_funker530", "default"))
    if target in ("geoconfirmed", "both"):
        tasks.append(("tasks.scrape_geoconfirmed.scrape_geoconfirmed", "default"))

    app = _make_celery()
    initial_downloaded = _count_clips(ClipStatus.DOWNLOADED)
    log.info(f"DOWNLOADED clips before scrape: {initial_downloaded}")

    for task_name, queue in tasks:
        t = app.send_task(task_name, queue=queue)
        log.info(f"Dispatched {task_name}  task_id={t.id}")

    log.info("Polling for new DOWNLOADED clips (up to 5min)...")
    deadline = time.time() + 300
    while time.time() < deadline:
        time.sleep(8)
        current = _count_clips(ClipStatus.DOWNLOADED)
        if current > initial_downloaded:
            log.info(f"New clips DOWNLOADED: {current - initial_downloaded}  (total: {current})")
            break
    else:
        log.info("No new clips scraped within 5min — may be no new content in lookback window")


def cmd_gdino(_args: list) -> None:
    pending = _count_downloaded_without_dataset()
    if pending == 0:
        log.info("No DOWNLOADED clips without a dataset — skipping GDINO")
        return

    log.info(f"DOWNLOADED clips pending GDINO: {pending}")

    app = _make_celery()
    initial_packaged = _count_datasets(DatasetStatus.PACKAGED) + _count_datasets(DatasetStatus.TRAINED)
    t = app.send_task("tasks.auto_label.auto_label_batch", queue="pipeline")
    log.info(f"Dispatched auto_label_batch  task_id={t.id}")
    log.info("Polling for PACKAGED datasets (up to 30min)...")

    deadline = time.time() + 1800
    last_packaged = initial_packaged
    last_change = time.time()
    while time.time() < deadline:
        time.sleep(10)
        total_packaged = _count_datasets(DatasetStatus.PACKAGED) + _count_datasets(DatasetStatus.TRAINED)
        if total_packaged > last_packaged:
            log.info(f"Datasets PACKAGED: {total_packaged} (+{total_packaged - last_packaged})")
            last_packaged = total_packaged
            last_change = time.time()
        remaining = _count_downloaded_without_dataset()
        if remaining == 0 and last_packaged > initial_packaged:
            log.info("All DOWNLOADED clips have datasets — GDINO complete")
            break
        if last_packaged > initial_packaged and (time.time() - last_change) > 60:
            log.info("PACKAGED count stable for 60s — GDINO complete")
            break
    else:
        log.info("GDINO timed out after 30min — check inference-engine worker logs")

    cmd_status([])


def cmd_annotate(_args: list) -> None:
    pending = _count_clips(ClipStatus.DOWNLOADED)
    if pending == 0:
        log.info("No DOWNLOADED clips — nothing to annotate")
        return

    log.info(f"DOWNLOADED clips pending annotation: {pending}")

    app = _make_celery()
    initial_annotated = _count_clips(ClipStatus.ANNOTATED)
    t = app.send_task("tasks.annotate_clips.annotate_clips", queue="pipeline")
    log.info(f"Dispatched annotate_clips  task_id={t.id}")
    log.info("Polling for ANNOTATED clips (up to 60min)...")

    deadline = time.time() + 3600
    last_annotated = initial_annotated
    while time.time() < deadline:
        time.sleep(10)
        current_annotated = _count_clips(ClipStatus.ANNOTATED)
        if current_annotated > last_annotated:
            log.info(f"Clips ANNOTATED: {current_annotated} (+{current_annotated - last_annotated})")
            last_annotated = current_annotated
        remaining = _count_clips(ClipStatus.DOWNLOADED)
        if remaining == 0:
            log.info("No DOWNLOADED clips remaining — annotation complete")
            break
    else:
        log.info("Annotation timed out after 60min — check inference-engine worker logs")

    cmd_status([])


def cmd_all(args: list) -> None:
    log.info(">>> FULL PIPELINE: scrape -> gdino -> annotate")
    cmd_scrape(args)
    cmd_gdino([])
    cmd_annotate([])


# ── Entry point ───────────────────────────────────────────────────────────────

COMMANDS = {
    "status":   cmd_status,
    "scrape":   cmd_scrape,
    "gdino":    cmd_gdino,
    "annotate": cmd_annotate,
    "all":      cmd_all,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(
            "Usage: python run_local.py <command> [args]\n"
            "\n"
            "Commands:\n"
            "  status                               show DB clip/dataset/training counts\n"
            "  scrape [funker530|geoconfirmed|both]  trigger scrapers (default: both)\n"
            "  gdino                                run GDINO auto-labeling batch\n"
            "  annotate                             run YOLO annotation batch\n"
            "  all [funker530|geoconfirmed|both]    full pipeline: scrape -> gdino -> annotate\n"
            "\n"
            "Prerequisites: PostgreSQL + Redis running, inference-engine Celery worker (Q=pipeline)\n"
            "               scraper worker (Q=default) needed only for 'scrape'\n"
            "               Start all: bash start_workers.sh\n"
        )
        sys.exit(1)

    COMMANDS[sys.argv[1]](sys.argv[2:])
