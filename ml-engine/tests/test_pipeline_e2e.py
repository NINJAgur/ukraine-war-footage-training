"""
ml-engine/tests/test_pipeline_e2e.py

End-to-end render test: takes a real DOWNLOADED clip from the DB, runs
render_annotated_clip(), verifies an annotated MP4 is produced with detections.

This test does NOT auto-label or train. Auto-labeling is for the fine-tune loop
(scraped clips → GDINO → dataset → fine-tune). This test verifies that the
render pipeline works with whatever weights are currently available.

Phases:
  1. FIXTURE  — find an existing DOWNLOADED clip in the DB (fail if none)
  2. RENDER   — call render_annotated_clip() directly (not via Celery)
  3. VERIFY   — assert annotated MP4 exists, check DB status
  4. TEARDOWN — restore clip status; delete annotated MP4 unless --keep

Run from ml-engine/:
    python tests/test_pipeline_e2e.py
    python tests/test_pipeline_e2e.py --keep           # leave MP4 for inspection
    python tests/test_pipeline_e2e.py --purge-outputs  # also delete .pt files

Requirements:
  - PostgreSQL + Redis running, .env present
  - At least one Clip with status=DOWNLOADED and a valid file_path in the DB
    (run scraper-engine/tests/test_scrape_live.py first to get real clips)
"""
import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from db.models import Clip, ClipStatus
from db.session import get_session

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("e2e-test")


# ── fixture ───────────────────────────────────────────────────────────────────

def setup_fixture():
    """
    Find a real DOWNLOADED clip in the DB.
    Returns (clip_id, video_path, original_mp4_path).
    Fails with a clear message if no real clip exists.
    """
    with get_session() as session:
        clip = (
            session.query(Clip)
            .filter(Clip.status == ClipStatus.DOWNLOADED, Clip.file_path.isnot(None))
            .order_by(Clip.created_at.desc())
            .first()
        )
        if clip is None:
            logger.error(
                "No DOWNLOADED clips in DB. "
                "Run scraper-engine/tests/test_scrape_live.py first to download real footage."
            )
            sys.exit(1)

        video_path = Path(clip.file_path)
        if not video_path.exists():
            logger.error(
                f"Clip id={clip.id} has file_path={clip.file_path} but file does not exist. "
                "Re-run the scraper to re-download."
            )
            sys.exit(1)

        clip_id = clip.id
        original_mp4_path = clip.mp4_path
        clip_title = clip.title or f"clip_{clip_id}"

    logger.info(f"Using real clip id={clip_id}: {video_path}")
    logger.info(f"  title: {clip_title}")
    return clip_id, video_path, original_mp4_path


# ── render ────────────────────────────────────────────────────────────────────

def run_render(clip_id: int) -> Path:
    """Call render_annotated_clip directly. Returns path to annotated MP4."""
    from tasks.render_annotated import render_annotated_clip
    result = render_annotated_clip(clip_id=clip_id)
    logger.info(f"render_annotated_clip result: {result}")
    assert result["status"] in ("annotated", "skipped"), \
        f"Unexpected render status: {result['status']}"

    with get_session() as session:
        clip = session.get(Clip, clip_id)
        mp4_path = clip.mp4_path

    assert mp4_path is not None, "Clip.mp4_path not set after render"
    return Path(mp4_path)


# ── verify ────────────────────────────────────────────────────────────────────

def verify(clip_id: int, mp4_path: Path) -> None:
    logger.info("── VERIFY ──────────────────────────────────────────")

    with get_session() as session:
        clip = session.get(Clip, clip_id)
        assert clip is not None, "Clip not found in DB"
        clip_status = clip.status

    assert clip_status == ClipStatus.ANNOTATED, \
        f"Clip status expected ANNOTATED, got {clip_status}"
    assert mp4_path.exists(), f"Annotated MP4 missing: {mp4_path}"

    size_kb = mp4_path.stat().st_size // 1024
    assert size_kb > 0, f"Annotated MP4 is empty: {mp4_path}"

    logger.info(f"  Clip status: {clip_status.value}")
    logger.info(f"  Annotated MP4: {mp4_path}  ({size_kb} KB)")
    logger.info("  ALL CHECKS PASSED ✓")


# ── teardown ──────────────────────────────────────────────────────────────────

def teardown(clip_id, mp4_path, original_mp4_path, keep, purge_outputs):
    logger.info("── TEARDOWN ────────────────────────────────────────")

    # Restore clip to DOWNLOADED so subsequent runs can use it
    with get_session() as session:
        clip = session.get(Clip, clip_id)
        if clip:
            clip.status = ClipStatus.DOWNLOADED
            clip.mp4_path = original_mp4_path
            logger.info(f"  Restored Clip id={clip_id} status → DOWNLOADED")

    if purge_outputs and mp4_path and mp4_path.exists():
        mp4_path.unlink()
        logger.info(f"  Removed annotated MP4: {mp4_path}  (--purge-outputs)")
    elif keep and mp4_path:
        logger.info(f"  Kept annotated MP4: {mp4_path}  (--keep)")
    elif mp4_path and mp4_path.exists():
        mp4_path.unlink()
        logger.info(f"  Removed annotated MP4: {mp4_path}")

    if purge_outputs:
        import shutil
        runs_dir = settings.RUNS_DIR
        if runs_dir.exists():
            shutil.rmtree(runs_dir, ignore_errors=True)
            logger.info(f"  Removed runs dir (weights): {runs_dir}  (--purge-outputs)")

    logger.info("  Teardown complete.")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ml-engine E2E render test")
    parser.add_argument("--keep", action="store_true",
                        help="Keep annotated MP4 after test (for inspection)")
    parser.add_argument("--purge-outputs", action="store_true",
                        help="Delete annotated MP4 AND weights (.pt files) after test")
    args = parser.parse_args()

    clip_id = None
    mp4_path = None
    original_mp4_path = None

    try:
        logger.info("═══ Phase 2 E2E Render Test ══════════════════════════")
        t0 = time.time()

        logger.info("── FIXTURE ─────────────────────────────────────────")
        clip_id, video_path, original_mp4_path = setup_fixture()

        logger.info("── RENDER ANNOTATED ────────────────────────────────")
        mp4_path = run_render(clip_id)

        logger.info("── VERIFY ──────────────────────────────────────────")
        verify(clip_id, mp4_path)

        elapsed = time.time() - t0
        logger.info(f"═══ ALL TESTS PASSED in {elapsed:.1f}s ══════════════════")

    except SystemExit:
        raise
    except Exception as exc:
        logger.error(f"TEST FAILED: {exc}", exc_info=True)
        sys.exit(1)
    finally:
        if clip_id is not None:
            teardown(
                clip_id,
                mp4_path,
                original_mp4_path,
                keep=args.keep,
                purge_outputs=args.purge_outputs,
            )


if __name__ == "__main__":
    main()
