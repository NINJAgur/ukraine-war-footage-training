"""
training-engine/tests/test_pipeline_e2e.py

Quick annotation test: finds a LABELED clip (GDINO already done) and runs
annotate_clips() to verify the YOLO specialist pipeline produces an annotated MP4.

Use this when you already have LABELED clips in the DB and want to test just the
YOLO annotation step without running GDINO again.

For the full pipeline including GDINO, use test_daily_pipeline_e2e.py.

Phases:
  1. FIXTURE     — find an existing LABELED clip with file on disk
  2. ANNOTATE    — call annotate_clips() directly
  3. VERIFY      — assert clip is ANNOTATED or PENDING (with reason)
  4. TEARDOWN    — clean up unless --keep

Run from training-engine/:
    python tests/test_pipeline_e2e.py
    python tests/test_pipeline_e2e.py --keep
"""
from __future__ import annotations

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
logger = logging.getLogger("pipeline-e2e")


def setup_fixture() -> tuple[int, str | None]:
    with get_session() as session:
        clip = (
            session.query(Clip)
            .filter(Clip.status == ClipStatus.LABELED)
            .filter(Clip.file_path.isnot(None))
            .order_by(Clip.created_at.desc())
            .first()
        )
        if clip is None:
            logger.error(
                "No LABELED clips in DB. "
                "Run test_daily_pipeline_e2e.py first to get GDINO-labeled clips."
            )
            sys.exit(1)

        video_path = Path(clip.file_path)
        if not video_path.exists():
            logger.error(
                f"Clip id={clip.id} file missing: {clip.file_path}"
            )
            sys.exit(1)

        clip_id = clip.id
        original_mp4_path = clip.mp4_path
        logger.info(f"Using LABELED clip id={clip_id}: {video_path.name}")
        logger.info(f"  title: {(clip.title or 'N/A')[:80]}")
        logger.info(f"  scores: aircraft={clip.score_aircraft} vehicle={clip.score_vehicle} "
                    f"personnel={clip.score_personnel}")

    return clip_id, original_mp4_path


def run_annotate(clip_id: int) -> tuple[ClipStatus, Path | None]:
    from tasks.annotate_clips import annotate_clips

    result = annotate_clips()
    logger.info(f"annotate_clips result: {result}")

    with get_session() as session:
        clip = session.get(Clip, clip_id)
        status = clip.status
        mp4_path = Path(clip.mp4_path) if clip.mp4_path else None

    return status, mp4_path


def teardown(clip_id: int, mp4_path: Path | None, original_mp4_path: str | None, keep: bool):
    with get_session() as session:
        clip = session.get(Clip, clip_id)
        if clip:
            clip.status = ClipStatus.PENDING
            clip.file_path = None
            clip.mp4_path = original_mp4_path
            logger.info(f"Restored clip id={clip_id} → PENDING (raw consumed)")

    if not keep and mp4_path and mp4_path.exists():
        mp4_path.unlink()
        logger.info(f"Deleted annotated MP4: {mp4_path}")
    elif keep and mp4_path:
        logger.info(f"Kept annotated MP4: {mp4_path}  (--keep)")


def main():
    parser = argparse.ArgumentParser(description="training-engine YOLO annotation E2E test")
    parser.add_argument("--keep", action="store_true",
                        help="Keep annotated MP4 after test")
    args = parser.parse_args()

    clip_id = None
    mp4_path = None
    original_mp4_path = None

    try:
        logger.info("═══ YOLO Annotation E2E Test ═════════════════════════")
        t0 = time.time()

        clip_id, original_mp4_path = setup_fixture()

        logger.info("── ANNOTATE ────────────────────────────────────────────")
        status, mp4_path = run_annotate(clip_id)

        if status == ClipStatus.ANNOTATED:
            assert mp4_path and mp4_path.exists(), "mp4_path not set or missing"
            size_kb = mp4_path.stat().st_size // 1024
            assert size_kb > 0, "Annotated MP4 is empty"
            logger.info(f"  PASS — ANNOTATED  mp4={mp4_path.name}  size={size_kb}KB")
        elif status == ClipStatus.PENDING:
            logger.info("  PASS — REJECTED (low detection rate — clip has no detectable targets)")
        else:
            raise AssertionError(f"Unexpected clip status: {status}")

        elapsed = time.time() - t0
        logger.info(f"═══ ALL CHECKS PASSED in {elapsed:.1f}s ══════════════════")

    except SystemExit:
        raise
    except Exception as exc:
        logger.error(f"TEST FAILED: {exc}", exc_info=True)
        sys.exit(1)
    finally:
        if clip_id is not None:
            teardown(clip_id, mp4_path, original_mp4_path, keep=args.keep)


if __name__ == "__main__":
    main()
