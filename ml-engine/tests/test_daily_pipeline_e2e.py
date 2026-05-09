"""
ml-engine/tests/test_daily_pipeline_e2e.py

End-to-end test for the full daily pipeline:
  DOWNLOADED → auto_label_clip (GDINO) → package_dataset → annotate_clips (YOLO) → ANNOTATED

Phases:
  1. FIXTURE     — find a DOWNLOADED clip with file on disk
  2. AUTO-LABEL  — GDINO frames → labels → Dataset(LABELED)
  3. PACKAGE     — train/val split → Dataset(PACKAGED)
  4. ANNOTATE    — YOLO specialist → annotated MP4 → Clip(ANNOTATED)
  5. TEARDOWN    — restore DB state; clean output files unless --keep

WARNING: annotate_clips deletes the raw video (production behavior).
         Run scraper-engine/tests/test_scrape_sample.py to get more clips.

Usage:
    cd ml-engine && python tests/test_daily_pipeline_e2e.py
    cd ml-engine && python tests/test_daily_pipeline_e2e.py --keep
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from db.models import Clip, ClipStatus, Dataset, DatasetStatus
from db.session import get_session

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("daily-pipeline-e2e")


# ── helpers ───────────────────────────────────────────────────────────────────

def _section(title: str) -> None:
    logger.info(f"── {title} {'─' * (50 - len(title))}")


# ── phase 1: fixture ──────────────────────────────────────────────────────────

def setup_fixture() -> tuple[int, Path, str | None, str | None]:
    """
    Find a real DOWNLOADED clip with its file on disk.
    Returns (clip_id, video_path, original_mp4_path, original_file_path).
    """
    with get_session() as session:
        clip = (
            session.query(Clip)
            .filter(Clip.status == ClipStatus.DOWNLOADED)
            .filter(Clip.file_path.isnot(None))
            .order_by(Clip.created_at.desc())
            .first()
        )
        if clip is None:
            logger.error(
                "No DOWNLOADED clips in DB. "
                "Run scraper-engine/tests/test_scrape_sample.py first."
            )
            sys.exit(1)

        video_path = Path(clip.file_path)
        if not video_path.exists():
            logger.error(
                f"Clip id={clip.id} has file_path={clip.file_path} but file is missing."
            )
            sys.exit(1)

        clip_id = clip.id
        original_mp4_path = clip.mp4_path
        original_file_path = clip.file_path
        clip_title = (clip.title or f"clip_{clip_id}")[:80]

    logger.info(f"  clip_id={clip_id}  file={video_path.name}")
    logger.info(f"  title: {clip_title}")
    return clip_id, video_path, original_mp4_path, original_file_path


# ── phase 2: auto-label (GDINO) ───────────────────────────────────────────────

def run_auto_label(clip_id: int) -> int:
    """
    Call auto_label_clip directly. Returns dataset_id.
    Requires GroundingDINO installed and checkpoint at settings.GDINO_CHECKPOINT.
    """
    gdino_ckpt = Path(settings.GDINO_CHECKPOINT)
    if not gdino_ckpt.exists():
        logger.error(
            f"GDINO checkpoint not found: {gdino_ckpt}\n"
            "Download groundingdino_swint_ogc.pth and place it at the repo root."
        )
        sys.exit(1)

    try:
        from tasks.auto_label import auto_label_clip
    except ImportError as e:
        logger.error(f"GroundingDINO not installed: {e}")
        sys.exit(1)

    result = auto_label_clip(clip_id=clip_id)
    logger.info(f"  auto_label_clip result: {result}")

    assert result["status"] == "labeled", f"Expected 'labeled', got: {result['status']}"
    assert result["frame_count"] > 0, "No frames extracted"

    dataset_id = result["dataset_id"]

    with get_session() as session:
        clip = session.get(Clip, clip_id)
        assert clip.status == ClipStatus.LABELED, \
            f"Clip status expected LABELED, got {clip.status}"

        dataset = session.get(Dataset, dataset_id)
        assert dataset is not None, "Dataset record not created"
        assert dataset.status == DatasetStatus.LABELED, \
            f"Dataset status expected LABELED, got {dataset.status}"

        dataset_dir = Path(dataset.yolo_dir_path)

    assert dataset_dir.exists(), f"Dataset dir missing: {dataset_dir}"
    label_count = sum(1 for f in (dataset_dir / "train" / "labels").glob("*.txt")
                      if f.stat().st_size > 0)

    logger.info(f"  dataset_id={dataset_id}  labeled_frames={result['labeled_frames']}"
                f"  labels_with_boxes={label_count}")
    return dataset_id


# ── phase 3: package dataset ──────────────────────────────────────────────────

def run_package(dataset_id: int) -> None:
    """Call package_dataset directly. Verifies train/val split."""
    from tasks.package_dataset import package_dataset

    result = package_dataset(dataset_id=dataset_id)
    logger.info(f"  package_dataset result: {result}")

    assert result["status"] in ("packaged", "skipped"), \
        f"Unexpected status: {result['status']}"

    with get_session() as session:
        dataset = session.get(Dataset, dataset_id)
        assert dataset.status == DatasetStatus.PACKAGED, \
            f"Dataset status expected PACKAGED, got {dataset.status}"
        dataset_dir = Path(dataset.yolo_dir_path)

    val_images = list((dataset_dir / "val" / "images").glob("*"))
    assert len(val_images) > 0, "val/images is empty after packaging"

    logger.info(f"  train={result['train_count']}  val={result['val_count']}")


# ── phase 4: annotate (YOLO) ──────────────────────────────────────────────────

def run_annotate(clip_id: int) -> Path | None:
    """
    Call annotate_clips directly (batch — processes all LABELED clips).
    Returns the annotated MP4 path if the fixture clip was accepted, else None.
    """
    from tasks.annotate_clips import annotate_clips

    result = annotate_clips()
    logger.info(f"  annotate_clips result: {result}")

    with get_session() as session:
        clip = session.get(Clip, clip_id)
        status = clip.status
        mp4_path = Path(clip.mp4_path) if clip.mp4_path else None

    if status == ClipStatus.ANNOTATED:
        assert mp4_path is not None, "mp4_path not set on ANNOTATED clip"
        assert mp4_path.exists(), f"Annotated MP4 missing: {mp4_path}"
        size_kb = mp4_path.stat().st_size // 1024
        assert size_kb > 0, "Annotated MP4 is empty"
        logger.info(f"  ACCEPTED — mp4={mp4_path.name}  size={size_kb}KB")
        return mp4_path
    elif status == ClipStatus.PENDING:
        logger.info("  REJECTED by annotate_clips (low detection rate or zero detections)")
        return None
    else:
        raise AssertionError(f"Unexpected clip status after annotate: {status}")


# ── phase 5: teardown ─────────────────────────────────────────────────────────

def teardown(
    clip_id: int,
    dataset_id: int | None,
    mp4_path: Path | None,
    original_mp4_path: str | None,
    original_file_path: str | None,
    keep: bool,
) -> None:
    with get_session() as session:
        clip = session.get(Clip, clip_id)
        if clip:
            # Raw file is gone (deleted by annotate_clips or auto_label failure).
            # Set to PENDING so it's not picked up as DOWNLOADED with no file.
            clip.status = ClipStatus.PENDING
            clip.file_path = None
            clip.mp4_path = original_mp4_path
            logger.info(f"  Restored Clip id={clip_id} → PENDING (raw file consumed)")

        if dataset_id is not None:
            dataset = session.get(Dataset, dataset_id)
            if dataset:
                dataset_dir = Path(dataset.yolo_dir_path)
                session.delete(dataset)
                if not keep and dataset_dir.exists():
                    shutil.rmtree(dataset_dir, ignore_errors=True)
                    logger.info(f"  Deleted dataset dir: {dataset_dir}")

    if not keep and mp4_path and mp4_path.exists():
        mp4_path.unlink()
        logger.info(f"  Deleted annotated MP4: {mp4_path}")
    elif keep and mp4_path:
        logger.info(f"  Kept annotated MP4: {mp4_path}  (--keep)")

    frames_dir = settings.FRAMES_DIR
    if not keep and frames_dir.exists():
        for d in frames_dir.iterdir():
            if d.is_dir():
                shutil.rmtree(d, ignore_errors=True)
        logger.info(f"  Cleared frames dir: {frames_dir}")

    logger.info("  Teardown complete.")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ml-engine daily pipeline E2E test")
    parser.add_argument("--keep", action="store_true",
                        help="Keep annotated MP4 and dataset dir after test")
    parser.add_argument("--skip-gdino", action="store_true",
                        help="Skip GDINO phase — use an existing LABELED clip instead")
    args = parser.parse_args()

    clip_id = None
    dataset_id = None
    mp4_path = None
    original_mp4_path = None
    original_file_path = None

    try:
        logger.info("═══ Daily Pipeline E2E Test ══════════════════════════")
        t0 = time.time()

        _section("FIXTURE")
        if args.skip_gdino:
            # Find a LABELED clip instead of DOWNLOADED
            with get_session() as session:
                clip = (
                    session.query(Clip)
                    .filter(Clip.status == ClipStatus.LABELED)
                    .filter(Clip.file_path.isnot(None))
                    .first()
                )
                if clip is None:
                    logger.error("No LABELED clips in DB. Run without --skip-gdino first.")
                    sys.exit(1)
                clip_id = clip.id
                original_mp4_path = clip.mp4_path
                original_file_path = clip.file_path
                logger.info(f"  Skipping GDINO — using LABELED clip_id={clip_id}")
        else:
            clip_id, _, original_mp4_path, original_file_path = setup_fixture()

        if not args.skip_gdino:
            _section("AUTO-LABEL (GDINO)")
            dataset_id = run_auto_label(clip_id)

            _section("PACKAGE DATASET")
            run_package(dataset_id)

        _section("ANNOTATE (YOLO)")
        mp4_path = run_annotate(clip_id)

        elapsed = time.time() - t0
        outcome = "ANNOTATED" if mp4_path else "REJECTED (no military content detected)"
        logger.info(f"═══ PIPELINE COMPLETE in {elapsed:.1f}s — {outcome} ══════════════")

    except SystemExit:
        raise
    except Exception as exc:
        logger.error(f"TEST FAILED: {exc}", exc_info=True)
        sys.exit(1)
    finally:
        if clip_id is not None:
            _section("TEARDOWN")
            teardown(
                clip_id=clip_id,
                dataset_id=dataset_id,
                mp4_path=mp4_path,
                original_mp4_path=original_mp4_path,
                original_file_path=original_file_path,
                keep=args.keep,
            )


if __name__ == "__main__":
    main()
