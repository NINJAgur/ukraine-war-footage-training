"""
ml-engine/tests/test_pipeline_e2e.py

End-to-end integration test: scraper → ML pipeline, with full self-cleaning.

Phases:
  1. FIXTURE  — find an existing DOWNLOADED clip OR create a synthetic one
  2. AUTO-LABEL — call auto_label_clip() directly (not via Celery)
  3. PACKAGE  — call package_dataset() directly
  4. VERIFY   — assert frames, labels, data.yaml, DB statuses
  5. TEARDOWN — delete all DB rows + files created during the test

Run from ml-engine/:
    python tests/test_pipeline_e2e.py
    python tests/test_pipeline_e2e.py --skip-gdino   # synthetic labels, no GPU needed
    python tests/test_pipeline_e2e.py --keep          # skip teardown for manual inspection

Requirements: PostgreSQL + Redis running, .env present.
"""
import argparse
import hashlib
import logging
import shutil
import sys
import time
import yaml
from datetime import datetime
from pathlib import Path

# Allow imports from ml-engine root
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

from config import settings
from db.models import Clip, ClipSource, ClipStatus, Dataset, DatasetStatus
from db.session import get_session

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("e2e-test")

# ── helpers ──────────────────────────────────────────────────────────────────

TEST_URL = "https://e2e-test.local/synthetic/clip_001"


def _synthetic_video(path: Path, seconds: int = 3, fps: int = 10) -> None:
    """Write a minimal valid MP4 with random noise frames."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, fps, (320, 240))
    for _ in range(seconds * fps):
        frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        out.write(frame)
    out.release()
    logger.info(f"Synthetic video written: {path}")


def _synthetic_labels(labels_dir: Path, images_dir: Path) -> None:
    """Create stub YOLO .txt label files (one bbox per image)."""
    labels_dir.mkdir(parents=True, exist_ok=True)
    for img in images_dir.glob("*.jpg"):
        lbl = labels_dir / (img.stem + ".txt")
        lbl.write_text("0 0.5 0.5 0.3 0.3\n")  # class 0, centred bbox


# ── fixture ───────────────────────────────────────────────────────────────────

def setup_fixture(skip_gdino: bool) -> tuple[int, Path, bool]:
    """
    Return (clip_id, video_path, synthetic).
    If an existing DOWNLOADED clip is found, reuse it (synthetic=False).
    Otherwise create a synthetic clip row + video file (synthetic=True).
    """
    with get_session() as session:
        existing = (
            session.query(Clip)
            .filter(Clip.status == ClipStatus.DOWNLOADED, Clip.file_path.isnot(None))
            .order_by(Clip.created_at.desc())
            .first()
        )
        if existing and existing.file_path and Path(existing.file_path).exists():
            logger.info(f"Reusing existing DOWNLOADED clip id={existing.id}: {existing.file_path}")
            return existing.id, Path(existing.file_path), False

    # Create synthetic clip
    url_hash = hashlib.sha256(TEST_URL.encode()).hexdigest()
    video_path = settings.RAW_VIDEO_DIR / "synthetic" / f"{url_hash[:12]}_e2e_test.mp4"
    _synthetic_video(video_path)

    with get_session() as session:
        # Remove leftover from a previous failed test run
        old = session.query(Clip).filter(Clip.url_hash == url_hash).first()
        if old:
            session.delete(old)

        clip = Clip(
            url=TEST_URL,
            url_hash=url_hash,
            source=ClipSource.FUNKER530,
            title="E2E Test Clip",
            status=ClipStatus.DOWNLOADED,
            file_path=str(video_path),
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        session.add(clip)
        session.flush()
        clip_id = clip.id

    logger.info(f"Created synthetic Clip id={clip_id}: {video_path}")
    return clip_id, video_path, True


# ── pipeline steps ────────────────────────────────────────────────────────────

def run_auto_label(clip_id: int, skip_gdino: bool) -> int:
    """Call auto_label_clip directly. Returns dataset_id."""
    if skip_gdino:
        logger.info("--skip-gdino: injecting synthetic labels instead of running GroundingDINO")
        return _synthetic_auto_label(clip_id)

    from tasks.auto_label import auto_label_clip
    result = auto_label_clip(clip_id=clip_id)
    logger.info(f"auto_label_clip result: {result}")
    assert result["status"] == "labeled", f"Expected 'labeled', got {result['status']}"
    return result["dataset_id"]


def _synthetic_auto_label(clip_id: int) -> int:
    """Bypass GroundingDINO: extract frames with OpenCV, write stub labels, create Dataset."""
    from tasks.auto_label import extract_frames

    with get_session() as session:
        clip = session.get(Clip, clip_id)
        video_path = Path(clip.file_path)
        clip_hash = clip.url_hash[:12]
        clip_title = (clip.title or f"clip_{clip_id}")[:60]

    frames_dir = settings.FRAMES_DIR / clip_hash
    frame_count = extract_frames(video_path, frames_dir)
    assert frame_count > 0, "Frame extraction produced 0 frames"

    dataset_dir = settings.DATASETS_DIR / clip_hash
    train_images = dataset_dir / "train" / "images"
    train_labels = dataset_dir / "train" / "labels"
    train_images.mkdir(parents=True, exist_ok=True)
    train_labels.mkdir(parents=True, exist_ok=True)

    # Copy frames into dataset dir and write stub labels
    for frame in frames_dir.glob("*.jpg"):
        dst = train_images / frame.name
        shutil.copy2(frame, dst)
        (train_labels / (frame.stem + ".txt")).write_text("0 0.5 0.5 0.3 0.3\n")

    classes = [c.strip() for c in settings.GDINO_TEXT_PROMPT.split(",") if c.strip()]
    yaml_path = dataset_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(
            {"path": str(dataset_dir), "train": "train/images",
             "nc": len(classes), "names": classes},
            f, default_flow_style=False,
        )

    with get_session() as session:
        dataset = Dataset(
            name=f"{clip_hash[:8]}_{clip_title}",
            clip_id=clip_id,
            yolo_dir_path=str(dataset_dir),
            yaml_path=str(yaml_path),
            status=DatasetStatus.LABELED,
            frame_count=frame_count,
            class_count=len(classes),
        )
        session.add(dataset)
        session.flush()
        dataset_id = dataset.id
        clip = session.get(Clip, clip_id)
        clip.status = ClipStatus.LABELED

    logger.info(f"Synthetic auto-label done: dataset_id={dataset_id}  frames={frame_count}")
    return dataset_id


def run_package_dataset(dataset_id: int) -> None:
    """Call package_dataset directly (no render dispatch)."""
    from tasks.package_dataset import package_dataset

    # Monkey-patch render dispatch so the test stays self-contained
    import tasks.package_dataset as pkg_mod
    original_render = None
    try:
        from tasks.render_annotated import render_annotated_clip
        original_render = render_annotated_clip.delay
        render_annotated_clip.delay = lambda **_: None
    except Exception:
        pass

    result = package_dataset(dataset_id=dataset_id)
    logger.info(f"package_dataset result: {result}")
    assert result["status"] in ("packaged", "skipped"), f"Unexpected status: {result}"

    if original_render is not None:
        from tasks.render_annotated import render_annotated_clip
        render_annotated_clip.delay = original_render


# ── verify ────────────────────────────────────────────────────────────────────

def verify(clip_id: int, dataset_id: int) -> None:
    logger.info("── VERIFY ──────────────────────────────────────────")

    with get_session() as session:
        clip = session.get(Clip, clip_id)
        dataset = session.get(Dataset, dataset_id)

        assert clip is not None, "Clip not found in DB"
        assert dataset is not None, "Dataset not found in DB"
        assert clip.status in (ClipStatus.LABELED, ClipStatus.ANNOTATED), \
            f"Clip status expected LABELED/ANNOTATED, got {clip.status}"
        assert dataset.status == DatasetStatus.PACKAGED, \
            f"Dataset status expected PACKAGED, got {dataset.status}"

        dataset_dir = Path(dataset.yolo_dir_path)
        yaml_path = Path(dataset.yaml_path)

    assert dataset_dir.exists(), f"Dataset dir missing: {dataset_dir}"
    assert yaml_path.exists(), f"data.yaml missing: {yaml_path}"

    train_imgs = list((dataset_dir / "train" / "images").glob("*.jpg"))
    val_imgs   = list((dataset_dir / "val"   / "images").glob("*.jpg"))
    assert len(train_imgs) > 0, "No training images after packaging"
    assert len(val_imgs)   > 0, "No validation images after packaging"

    with open(yaml_path) as f:
        data_yaml = yaml.safe_load(f)
    assert "nc" in data_yaml and data_yaml["nc"] > 0, "data.yaml missing nc"
    assert "names" in data_yaml, "data.yaml missing names"

    logger.info(f"  Clip     status={clip.status.value}")
    logger.info(f"  Dataset  status={dataset.status.value}")
    logger.info(f"  train={len(train_imgs)}  val={len(val_imgs)}")
    logger.info(f"  nc={data_yaml['nc']}  names={data_yaml['names']}")
    logger.info("  ALL CHECKS PASSED ✓")


# ── teardown ──────────────────────────────────────────────────────────────────

def teardown(clip_id: int, dataset_id: int, synthetic: bool) -> None:
    logger.info("── TEARDOWN ────────────────────────────────────────")

    with get_session() as session:
        dataset = session.get(Dataset, dataset_id)
        if dataset:
            yolo_dir = Path(dataset.yolo_dir_path) if dataset.yolo_dir_path else None
            session.delete(dataset)
            logger.info(f"  Deleted Dataset id={dataset_id}")

        if synthetic:
            clip = session.get(Clip, clip_id)
            if clip:
                video_path = Path(clip.file_path) if clip.file_path else None
                session.delete(clip)
                logger.info(f"  Deleted synthetic Clip id={clip_id}")
            else:
                video_path = None
        else:
            clip = None
            video_path = None

    # Remove frames dir
    clip_hash = None
    if synthetic and video_path:
        clip_hash = hashlib.sha256(TEST_URL.encode()).hexdigest()[:12]
    elif dataset_id:
        with get_session() as session:
            pass  # already deleted; use yolo_dir to infer

    if yolo_dir and yolo_dir.exists():
        shutil.rmtree(yolo_dir, ignore_errors=True)
        logger.info(f"  Removed dataset dir: {yolo_dir}")

    if clip_hash:
        frames_dir = settings.FRAMES_DIR / clip_hash
        if frames_dir.exists():
            shutil.rmtree(frames_dir, ignore_errors=True)
            logger.info(f"  Removed frames dir: {frames_dir}")

    if synthetic and video_path and video_path.exists():
        video_path.unlink()
        logger.info(f"  Removed synthetic video: {video_path}")

    logger.info("  Teardown complete.")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ml-engine E2E pipeline test")
    parser.add_argument("--skip-gdino", action="store_true",
                        help="Skip GroundingDINO; use synthetic YOLO labels")
    parser.add_argument("--keep", action="store_true",
                        help="Skip teardown (leave DB rows and files for inspection)")
    args = parser.parse_args()

    clip_id = dataset_id = None
    synthetic = False

    try:
        logger.info("═══ Phase 2 E2E Pipeline Test ═══════════════════════")
        t0 = time.time()

        logger.info("── FIXTURE ─────────────────────────────────────────")
        clip_id, video_path, synthetic = setup_fixture(args.skip_gdino)

        logger.info("── AUTO-LABEL ───────────────────────────────────────")
        dataset_id = run_auto_label(clip_id, args.skip_gdino)

        logger.info("── PACKAGE DATASET ─────────────────────────────────")
        run_package_dataset(dataset_id)

        logger.info("── VERIFY ──────────────────────────────────────────")
        verify(clip_id, dataset_id)

        elapsed = time.time() - t0
        logger.info(f"═══ ALL TESTS PASSED in {elapsed:.1f}s ══════════════════")

    except Exception as exc:
        logger.error(f"TEST FAILED: {exc}", exc_info=True)
        sys.exit(1)
    finally:
        if not args.keep and clip_id is not None and dataset_id is not None:
            teardown(clip_id, dataset_id, synthetic)


if __name__ == "__main__":
    main()
