"""
build_scraped_dataset.py

Batch script: extract frames from downloaded videos and auto-label them with
GroundingDINO to produce a dated YOLO dataset.

Reads videos from:
    scraper-engine/media/funker530/<date>/
    scraper-engine/media/geoconfirmed/<date>/

Output:
    ml-engine/media/scraped_datasets/<YYYY-MM-DD>/
        train/images/   train/labels/
        val/images/     val/labels/
        dataset.yaml

Usage (from repo root):
    cd ml-engine && python scripts/build_scraped_dataset.py
    cd ml-engine && python scripts/build_scraped_dataset.py --date 2026-05-07
    cd ml-engine && python scripts/build_scraped_dataset.py --keep-frames
"""
import sys
import shutil
import logging
import argparse
import random
import tempfile
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("build_scraped_dataset")

REPO_ROOT = Path(__file__).resolve().parents[2]
ML_ENGINE_DIR = REPO_ROOT / "ml-engine"
SCRAPER_ENGINE_DIR = REPO_ROOT / "scraper-engine"

sys.path.insert(0, str(ML_ENGINE_DIR))
sys.path.insert(0, str(REPO_ROOT))

from config import settings

VIDEO_DIRS = [
    SCRAPER_ENGINE_DIR / "media" / "funker530",
    SCRAPER_ENGINE_DIR / "media" / "geoconfirmed",
]
VID_EXTS = {".mp4", ".webm", ".mkv", ".avi", ".mov"}
IMG_EXTS = {".jpg", ".jpeg", ".png"}

# GDINO term index → canonical class ID (mirrors config.py GDINO_CLASS_TO_MODEL)
_GDINO_TO_CANONICAL = {
    0: 0, 1: 0, 2: 0, 3: 0, 4: 0,       # aircraft
    5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, # vehicle
    11: 2, 12: 2, 13: 2, 14: 2,           # personnel
}

VAL_SPLIT = 0.2
SPLIT_SEED = 42


def _find_videos() -> list[Path]:
    """Recursively find all video files under the scraper media dirs."""
    videos = []
    for base in VIDEO_DIRS:
        if base.exists():
            videos.extend(f for f in base.rglob("*") if f.suffix.lower() in VID_EXTS)
    return sorted(videos)


def _extract_frames(video_path: Path, frames_dir: Path) -> int:
    import cv2
    frames_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.warning(f"  Cannot open video: {video_path.name}")
        return 0
    frame_idx = saved = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % settings.FRAME_INTERVAL == 0:
            cv2.imwrite(str(frames_dir / f"frame_{frame_idx:06d}.jpg"), frame)
            saved += 1
            if saved >= settings.MAX_FRAMES_PER_CLIP:
                break
        frame_idx += 1
    cap.release()
    return saved


def _run_gdino(frames_dir: Path, dst_img_dir: Path, dst_lbl_dir: Path) -> int:
    """Run GDINO on frames_dir, remap class IDs, copy labeled images to dst dirs."""
    from core.autolabeling.auto_label import create_yolo_dataset

    tmp_dir = Path(tempfile.mkdtemp(prefix="gdino_"))
    try:
        create_yolo_dataset(
            input_folder=str(frames_dir),
            text_prompt=settings.GDINO_TEXT_PROMPT,
            output_path=str(tmp_dir),
            config_path=settings.GDINO_CONFIG,
            checkpoint_path=settings.GDINO_CHECKPOINT,
            box_threshold=settings.GDINO_BOX_THRESHOLD,
            text_threshold=settings.GDINO_TEXT_THRESHOLD,
        )

        labeled = 0
        for img_src in (tmp_dir / "train" / "images").glob("*"):
            if img_src.suffix.lower() not in IMG_EXTS:
                continue
            lbl_src = tmp_dir / "train" / "labels" / (img_src.stem + ".txt")
            if not lbl_src.exists():
                continue

            lines_out = []
            for line in lbl_src.read_text().splitlines():
                parts = line.strip().split()
                if not parts:
                    continue
                cid = _GDINO_TO_CANONICAL.get(int(parts[0]), -1)
                if cid >= 0:
                    lines_out.append(f"{cid} {' '.join(parts[1:])}")

            if not lines_out:
                continue

            shutil.copy2(img_src, dst_img_dir / img_src.name)
            (dst_lbl_dir / (img_src.stem + ".txt")).write_text("\n".join(lines_out) + "\n")
            labeled += 1

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return labeled


def _create_val_split(
    train_img: Path, train_lbl: Path, val_img: Path, val_lbl: Path
) -> tuple[int, int]:
    images = sorted(f for f in train_img.iterdir() if f.suffix.lower() in IMG_EXTS)
    random.seed(SPLIT_SEED)
    random.shuffle(images)
    n_val = max(1, int(len(images) * VAL_SPLIT))
    for img in images[:n_val]:
        lbl = train_lbl / (img.stem + ".txt")
        shutil.move(str(img), val_img / img.name)
        if lbl.exists():
            shutil.move(str(lbl), val_lbl / lbl.name)
    return len(images) - n_val, n_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--date", default=None,
        help="Output folder name (YYYY-MM-DD). Default: today UTC."
    )
    parser.add_argument(
        "--keep-frames", action="store_true",
        help="Keep frames dirs after processing (default: delete)."
    )
    args = parser.parse_args()

    date_str = args.date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_dir  = settings.DATASETS_DIR / date_str
    train_img = out_dir / "train" / "images"
    train_lbl = out_dir / "train" / "labels"
    val_img   = out_dir / "val"   / "images"
    val_lbl   = out_dir / "val"   / "labels"
    for d in (train_img, train_lbl, val_img, val_lbl):
        d.mkdir(parents=True, exist_ok=True)

    videos = _find_videos()
    log.info(f"Found {len(videos)} video(s) to process")
    if not videos:
        log.warning("No videos found. Run scraping first.")
        return

    total_frames = total_labeled = 0
    frames_dirs: list[Path] = []

    for video in videos:
        stem = video.stem[:20]
        frames_dir = settings.FRAMES_DIR / stem
        frames_dirs.append(frames_dir)

        log.info(f"[{video.parent.name}/{video.name}] Extracting frames...")
        n_frames = _extract_frames(video, frames_dir)
        if n_frames == 0:
            continue
        total_frames += n_frames
        log.info(f"  Extracted {n_frames} frames — running GDINO...")

        n_labeled = _run_gdino(frames_dir, train_img, train_lbl)
        log.info(f"  Labeled: {n_labeled}/{n_frames} frames")
        total_labeled += n_labeled

    if total_labeled == 0:
        log.warning("No frames were labeled — check GDINO model files.")
        return

    train_n, val_n = _create_val_split(train_img, train_lbl, val_img, val_lbl)

    yaml_path = out_dir / "dataset.yaml"
    yaml_path.write_text(
        f"path: {out_dir.as_posix()}\n"
        f"train: train/images\n"
        f"val: val/images\n"
        f"nc: 3\n"
        f"names: ['aircraft', 'vehicle', 'personnel']\n"
    )

    if not args.keep_frames:
        for fd in frames_dirs:
            if fd.exists():
                shutil.rmtree(fd)
        log.info(f"Deleted {len(frames_dirs)} frames dir(s)")

    log.info(
        f"=== DONE ===\n"
        f"  output       : {out_dir}\n"
        f"  videos       : {len(videos)}\n"
        f"  frames       : {total_frames}\n"
        f"  labeled      : {total_labeled}\n"
        f"  train / val  : {train_n} / {val_n}\n"
        f"  dataset.yaml : {yaml_path}"
    )


if __name__ == "__main__":
    main()
