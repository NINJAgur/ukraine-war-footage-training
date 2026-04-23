"""
ml-engine/tasks/auto_label.py

Celery task: extract frames from a downloaded Clip, run GroundingDINO
zero-shot auto-labeling, create a Dataset record, dispatch package_dataset.

Pipeline:
  Clip(DOWNLOADED) → frames → GroundingDINO → .txt labels → Dataset(LABELED)
                                                                    ↓
                                                        dispatch package_dataset
"""
import logging
import sys
from pathlib import Path

import cv2

from celery_app import celery_app
from config import settings
from db.models import Clip, ClipStatus, Dataset, DatasetStatus

_RUNNABLE_STATUSES = {ClipStatus.DOWNLOADED, ClipStatus.QUEUED}
from db.session import get_session

sys.path.insert(0, str(Path(__file__).parent.parent / "core"))

logger = logging.getLogger(__name__)

CLASSES = [c.strip() for c in settings.GDINO_TEXT_PROMPT.split(",") if c.strip()]


def extract_frames(video_path: Path, output_dir: Path) -> int:
    """Extract one frame every FRAME_INTERVAL frames, up to MAX_FRAMES_PER_CLIP."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    frame_idx = saved = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % settings.FRAME_INTERVAL == 0:
            cv2.imwrite(str(output_dir / f"frame_{frame_idx:06d}.jpg"), frame)
            saved += 1
            if saved >= settings.MAX_FRAMES_PER_CLIP:
                break
        frame_idx += 1

    cap.release()
    logger.info(
        f"Extracted {saved} frames from {video_path.name} "
        f"(every {settings.FRAME_INTERVAL} frames, max {settings.MAX_FRAMES_PER_CLIP})"
    )
    return saved


@celery_app.task(
    bind=True,
    name="tasks.auto_label.auto_label_clip",
    queue="gpu",
    autoretry_for=(Exception,),
    max_retries=2,
    default_retry_delay=120,
)
def auto_label_clip(self, clip_id: int) -> dict:
    """
    Extract frames from a downloaded clip and run GroundingDINO auto-labeling.
    Creates a Dataset record (LABELED) and dispatches package_dataset on completion.
    Idempotent: re-uses existing frames dir if already extracted.
    """
    logger.info(f"[{self.request.id}] auto_label_clip clip_id={clip_id}")

    with get_session() as session:
        clip = session.get(Clip, clip_id)
        if clip is None:
            raise ValueError(f"Clip {clip_id} not found")
        if clip.status not in _RUNNABLE_STATUSES:
            logger.info(
                f"[{self.request.id}] Clip {clip_id} status={clip.status} — skipping "
                f"(already processed or in wrong state)"
            )
            return {"status": "skipped", "clip_id": clip_id, "reason": clip.status}
        if not clip.file_path or not Path(clip.file_path).exists():
            raise ValueError(f"Clip {clip_id} has no file on disk: {clip.file_path}")
        video_path = Path(clip.file_path)
        clip_hash = clip.url_hash[:12]
        clip_title = (clip.title or f"clip_{clip_id}")[:60]

    # ── Extract frames ────────────────────────────────────────────────
    frames_dir = settings.FRAMES_DIR / clip_hash
    if not any(frames_dir.glob("*.jpg")):
        frame_count = extract_frames(video_path, frames_dir)
    else:
        frame_count = len(list(frames_dir.glob("*.jpg")))
        logger.info(f"Reusing {frame_count} existing frames in {frames_dir}")

    if frame_count == 0:
        raise ValueError(f"No frames extracted from {video_path}")

    # ── Run GroundingDINO ─────────────────────────────────────────────
    from autolabeling.auto_label import create_yolo_dataset

    dataset_dir = settings.DATASETS_DIR / clip_hash
    create_yolo_dataset(
        input_folder=str(frames_dir),
        text_prompt=settings.GDINO_TEXT_PROMPT,
        output_path=str(dataset_dir),
        config_path=settings.GDINO_CONFIG,
        checkpoint_path=settings.GDINO_CHECKPOINT,
        box_threshold=settings.GDINO_BOX_THRESHOLD,
        text_threshold=settings.GDINO_TEXT_THRESHOLD,
    )

    labels_dir = dataset_dir / "train" / "labels"
    labeled_frames = sum(
        1 for f in labels_dir.glob("*.txt") if f.stat().st_size > 0
    )
    yaml_path = dataset_dir / "data.yaml"

    logger.info(
        f"[{self.request.id}] GroundingDINO: {labeled_frames}/{frame_count} "
        f"frames with detections"
    )

    # ── Tag which model types appear in the labels ────────────────────
    detected_types: set[str] = set()
    for lbl in labels_dir.glob("*.txt"):
        try:
            for line in lbl.read_text().splitlines():
                parts = line.split()
                if parts:
                    cls_id = int(parts[0])
                    mt = settings.GDINO_CLASS_TO_MODEL.get(cls_id)
                    if mt:
                        detected_types.add(mt)
        except Exception:
            pass
    detected_model_types_list = sorted(detected_types)
    logger.info(f"[{self.request.id}] Detected model types: {detected_model_types_list}")

    # ── Create Dataset record ─────────────────────────────────────────
    with get_session() as session:
        dataset = Dataset(
            name=f"{clip_hash[:8]}_{clip_title}",
            clip_id=clip_id,
            yolo_dir_path=str(dataset_dir),
            yaml_path=str(yaml_path),
            status=DatasetStatus.LABELED,
            frame_count=frame_count,
            class_count=len(CLASSES),
            detected_model_types=detected_model_types_list,
        )
        session.add(dataset)
        session.flush()
        dataset_id = dataset.id

        clip = session.get(Clip, clip_id)
        clip.status = ClipStatus.LABELED

    logger.info(
        f"[{self.request.id}] Dataset id={dataset_id} created  clip_id={clip_id} → LABELED"
    )

    from tasks.package_dataset import package_dataset
    package_dataset.delay(dataset_id=dataset_id)

    return {
        "status": "labeled",
        "clip_id": clip_id,
        "dataset_id": dataset_id,
        "frame_count": frame_count,
        "labeled_frames": labeled_frames,
    }
