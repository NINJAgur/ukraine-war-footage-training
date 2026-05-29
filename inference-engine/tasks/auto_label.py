"""
inference-engine/tasks/auto_label.py

Celery task: extract frames from a downloaded Clip, run GroundingDINO
zero-shot auto-labeling, create a Dataset record, dispatch package_dataset.

Pipeline:
  Clip(DOWNLOADED) → frames → GroundingDINO → .txt labels → Dataset(LABELED)
                                                                    ↓
                                                        dispatch package_dataset
"""
import logging
import shutil
import sys
from pathlib import Path
from typing import Dict

import cv2
import yaml as _yaml

from celery_app import celery_app
from config import settings
from db.models import Clip, ClipStatus, Dataset, DatasetStatus

_RUNNABLE_STATUSES = {ClipStatus.DOWNLOADED}
from db.session import get_session

_PROJECT_DIR = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(Path(__file__).parent.parent / "core"))

logger = logging.getLogger(__name__)


def _download_from_gcs(gs_url: str) -> Path:
    import tempfile
    from google.cloud import storage as gcs
    without_scheme = gs_url[len("gs://"):]
    bucket_name, _, blob_name = without_scheme.partition("/")
    suffix = Path(blob_name).suffix or ".mp4"
    tmp = Path(tempfile.mktemp(suffix=suffix, dir="/tmp"))
    client = gcs.Client()
    client.bucket(bucket_name).blob(blob_name).download_to_filename(str(tmp))
    logger.info(f"Downloaded from GCS: {gs_url} → {tmp}")
    return tmp


def _resolve_clip_path(raw: str) -> Path:
    if raw.startswith("gs://"):
        return _download_from_gcs(raw)
    p = Path(raw)
    if p.exists():
        return p
    normalized = raw.replace("\\", "/")
    marker = "scraper-engine/media/"
    if marker in normalized:
        rel = normalized[normalized.index(marker):]
        return _PROJECT_DIR / rel
    return p

CLASSES = [c.strip() for c in settings.GDINO_TEXT_PROMPT.replace(",", ".").split(".") if c.strip()]

# GDINO prompt-term index → canonical class ID (0=aircraft, 1=vehicle, 2=personnel)
_MT_TO_CANONICAL: Dict[str, int] = {"AIRCRAFT": 0, "VEHICLE": 1, "PERSONNEL": 2}
_GDINO_TO_CANONICAL: Dict[int, int] = {
    term_idx: _MT_TO_CANONICAL[mt]
    for term_idx, mt in settings.GDINO_CLASS_TO_MODEL.items()
}


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
    queue="pipeline",
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
                f"[{self.request.id}] clip_id={clip_id}  -> SKIP: status={clip.status.value}"
            )
            return None
        existing_ds = session.query(Dataset).filter_by(clip_id=clip_id).first()
        if existing_ds:
            logger.info(
                f"[{self.request.id}] clip_id={clip_id}  -> SKIP: dataset_id={existing_ds.id} already exists"
            )
            return None
        if not clip.file_path:
            raise ValueError(f"Clip {clip_id} has no file_path in DB")
        is_gcs = (clip.file_path or "").startswith("gs://")
        video_path = _resolve_clip_path(clip.file_path)
        if not video_path.exists():
            raise ValueError(f"Clip {clip_id} has no file on disk: {clip.file_path}")
        clip_hash = clip.url_hash[:12]
        clip_title = (clip.title or f"clip_{clip_id}")[:60]
        source = "funker530" if "funker530" in (clip.file_path or "") else "geoconfirmed"
        logger.info(
            f"[{self.request.id}] clip_id={clip_id}  source={source}  hash={clip_hash}  "
            f"aircraft={clip.score_aircraft:.2f}  vehicle={clip.score_vehicle:.2f}  personnel={clip.score_personnel:.2f}\n"
            f"    title: {clip_title}"
        )

    # ── Extract frames ────────────────────────────────────────────────
    frames_dir = settings.FRAMES_DIR / clip_hash
    if not any(frames_dir.glob("*.jpg")):
        frame_count = extract_frames(video_path, frames_dir)
    else:
        frame_count = len(list(frames_dir.glob("*.jpg")))
        logger.info(f"Reusing {frame_count} existing frames in {frames_dir}")

    if frame_count == 0:
        raise ValueError(f"No frames extracted from {video_path}")

    if is_gcs and video_path.exists():
        video_path.unlink()
        logger.info(f"Deleted temp GCS download: {video_path}")

    # ── Run GroundingDINO ─────────────────────────────────────────────
    from core.autolabeling.auto_label import create_yolo_dataset

    dataset_dir = settings.DATASETS_DIR / clip_hash
    create_yolo_dataset(
        input_folder=str(frames_dir),
        text_prompt=settings.GDINO_TEXT_PROMPT,
        output_path=str(dataset_dir),
        config_path=settings.GDINO_CONFIG,
        checkpoint_path=settings.GDINO_CHECKPOINT,
        box_threshold=settings.GDINO_BOX_THRESHOLD,
        text_threshold=settings.GDINO_TEXT_THRESHOLD,
        device=settings.GPU_DEVICE,
    )
    shutil.rmtree(frames_dir, ignore_errors=True)
    logger.info(f"[{self.request.id}] Deleted frames scratch dir {frames_dir}")

    labels_dir = dataset_dir / "train" / "labels"
    yaml_path = dataset_dir / "data.yaml"

    # ── Remap GDINO term indices → canonical 3-class IDs ─────────────
    for lbl_path in labels_dir.glob("*.txt"):
        lines_out = []
        try:
            for line in lbl_path.read_text().splitlines():
                parts = line.strip().split()
                if not parts:
                    continue
                canonical_id = _GDINO_TO_CANONICAL.get(int(parts[0]), -1)
                if canonical_id >= 0:
                    lines_out.append(f"{canonical_id} {' '.join(parts[1:])}")
        except Exception:
            pass
        lbl_path.write_text("\n".join(lines_out) + ("\n" if lines_out else ""))

    # ── Remove frames with empty labels after remapping ───────────────
    images_dir = dataset_dir / "train" / "images"
    removed = 0
    class_box_counts: dict[int, int] = {0: 0, 1: 0, 2: 0}
    for lbl_path in list(labels_dir.glob("*.txt")):
        if lbl_path.stat().st_size == 0:
            img_path = images_dir / (lbl_path.stem + ".jpg")
            lbl_path.unlink()
            if img_path.exists():
                img_path.unlink()
            removed += 1
        else:
            for line in lbl_path.read_text().splitlines():
                parts = line.split()
                if parts:
                    cid = int(parts[0])
                    if cid in class_box_counts:
                        class_box_counts[cid] += 1
    if removed:
        logger.info(f"[{self.request.id}] Removed {removed} empty-label frames from dataset")

    with open(yaml_path, "w") as _f:
        _yaml.dump(
            {
                "path": str(dataset_dir),
                "train": "train/images",
                "val": "train/images",
                "nc": 3,
                "names": ["aircraft", "vehicle", "personnel"],
            },
            _f,
            default_flow_style=False,
        )

    labeled_frames = sum(
        1 for f in labels_dir.glob("*.txt") if f.stat().st_size > 0
    )

    logger.info(
        f"[{self.request.id}] GDINO: labeled={labeled_frames}/{frame_count} frames  "
        f"aircraft={class_box_counts[0]}  vehicle={class_box_counts[1]}  personnel={class_box_counts[2]} boxes"
    )

    # ── Tag which model types appear in the labels ────────────────────
    # Labels are now canonical IDs: 0=AIRCRAFT, 1=VEHICLE, 2=PERSONNEL
    _CANONICAL_TO_MT = {0: "AIRCRAFT", 1: "VEHICLE", 2: "PERSONNEL"}
    detected_types: set[str] = set()
    for lbl in labels_dir.glob("*.txt"):
        try:
            for line in lbl.read_text().splitlines():
                parts = line.split()
                if parts:
                    mt = _CANONICAL_TO_MT.get(int(parts[0]))
                    if mt:
                        detected_types.add(mt)
        except Exception:
            pass
    detected_model_types_list = sorted(detected_types)
    logger.info(f"[{self.request.id}] detected_model_types={detected_model_types_list}")

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

    logger.info(
        f"[{self.request.id}] clip_id={clip_id}  -> LABELED: dataset_id={dataset_id}  "
        f"types={detected_model_types_list}  (clip stays DOWNLOADED)"
    )

    # Return dataset_id for the Celery chain: auto_label_clip | package_dataset
    return dataset_id


_BATCH_SIZE = 10


@celery_app.task(
    bind=True,
    name="tasks.auto_label.auto_label_batch",
    queue="pipeline",
    max_retries=0,
)
def auto_label_batch(self) -> dict:
    """
    Find DOWNLOADED clips without a Dataset and build a Celery chord:
      group(auto_label_clip.s(id) | package_dataset.s() for each clip)
      | trigger_finetune_check.si()

    The chord fires trigger_finetune_check exactly once after ALL clips are
    GDINO-labeled and packaged into merged/<MODEL>/ dirs. Clips stay DOWNLOADED
    so annotate_clips can run YOLO on them separately.
    """
    with get_session() as session:
        from sqlalchemy import exists as sa_exists
        clips = (
            session.query(Clip)
            .filter(Clip.status == ClipStatus.DOWNLOADED)
            .filter(Clip.file_path.isnot(None))
            .filter(~sa_exists().where(Dataset.clip_id == Clip.id))
            .order_by(Clip.created_at.asc())
            .limit(_BATCH_SIZE)
            .all()
        )
        clip_snapshot = [
            (c.id, c.title or f"clip_{c.id}", c.score_aircraft, c.score_vehicle, c.score_personnel)
            for c in clips
        ]

    clip_ids = [c[0] for c in clip_snapshot]
    logger.info(f"[{self.request.id}] auto_label_batch: {len(clip_ids)} clips to process")
    for clip_id, title, sc_a, sc_v, sc_p in clip_snapshot:
        logger.info(
            f"[{self.request.id}]   clip_id={clip_id}  "
            f"aircraft={sc_a:.2f}  vehicle={sc_v:.2f}  personnel={sc_p:.2f}\n"
            f"    title: {title}"
        )

    if not clip_ids:
        return {"dispatched": 0, "clip_ids": []}

    from celery import chord as _chord, group as _group, chain as _chain
    from tasks.package_dataset import package_dataset, trigger_finetune_check

    batch = _group(
        _chain(auto_label_clip.s(cid), package_dataset.s())
        for cid in clip_ids
    )
    _chord(batch)(trigger_finetune_check.si())

    return {"dispatched": len(clip_ids), "clip_ids": clip_ids}
