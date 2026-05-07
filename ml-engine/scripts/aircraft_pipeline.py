"""
aircraft_pipeline.py

Query database for aircraft-relevant clips (Majority Voting),
validate each clip has visible aircraft (>=15% frames with detections),
then run the AIRCRAFT model -> annotated MP4 saved to remote storage or local media/.
DB Clip entry written for each annotated clip, and heavy raw file deleted.

Usage (from repo root):
    cd ml-engine && python scripts/aircraft_pipeline.py
"""
import sys
import os
import shutil
import logging
from pathlib import Path
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("aircraft_pipeline")

REPO_ROOT = Path(__file__).resolve().parents[2]
ML_ENGINE_DIR = REPO_ROOT / "ml-engine"

sys.path.insert(0, str(ML_ENGINE_DIR))
sys.path.insert(0, str(REPO_ROOT))

from core.inference import validate_clip, infer_video_multi_model
from db.session import get_session
from shared.db.models import Clip, ClipStatus
from config import settings

COLOR = settings.MODEL_COLORS["AIRCRAFT"]
CONF_THRESH = 0.15
MIN_RATE    = 0.10


def _latest_weights(model_name: str) -> Path:
    """Return best.pt from the highest-numbered run that has it."""
    runs_dir = ML_ENGINE_DIR / "runs/baseline" / model_name
    if not runs_dir.exists():
        raise FileNotFoundError(f"No runs directory: {runs_dir}")
    candidates = sorted(
        (d for d in runs_dir.iterdir() if d.is_dir()),
        key=lambda d: int(d.name.rsplit("_", 1)[-1]) if d.name.rsplit("_", 1)[-1].isdigit() else 0,
        reverse=True,
    )
    for run_dir in candidates:
        w = run_dir / "weights" / "best.pt"
        if w.exists():
            return w
    raise FileNotFoundError(f"No best.pt found in {runs_dir}")


def finalize_storage(clip: Clip, temp_path: Path) -> str:
    storage_mode = getattr(settings, "STORAGE_MODE", "local")
    if storage_mode == "remote":
        bucket = getattr(settings, "REMOTE_STORAGE_BUCKET", "my-bucket")
        final_url = f"https://storage.googleapis.com/{bucket}/aircraft/{temp_path.name}"
        if temp_path.exists():
            os.remove(temp_path)
    else:
        clean_name = temp_path.stem.removeprefix("temp_").replace("_clip", "") + "_annotated.mp4"
        perm_path = temp_path.parent / clean_name
        shutil.move(str(temp_path), str(perm_path))
        final_url = str(perm_path)

    if clip.file_path:
        if os.path.exists(clip.file_path):
            try:
                os.remove(clip.file_path)
            except PermissionError:
                log.warning(f"Could not delete raw file (Windows lock): {clip.file_path}")
        clip.file_path = None

    return final_url


def _db_match_reason(clip: Clip) -> str:
    parts = []
    if clip.score_aircraft and clip.score_aircraft > 0:
        parts.append(f"score_aircraft={clip.score_aircraft}")
    if clip.score_uas and clip.score_uas > 0:
        parts.append(f"score_uas={clip.score_uas}")
    return " + ".join(parts) if parts else "unknown"


if __name__ == "__main__":
    log.info("=" * 60)
    log.info("AIRCRAFT PIPELINE START")
    log.info("=" * 60)
    from ultralytics import YOLO

    weights_path = _latest_weights("AIRCRAFT")
    log.info(f"Using weights: {weights_path}")
    model = YOLO(str(weights_path))

    with get_session() as session:
        candidates = (
            session.query(Clip)
            .filter(Clip.status == ClipStatus.DOWNLOADED)
            .filter(Clip.file_path.isnot(None))
            .filter(
                (Clip.score_aircraft > 0) &
                (Clip.score_aircraft >= Clip.score_vehicle) &
                (Clip.score_aircraft >= Clip.score_personnel)
            )
            .limit(10)
            .all()
        )

        log.info(f"DB query returned {len(candidates)} candidates")
        log.info("-" * 60)

        accepted = rejected = errors = 0
        total_detections = 0

        for clip in candidates:
            raw_path = Path(clip.file_path)
            match_reason = _db_match_reason(clip)

            log.info(
                f"[clip_id={clip.id}] {clip.source.value if hasattr(clip.source, 'value') else clip.source}\n"
                f"  title   : {(clip.title or 'N/A')[:100]}\n"
                f"  scores  : aircraft={clip.score_aircraft} vehicle={clip.score_vehicle} "
                f"uas={clip.score_uas} pov={clip.is_pov}\n"
                f"  matched : {match_reason}\n"
                f"  file    : {raw_path.name}"
            )

            if not raw_path.exists():
                log.warning(f"  SKIP — file not on disk: {raw_path}")
                clip.status = ClipStatus.ERROR
                errors += 1
                continue

            passed, rate = validate_clip(model, raw_path, conf_thresh=CONF_THRESH, min_rate=MIN_RATE)

            if not passed:
                log.info(f"  REJECT — detection rate {rate:.0%} < {MIN_RATE:.0%} threshold")
                if raw_path.exists():
                    raw_path.unlink()
                    clip.file_path = None
                clip.status = ClipStatus.PENDING
                rejected += 1
                continue

            log.info(f"  ACCEPT — detection rate {rate:.0%} — running full inference...")

            date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            out_dir = settings.ANNOTATED_VIDEO_DIR / "aircraft" / date_str
            out_dir.mkdir(parents=True, exist_ok=True)
            temp_out = out_dir / f"temp_{raw_path.name}"
            _, det_counts = infer_video_multi_model(
                [(model, "AIRCRAFT", COLOR)], str(raw_path), save_path=str(temp_out),
                no_display=True, conf_thresh=CONF_THRESH
            )
            clip_dets = sum(det_counts.values())

            if clip_dets == 0:
                log.info(f"  REJECT — full inference produced 0 detections, skipping")
                if temp_out.exists():
                    temp_out.unlink()
                if raw_path.exists():
                    raw_path.unlink()
                    clip.file_path = None
                clip.status = ClipStatus.PENDING
                rejected += 1
                continue

            total_detections += clip_dets
            clip.mp4_path = finalize_storage(clip, temp_out)
            clip.det_class = "AIRCRAFT"
            clip.status = ClipStatus.ANNOTATED
            clip.updated_at = datetime.now(timezone.utc)
            accepted += 1

            log.info(
                f"  ANNOTATED — detections={clip_dets}  output={Path(clip.mp4_path).name}"
            )

        session.commit()

    log.info("=" * 60)
    log.info(
        f"AIRCRAFT PIPELINE DONE\n"
        f"  candidates : {len(candidates)}\n"
        f"  accepted   : {accepted}\n"
        f"  rejected   : {rejected}\n"
        f"  errors     : {errors}\n"
        f"  detections : {total_detections}"
    )
    log.info("=" * 60)
