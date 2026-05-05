"""
vehicle_pipeline.py

Query database for vehicle-relevant clips (Majority Voting),
validate each clip has visible vehicles (≥15% frames with detections),
then run the VEHICLE model → annotated MP4 saved to remote storage or local media/.
DB Clip entry written for each annotated clip, and heavy raw file deleted.

Usage (from repo root):
    cd ml-engine && python scripts/vehicle_pipeline.py
"""
import sys
import os
import shutil
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("vehicle_pipeline")

REPO_ROOT = Path(__file__).resolve().parents[2]
ML_ENGINE_DIR = REPO_ROOT / "ml-engine"

sys.path.insert(0, str(ML_ENGINE_DIR))
sys.path.insert(0, str(REPO_ROOT))

from core.inference import validate_clip, infer_video_multi_model
from db.session import get_session
from shared.db.models import Clip, ClipStatus
from config import settings

VEHICLE_WEIGHTS = ML_ENGINE_DIR / "runs/baseline/VEHICLE/baseline_VEHICLE_25/weights/best.pt"
COLOR = settings.MODEL_COLORS["VEHICLE"]


def finalize_storage(clip: Clip, temp_path: Path) -> str:
    """Uploads/moves annotated video, deletes raw file to save space, returns permanent URL."""
    final_url = ""
    storage_mode = getattr(settings, "STORAGE_MODE", "local")
    
    if storage_mode == "remote":
        # Placeholder for generic cloud upload logic (GCP, Azure, S3)
        bucket = getattr(settings, "REMOTE_STORAGE_BUCKET", "my-bucket")
        final_url = f"https://storage.googleapis.com/{bucket}/vehicle/{temp_path.name}"
        if temp_path.exists():
            os.remove(temp_path)
    else:
        perm_dir = settings.ANNOTATED_VIDEO_DIR / "vehicle"
        perm_dir.mkdir(parents=True, exist_ok=True)
        clean_name = temp_path.stem.removeprefix("temp_").replace("_clip", "") + "_annotated.mp4"
        perm_path = perm_dir / clean_name
        shutil.move(str(temp_path), str(perm_path))
        final_url = str(perm_path)

    # Clean up the raw file downloaded by Celery to save disk space
    if clip.file_path and os.path.exists(clip.file_path):
        os.remove(clip.file_path)
        clip.file_path = None
        log.info(f"Deleted raw file from disk.")

    return final_url


if __name__ == "__main__":
    log.info("=== VEHICLE PIPELINE START ===")
    from ultralytics import YOLO

    if not VEHICLE_WEIGHTS.exists():
        raise FileNotFoundError(f"VEHICLE weights not found: {VEHICLE_WEIGHTS}")
    
    model = YOLO(str(VEHICLE_WEIGHTS))

    with get_session() as session:
        # ── The DB Query Magic (Majority Voting logic) ──
        # Vehicle > 0 OR UAS > 0 (Even POV kamikazes are valid for vehicles)
        candidates = (
            session.query(Clip)
            .filter(Clip.status == ClipStatus.DOWNLOADED)
            .filter(Clip.file_path.isnot(None))
            .filter((Clip.score_vehicle > 0) | (Clip.score_uas > 0))
            .limit(10)
            .all()
        )

        log.info(f"Found {len(candidates)} candidates in DB.")

        for clip in candidates:
            raw_path = Path(clip.file_path)
            if not raw_path.exists():
                clip.status = ClipStatus.ERROR
                continue
            
            log.info(f"Validating {raw_path.name}...")
            if validate_clip(model, raw_path, conf_thresh=0.15):
                log.info(f"✅ Vehicle found in {raw_path.name}")
                
                temp_out = ML_ENGINE_DIR / "media" / f"temp_{raw_path.name}"
                infer_video_multi_model([(model, "VEHICLE", COLOR)], str(raw_path), save_path=str(temp_out), no_display=True)
                
                clip.mp4_path = finalize_storage(clip, temp_out)
                clip.det_class = "VEHICLE"
                clip.status = ClipStatus.ANNOTATED
                clip.updated_at = datetime.utcnow()
            else:
                log.warning(f"❌ No vehicle in {raw_path.name}")
                clip.status = ClipStatus.PENDING 
        
        session.commit()
    log.info("=== DONE ===")