"""
Nulls out mp4_path for clips whose video file no longer exists on disk.
Run inside the backend container after seeding media:
  docker-compose -f docker-compose.prod.yml exec backend python scripts/sync_media.py
"""
import logging
from pathlib import Path

from db.session import get_session
from db.models import Clip, ClipStatus

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

ANNOTATED_DIR = Path("/app/ml-engine/media/annotated")


def main():
    with get_session() as session:
        clips = session.query(Clip).filter(
            Clip.mp4_path.isnot(None),
            Clip.status == ClipStatus.ANNOTATED,
        ).all()

        missing = 0
        for clip in clips:
            path = Path(clip.mp4_path)
            if not path.is_absolute():
                path = ANNOTATED_DIR / path
            if not path.exists():
                logger.info(f"Missing: {clip.mp4_path} — clearing mp4_path (id={clip.id})")
                clip.mp4_path = None
                missing += 1

        session.flush()

    logger.info(f"Done — {missing}/{len(clips)} clips had missing files, mp4_path cleared.")


if __name__ == "__main__":
    main()
