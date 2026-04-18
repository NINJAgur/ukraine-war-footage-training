"""
ml-engine/tasks/poll_clips.py

Celery Beat task: poll PostgreSQL for newly downloaded clips and dispatch
auto_label_clip for each one.

This is the bridge between scraper-engine and ml-engine. The scraper sets
Clip.status = DOWNLOADED when a video lands on disk. This task picks those
up and moves them into the GPU pipeline.

Clip status flow:
  DOWNLOADED → QUEUED (set here, before dispatch)
             → LABELED (set by auto_label_clip on completion)
             → ANNOTATED (set by render_annotated_clip on completion)

Setting QUEUED before dispatching prevents duplicate dispatch if Beat fires
again before the GPU worker starts the task.
"""
import logging

from celery_app import celery_app
from config import settings
from db.models import Clip, ClipStatus
from db.session import get_session

logger = logging.getLogger(__name__)

MAX_BATCH = 10  # max clips to dispatch per poll cycle — avoids flooding gpu queue


@celery_app.task(
    bind=True,
    name="tasks.poll_clips.poll_downloaded_clips",
    queue="gpu",
)
def poll_downloaded_clips(self) -> dict:
    """
    Find all DOWNLOADED clips, mark them QUEUED, dispatch auto_label_clip.
    Runs on a Beat schedule (every 5 minutes).
    Capped at MAX_BATCH per run to avoid overwhelming the GPU queue.
    """
    with get_session() as session:
        clips = (
            session.query(Clip)
            .filter(Clip.status == ClipStatus.DOWNLOADED)
            .filter(Clip.file_path.isnot(None))
            .order_by(Clip.created_at.asc())   # oldest first
            .limit(MAX_BATCH)
            .all()
        )

        if not clips:
            return {"dispatched": 0}

        clip_ids = [c.id for c in clips]
        # Mark QUEUED atomically before dispatch — prevents re-dispatch on next poll
        for clip in clips:
            clip.status = ClipStatus.QUEUED

    logger.info(f"[poll] Dispatching auto_label_clip for {len(clip_ids)} clips: {clip_ids}")

    from tasks.auto_label import auto_label_clip
    for clip_id in clip_ids:
        auto_label_clip.delay(clip_id=clip_id)

    return {"dispatched": len(clip_ids), "clip_ids": clip_ids}
