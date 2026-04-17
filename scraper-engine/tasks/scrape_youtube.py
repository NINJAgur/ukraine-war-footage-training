"""
scraper-engine/tasks/scrape_youtube.py

Celery task: download videos from configured YouTube channels via yt-dlp.

Flow:
  1. For each channel URL in settings.YOUTUBE_CHANNELS:
     a. Fetch the channel's recent video list (flat extraction, no download)
     b. For each video not yet in DB: create Clip record + download file
  2. Single-video mode: download a specific URL (used by /submit form)
"""
import asyncio
import hashlib
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import yt_dlp
from sqlalchemy.dialects.postgresql import insert as pg_insert

from celery_app import celery_app
from config import settings
from db.models import Clip, ClipSource, ClipStatus
from db.session import get_session

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────

def canonical_url(url: str) -> str:
    """Normalize YouTube URL: strip playlist params, keep only watch?v=ID."""
    parsed = urlparse(url)
    if "youtube.com" in parsed.netloc:
        # Extract video ID and reconstruct clean URL
        video_id_match = re.search(r"[?&]v=([a-zA-Z0-9_-]{11})", url)
        if video_id_match:
            return f"https://www.youtube.com/watch?v={video_id_match.group(1)}"
    if "youtu.be" in parsed.netloc:
        video_id = parsed.path.lstrip("/").split("/")[0]
        return f"https://www.youtube.com/watch?v={video_id}"
    return url


def url_hash(url: str) -> str:
    return hashlib.sha256(canonical_url(url).encode()).hexdigest()


def slugify(text: str, max_len: int = 60) -> str:
    """Convert title to filesystem-safe slug."""
    slug = re.sub(r"[^\w\s-]", "", text.lower())
    slug = re.sub(r"[\s_-]+", "-", slug).strip("-")
    return slug[:max_len]


def get_output_path(video_id: str, title: str) -> Path:
    """Deterministic output path: media/raw/youtube/{hash8}_{slug}.mp4"""
    h = hashlib.sha256(f"https://www.youtube.com/watch?v={video_id}".encode()).hexdigest()
    slug = slugify(title or video_id)
    path = settings.RAW_VIDEO_DIR / "youtube" / f"{h[:8]}_{slug}.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


# ── Core download logic ───────────────────────────────────────────────

def fetch_channel_videos(channel_url: str, max_videos: int = 50) -> list[dict]:
    """
    Use yt-dlp flat extraction to get recent video metadata without downloading.
    Returns list of {id, url, title, description, channel, published_at}.
    """
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": "in_playlist",
        "playlistend": max_videos,
        "socket_timeout": 30,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(channel_url, download=False)

    if not info or "entries" not in info:
        return []

    videos = []
    for entry in info.get("entries", []):
        if not entry or not entry.get("id"):
            continue
        vid_url = f"https://www.youtube.com/watch?v={entry['id']}"
        published_at: Optional[datetime] = None
        if entry.get("upload_date"):
            try:
                published_at = datetime.strptime(entry["upload_date"], "%Y%m%d")
            except ValueError:
                pass
        videos.append({
            "id": entry["id"],
            "url": vid_url,
            "url_hash": url_hash(vid_url),
            "title": (entry.get("title") or "")[:500],
            "description": (entry.get("description") or "")[:2000],
            "channel": info.get("channel") or info.get("uploader") or "",
            "published_at": published_at,
        })
    return videos


def download_video(video_url: str, output_path: Path) -> dict:
    """
    Download a single video with yt-dlp.
    Returns metadata dict on success.
    """
    ydl_opts = {
        "format": settings.YTDLP_FORMAT,
        "outtmpl": str(output_path.with_suffix("")),  # yt-dlp adds extension
        "quiet": True,
        "no_warnings": True,
        "socket_timeout": 60,
        "retries": 3,
        "merge_output_format": "mp4",
        # Write thumbnail as cover art
        "writethumbnail": False,
        # Metadata
        "writeinfojson": False,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=True)

    # yt-dlp may produce .mp4 directly or with extension appended
    final_path = output_path if output_path.exists() else output_path.with_suffix(".mp4")

    return {
        "file_path": str(final_path),
        "duration_seconds": int(info.get("duration") or 0),
        "width": info.get("width"),
        "height": info.get("height"),
        "title": (info.get("title") or "")[:500],
        "description": (info.get("description") or "")[:2000],
        "channel": info.get("channel") or info.get("uploader") or "",
    }


# ── Celery Tasks ──────────────────────────────────────────────────────

@celery_app.task(
    bind=True,
    name="tasks.scrape_youtube.scrape_youtube_channels",
    queue="default",
    autoretry_for=(Exception,),
    max_retries=3,
    default_retry_delay=300,
)
def scrape_youtube_channels(self) -> dict:
    """
    Fetch recent videos from all configured YouTube channels.
    Creates Clip records for new videos; skips known ones.
    Does NOT download the videos — dispatches download_youtube_video for each new clip.
    """
    import redis as redis_lib

    channels = settings.youtube_channel_list
    if not channels:
        logger.info(f"[{self.request.id}] No YouTube channels configured — skipping")
        return {"status": "skipped", "reason": "no_channels_configured"}

    r = redis_lib.from_url(settings.REDIS_URL)
    lock_key = "lock:scrape_youtube_channels"
    if not r.set(lock_key, self.request.id, ex=3600, nx=True):
        logger.info(f"[{self.request.id}] scrape_youtube_channels already running — skipping")
        return {"status": "skipped", "reason": "lock_held"}

    logger.info(f"[{self.request.id}] scrape_youtube_channels started ({len(channels)} channels)")
    new_count = 0
    skipped_count = 0

    try:
        for channel_url in channels:
            logger.info(f"Fetching channel: {channel_url}")
            try:
                videos = fetch_channel_videos(channel_url, max_videos=30)
            except Exception as exc:
                logger.warning(f"Failed to fetch channel {channel_url}: {exc}")
                continue

            with get_session() as session:
                for video in videos:
                    stmt = (
                        pg_insert(Clip)
                        .values(
                            url=video["url"],
                            url_hash=video["url_hash"],
                            source=ClipSource.YOUTUBE,
                            title=video["title"],
                            description=video["description"],
                            channel=video["channel"],
                            published_at=video["published_at"],
                            status=ClipStatus.PENDING,
                        )
                        .on_conflict_do_nothing(index_elements=["url_hash"])
                        .returning(Clip.id)
                    )
                    result = session.execute(stmt)
                    row = result.fetchone()
                    if row:
                        clip_id = row[0]
                        new_count += 1
                        # Dispatch actual download as a separate task
                        download_youtube_video.delay(clip_id=clip_id, video_url=video["url"])
                    else:
                        skipped_count += 1

        result_data = {
            "source": "youtube",
            "channels_checked": len(channels),
            "new": new_count,
            "skipped": skipped_count,
        }
        logger.info(f"[{self.request.id}] scrape_youtube_channels completed: {result_data}")
        return result_data

    finally:
        r.delete(lock_key)


@celery_app.task(
    bind=True,
    name="tasks.scrape_youtube.download_youtube_video",
    queue="default",
    autoretry_for=(Exception,),
    max_retries=3,
    default_retry_delay=60,
)
def download_youtube_video(self, clip_id: int, video_url: str) -> dict:
    """
    Download a single YouTube video for an existing Clip record.
    Updates Clip.status to DOWNLOADED on success, ERROR on failure.
    Idempotent: skips if file already exists.
    """
    logger.info(f"[{self.request.id}] download_youtube_video clip_id={clip_id} url={video_url}")

    with get_session() as session:
        clip = session.get(Clip, clip_id)
        if clip is None:
            raise ValueError(f"Clip {clip_id} not found")

        # Idempotency: skip if already downloaded
        if clip.file_path and Path(clip.file_path).exists():
            logger.info(f"[{self.request.id}] Already downloaded: {clip.file_path}")
            return {"status": "skipped", "clip_id": clip_id}

        # Mark as in-progress
        clip.status = ClipStatus.DOWNLOADING
        clip.error_message = None

    # Determine output path from clip data
    video_id_match = re.search(r"v=([a-zA-Z0-9_-]{11})", video_url)
    video_id = video_id_match.group(1) if video_id_match else clip_id
    output_path = get_output_path(str(video_id), clip.title or "")

    try:
        meta = download_video(video_url, output_path)

        with get_session() as session:
            clip = session.get(Clip, clip_id)
            clip.status = ClipStatus.DOWNLOADED
            clip.file_path = meta["file_path"]
            clip.duration_seconds = meta["duration_seconds"]
            clip.width = meta["width"]
            clip.height = meta["height"]
            if not clip.title and meta["title"]:
                clip.title = meta["title"]

        logger.info(f"[{self.request.id}] Downloaded: {meta['file_path']}")
        return {"status": "downloaded", "clip_id": clip_id, "file_path": meta["file_path"]}

    except Exception as exc:
        logger.error(f"[{self.request.id}] Download failed for clip {clip_id}: {exc}")
        with get_session() as session:
            clip = session.get(Clip, clip_id)
            if clip:
                clip.status = ClipStatus.ERROR
                clip.error_message = str(exc)[:1000]
        raise
