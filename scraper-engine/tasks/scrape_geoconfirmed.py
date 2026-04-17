"""
scraper-engine/tasks/scrape_geoconfirmed.py

Celery task: fetch geolocated Ukraine incidents from GeoConfirmed REST API,
create Clip records for new video entries, and dispatch yt-dlp downloads.

API structure (discovered via browser DevTools):
  GET /api/placemark/Ukraine
    → [{factionId, name, color, icons: [{iconId, icon, placemarks: [{id, date, la, lo, ds}]}]}]
  GET /api/placemark/detail/{id}
    → {id, name, description, date, latitude, longitude, icon, faction,
       originalSource, geolocation, origin, gear, units, plusCode, ...}

Flow:
  1. Fetch all placemarks, sort by date desc, take GEOCONFIRMED_MAX_INCIDENTS newest
  2. For each, fetch detail and filter origin=="VID" (video clips only)
  3. Skip if no originalSource or if URL not downloadable by yt-dlp
  4. Insert Clip records (ON CONFLICT DO NOTHING) and dispatch downloads
"""
import hashlib
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests
from sqlalchemy.dialects.postgresql import insert as pg_insert

from celery_app import celery_app
from config import settings
from db.models import Clip, ClipSource, ClipStatus
from db.session import get_session

logger = logging.getLogger(__name__)

GEOCONFIRMED_BASE = "https://geoconfirmed.org"
GEOCONFIRMED_LIST_URL = f"{GEOCONFIRMED_BASE}/api/placemark/Ukraine"
GEOCONFIRMED_DETAIL_URL = f"{GEOCONFIRMED_BASE}/api/placemark/detail/{{id}}"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Referer": "https://geoconfirmed.org/map/ukraine",
}

# Domains yt-dlp can download from
_DOWNLOADABLE_DOMAINS = (
    "t.me",
    "twitter.com",
    "x.com",
    "youtube.com",
    "youtu.be",
    "rumble.com",
    "telegram.org",
    "vxtwitter.com",
    "fxtwitter.com",
)


# ── Helpers ───────────────────────────────────────────────────────────

def canonical_url(url: str) -> str:
    url = url.strip()
    parsed = urlparse(url)
    if "t.me" in parsed.netloc:
        return f"https://t.me{parsed.path}"
    if "twitter.com" in parsed.netloc or "x.com" in parsed.netloc:
        # Keep only scheme + netloc + path (strip query/fragment)
        return f"https://twitter.com{parsed.path}"
    return url


def url_hash(url: str) -> str:
    return hashlib.sha256(canonical_url(url).encode()).hexdigest()


def slugify(text: str, max_len: int = 60) -> str:
    slug = re.sub(r"[^\w\s-]", "", (text or "").lower())
    slug = re.sub(r"[\s_-]+", "-", slug).strip("-")
    return slug[:max_len] or "video"


def get_output_path(url: str, title: str) -> Path:
    h = url_hash(url)
    slug = slugify(title)
    path = settings.RAW_VIDEO_DIR / "geoconfirmed" / f"{h[:8]}_{slug}.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def is_downloadable(url: str) -> bool:
    if not url:
        return False
    return any(domain in url for domain in _DOWNLOADABLE_DOMAINS)


# ── GeoConfirmed API ──────────────────────────────────────────────────

def fetch_recent_placemark_ids(max_count: int, days_back: int = 30) -> list[dict]:
    """
    Fetch all placemarks from GeoConfirmed API, return the `max_count` most recent.
    Filters to last `days_back` days to avoid re-processing ancient data.
    Returns list of {id, date}.
    """
    logger.info(f"Fetching GeoConfirmed placemark list...")
    resp = requests.get(GEOCONFIRMED_LIST_URL, headers=_HEADERS, timeout=30)
    resp.raise_for_status()
    factions = resp.json()  # list of faction objects

    cutoff = datetime.utcnow() - timedelta(days=days_back)
    all_pms: list[dict] = []
    for faction in factions:
        for icon in faction.get("icons", []):
            for pm in icon.get("placemarks", []):
                pm_date = None
                if pm.get("date"):
                    try:
                        pm_date = datetime.fromisoformat(pm["date"])
                    except (ValueError, TypeError):
                        pass
                if pm_date and pm_date >= cutoff:
                    all_pms.append({"id": pm["id"], "date": pm_date})

    # Sort by date descending, take most recent max_count
    all_pms.sort(key=lambda x: x["date"], reverse=True)
    selected = all_pms[:max_count]
    logger.info(
        f"GeoConfirmed: {len(all_pms)} placemarks in last {days_back} days, "
        f"processing {len(selected)}"
    )
    return selected


def fetch_placemark_detail(placemark_id: str) -> Optional[dict]:
    """Fetch full incident detail for a single placemark ID."""
    url = GEOCONFIRMED_DETAIL_URL.format(id=placemark_id)
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning(f"Failed to fetch detail for {placemark_id}: {exc}")
        return None


def extract_first_url(raw: str) -> Optional[str]:
    """
    Extract the first valid URL from a GeoConfirmed originalSource string.
    The field often contains multiple newline-separated URLs plus free text.
    """
    if not raw:
        return None
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("http://") or line.startswith("https://"):
            return line
    return None


def extract_video_incidents(max_incidents: int) -> list[dict]:
    """
    Fetch recent GeoConfirmed video incidents.
    Returns list of {url, url_hash, title, description, published_at}.
    """
    recent_pms = fetch_recent_placemark_ids(
        max_count=max_incidents * 3,  # over-fetch since many won't be videos
        days_back=30,
    )

    seen_hashes: set[str] = set()
    incidents: list[dict] = []
    for pm in recent_pms:
        if len(incidents) >= max_incidents:
            break
        detail = fetch_placemark_detail(pm["id"])
        if not detail:
            continue

        # Only process video clips
        if detail.get("origin") != "VID":
            continue

        raw_source = detail.get("originalSource") or ""
        source_url = extract_first_url(raw_source)
        if not source_url or not is_downloadable(source_url):
            continue

        h = url_hash(source_url)
        if h in seen_hashes:
            continue
        seen_hashes.add(h)

        # Build a descriptive title from name + description
        name = (detail.get("name") or "").strip()
        desc = (detail.get("description") or "").strip()
        title = f"{name} — {desc}" if name and desc else name or desc

        incidents.append({
            "url": canonical_url(source_url),
            "url_hash": h,
            "title": title[:500],
            "description": raw_source[:2000],
            "published_at": pm["date"],
        })

    logger.info(f"GeoConfirmed: found {len(incidents)} downloadable video incidents")
    return incidents


# ── yt-dlp download ───────────────────────────────────────────────────

def _download_video(video_url: str, output_path: Path) -> dict:
    import yt_dlp

    ydl_opts = {
        "format": settings.YTDLP_FORMAT,
        "outtmpl": str(output_path.with_suffix("")),
        "quiet": True,
        "no_warnings": True,
        "socket_timeout": 60,
        "retries": 3,
        "merge_output_format": "mp4",
        "writeinfojson": False,
        "writethumbnail": False,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=True)

    final_path = output_path if output_path.exists() else output_path.with_suffix(".mp4")
    return {
        "file_path": str(final_path),
        "duration_seconds": int(info.get("duration") or 0),
        "width": info.get("width"),
        "height": info.get("height"),
        "title": (info.get("title") or "")[:500],
        "description": (info.get("description") or "")[:2000],
    }


# ── Celery Tasks ──────────────────────────────────────────────────────

@celery_app.task(
    bind=True,
    name="tasks.scrape_geoconfirmed.scrape_geoconfirmed",
    queue="default",
    autoretry_for=(Exception,),
    max_retries=3,
    default_retry_delay=300,
)
def scrape_geoconfirmed(self) -> dict:
    """
    Fetch latest GeoConfirmed video incidents, create Clip records for new ones,
    and dispatch per-video download tasks.
    Uses a Redis lock to prevent overlapping Beat executions.
    """
    import redis as redis_lib

    r = redis_lib.from_url(settings.REDIS_URL)
    lock_key = "lock:scrape_geoconfirmed"
    if not r.set(lock_key, self.request.id, ex=3600, nx=True):
        logger.info(f"[{self.request.id}] scrape_geoconfirmed already running — skipping")
        return {"status": "skipped", "reason": "lock_held"}

    logger.info(f"[{self.request.id}] scrape_geoconfirmed started")
    new_count = 0
    skipped_count = 0

    try:
        incidents = extract_video_incidents(settings.GEOCONFIRMED_MAX_INCIDENTS)
        if not incidents:
            return {"status": "ok", "new": 0, "skipped": 0}

        with get_session() as session:
            for incident in incidents:
                stmt = (
                    pg_insert(Clip)
                    .values(
                        url=incident["url"],
                        url_hash=incident["url_hash"],
                        source=ClipSource.GEOCONFIRMED,
                        title=incident["title"],
                        description=incident["description"],
                        published_at=incident["published_at"],
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
                    download_geoconfirmed_video.delay(
                        clip_id=clip_id, video_url=incident["url"]
                    )
                else:
                    skipped_count += 1

        summary = {
            "source": "geoconfirmed",
            "incidents_checked": len(incidents),
            "new": new_count,
            "skipped": skipped_count,
        }
        logger.info(f"[{self.request.id}] scrape_geoconfirmed completed: {summary}")
        return summary

    finally:
        r.delete(lock_key)


@celery_app.task(
    bind=True,
    name="tasks.scrape_geoconfirmed.download_geoconfirmed_video",
    queue="default",
    autoretry_for=(Exception,),
    max_retries=3,
    default_retry_delay=60,
)
def download_geoconfirmed_video(self, clip_id: int, video_url: str) -> dict:
    """
    Download a single GeoConfirmed-sourced video for an existing Clip record.
    Updates status to DOWNLOADED on success, ERROR on failure.
    Idempotent: skips if file already on disk.
    """
    logger.info(
        f"[{self.request.id}] download_geoconfirmed_video clip_id={clip_id} url={video_url}"
    )

    with get_session() as session:
        clip = session.get(Clip, clip_id)
        if clip is None:
            raise ValueError(f"Clip {clip_id} not found")
        if clip.file_path and Path(clip.file_path).exists():
            logger.info(f"[{self.request.id}] Already downloaded: {clip.file_path}")
            return {"status": "skipped", "clip_id": clip_id}
        clip.status = ClipStatus.DOWNLOADING
        clip.error_message = None

    output_path = get_output_path(video_url, "")

    try:
        meta = _download_video(video_url, output_path)

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
