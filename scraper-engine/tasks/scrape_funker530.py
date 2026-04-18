"""
Celery task: fetch latest Ukraine war video posts from Funker530 REST API,
create Clip records for new entries, and dispatch yt-dlp downloads.

Flow:
  1. Fetch recent video posts strictly from the Ukraine category (categories=16)
  2. Iterate until max_count is reached
  3. Require: geo keyword match (Ukraine/Russia theater)
  4. Require: explicit equipment or personnel keyword
  5. Reject: infrastructure or civilian targeting
  6. Resolve video URL from rumbleJson or bunnyId
  7. Insert Clip records (ON CONFLICT DO NOTHING) and dispatch downloads
"""
import hashlib
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests
from sqlalchemy.dialects.postgresql import insert as pg_insert

from celery_app import celery_app
from config import settings
from db.models import Clip, ClipSource, ClipStatus
from db.session import get_session
from tasks._filter import check_equipment, check_geo, is_infrastructure_strike

logger = logging.getLogger(__name__)

FUNKER530_API_URL = (
    "https://api.funker530.com/api/Get"
    "?code=sL3mjD-c0BJdI9b9h4s7WhIPU8ca9p6h3yiLyFczS-I9AzFupvbo9g%3D%3D"
    "&categories=16"
)
BUNNY_LIBRARY_ID = "167129"
BUNNY_EMBED_BASE = f"https://iframe.mediadelivery.net/embed/{BUNNY_LIBRARY_ID}"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "gettype": "Video",
    "Accept": "application/json",
    "Referer": "https://funker530.com/",
}


# ── Helpers ───────────────────────────────────────────────────────────

def canonical_url(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.netloc.lstrip("www.")
    path = parsed.path.rstrip("/")
    return f"{parsed.scheme}://{host}{path}"


def url_hash(url: str) -> str:
    return hashlib.sha256(canonical_url(url).encode()).hexdigest()


def slugify(text: str, max_len: int = 60) -> str:
    slug = re.sub(r"[^\w\s-]", "", (text or "").lower())
    slug = re.sub(r"[\s_-]+", "-", slug).strip("-")
    return slug[:max_len] or "video"


def get_output_path(url: str, title: str) -> Path:
    h = url_hash(url)
    slug = slugify(title)
    path = settings.RAW_VIDEO_DIR / "funker530" / f"{h[:8]}_{slug}.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def resolve_video_url(post: dict) -> Optional[str]:
    """
    Resolve a yt-dlp-downloadable video URL from a Funker530 API post object.
    Priority: rumbleJson URL → Bunny.net embed URL.
    """
    rumble_raw = post.get("rumbleJson") or ""
    if rumble_raw:
        try:
            rj = json.loads(rumble_raw)
            rumble_url = rj.get("url", "")
            if rumble_url and rumble_url.startswith("http"):
                return rumble_url
        except (json.JSONDecodeError, AttributeError):
            pass

    bunny_id = (post.get("bunnyId") or "").strip()
    if bunny_id:
        return f"{BUNNY_EMBED_BASE}/{bunny_id}"

    return None


# ── Funker530 API ──────────────────────────────────────────────────────

def fetch_ukraine_posts(max_count: int) -> list[dict]:
    """
    Fetch Ukraine-category video posts from Funker530 REST API.
    Returns list of {url_hash, page_url, video_url, title, description, published_at}.
    """
    logger.info("Fetching Funker530 Ukraine posts from REST API...")
    resp = requests.get(FUNKER530_API_URL, headers=_HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    posts = data if isinstance(data, list) else data.get("posts", data.get("items", []))

    def parse_date(p: dict) -> datetime:
        raw = p.get("publicationDate") or p.get("creationDate") or ""
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return datetime.min

    posts_sorted = sorted(posts, key=parse_date, reverse=True)
    logger.info(f"Funker530: {len(posts_sorted)} posts loaded. Searching for {max_count} valid clips...")

    seen_hashes: set[str] = set()
    results: list[dict] = []
    skipped = 0
    checked = 0

    for post in posts_sorted:
        if len(results) >= max_count:
            break

        slug = (post.get("slug") or "").strip()
        if not slug:
            continue

        title = (post.get("title") or "").strip()
        raw_desc = (
            post.get("ogDescription") or 
            post.get("excerpt") or 
            post.get("description") or 
            post.get("content") or 
            ""
        ).strip()
        
        description = re.sub(r'<[^>]+>', '', raw_desc).strip()
        checked += 1

        geo = check_geo(title, description)
        equip_ok, equip_reason = check_equipment(title, description)
        is_infra, infra_reason = is_infrastructure_strike(title, description)

        logger.info(
            f"  Funker530 candidate  geo={geo!r}  equipment={equip_reason!r}  impact={is_infra}\n"
            f"    title: {title}\n"
            f"    desc:  {description}"
        )

        if not geo:
            logger.info(f"    → SKIP: no Ukraine/Russia geo keyword")
            skipped += 1
            continue
        if is_infra:
            logger.info(f"    → SKIP: {infra_reason}")
            skipped += 1
            continue
        if not equip_ok:
            logger.info(f"    → SKIP: {equip_reason}")
            skipped += 1
            continue

        video_url = resolve_video_url(post)
        if not video_url:
            logger.info(f"    → SKIP: no downloadable URL")
            skipped += 1
            continue

        page_url = f"https://funker530.com/video/{slug}/"
        h = url_hash(page_url)
        if h in seen_hashes:
            continue
        seen_hashes.add(h)

        published_at = parse_date(post)
        results.append({
            "page_url": page_url,
            "video_url": video_url,
            "url_hash": h,
            "title": title[:500],
            "description": description[:2000],
            "published_at": published_at if published_at != datetime.min else None,
        })
        logger.info(f"    → ACCEPT  equipment='{equip_reason}'  geo='{geo}'")

    logger.info(f"Funker530: {len(results)} accepted, {skipped} skipped (checked {checked} candidates)")
    return results


# ── yt-dlp download ───────────────────────────────────────────────────

def _download_video(video_url: str, output_path: Path) -> dict:
    import yt_dlp
    stem = str(output_path.with_suffix(""))
    fmt = "bestvideo[height<=1080]+bestaudio/bestvideo+bestaudio/best[height<=1080]/best"
    ydl_opts = {
        "format": fmt,
        "outtmpl": f"{stem}.%(ext)s",
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

    ext = info.get("ext") or "mp4"
    final_path = output_path.with_suffix(f".{ext}")
    if not final_path.exists():
        matches = list(output_path.parent.glob(f"{output_path.stem}.*"))
        if matches:
            final_path = matches[0]

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
    name="tasks.scrape_funker530.scrape_funker530",
    queue="default",
    autoretry_for=(Exception,),
    max_retries=3,
    default_retry_delay=300,
)
def scrape_funker530(self) -> dict:
    import redis as redis_lib
    r = redis_lib.from_url(settings.REDIS_URL)
    lock_key = "lock:scrape_funker530"
    if not r.set(lock_key, self.request.id, ex=3600, nx=True):
        return {"status": "skipped", "reason": "lock_held"}

    logger.info(f"[{self.request.id}] scrape_funker530 started")
    new_count = 0
    skipped_count = 0

    try:
        posts = fetch_ukraine_posts(settings.FUNKER530_MAX_POSTS)
        if not posts:
            return {"status": "ok", "new": 0, "skipped": 0}

        with get_session() as session:
            for post in posts:
                stmt = (
                    pg_insert(Clip)
                    .values(
                        url=post["page_url"],
                        url_hash=post["url_hash"],
                        source=ClipSource.FUNKER530,
                        title=post["title"] or None,
                        description=post["description"] or None,
                        published_at=post["published_at"],
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
                    download_funker530_video.delay(
                        clip_id=clip_id,
                        video_url=post["video_url"],
                        page_url=post["page_url"],
                    )
                else:
                    skipped_count += 1

        summary = {"source": "funker530", "posts_checked": len(posts), "new": new_count, "skipped": skipped_count}
        logger.info(f"[{self.request.id}] scrape_funker530 completed: {summary}")
        return summary
    finally:
        r.delete(lock_key)


@celery_app.task(
    bind=True,
    name="tasks.scrape_funker530.download_funker530_video",
    queue="default",
    autoretry_for=(Exception,),
    max_retries=3,
    default_retry_delay=60,
)
def download_funker530_video(self, clip_id: int, video_url: str, page_url: str) -> dict:
    with get_session() as session:
        clip = session.get(Clip, clip_id)
        if clip is None:
            raise ValueError(f"Clip {clip_id} not found")
        if clip.file_path and Path(clip.file_path).exists():
            return {"status": "skipped", "clip_id": clip_id}
        clip.status = ClipStatus.DOWNLOADING
        clip.error_message = None

    output_path = get_output_path(page_url, "")
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
        return {"status": "downloaded", "clip_id": clip_id, "file_path": meta["file_path"]}
    except Exception as exc:
        with get_session() as session:
            clip = session.get(Clip, clip_id)
            if clip:
                clip.status = ClipStatus.ERROR
                clip.error_message = str(exc)[:1000]
        raise