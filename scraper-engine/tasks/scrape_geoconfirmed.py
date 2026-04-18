"""
Celery task: fetch geolocated Ukraine incidents from GeoConfirmed REST API,
create Clip records for new video entries, and dispatch yt-dlp downloads.
"""
import concurrent.futures
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
from tasks._filter import check_equipment, is_infrastructure_strike

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

_DOWNLOADABLE_DOMAINS = (
    "t.me",
    "twitter.com",
    "x.com",
    "rumble.com",
    "telegram.org",
    "vxtwitter.com",
    "fxtwitter.com",
)


def canonical_url(url: str) -> str:
    url = url.strip()
    parsed = urlparse(url)
    if "t.me" in parsed.netloc:
        return f"https://t.me{parsed.path}"
    if "twitter.com" in parsed.netloc or "x.com" in parsed.netloc:
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


def fetch_recent_placemark_ids(days_back: int = 60) -> list[dict]:
    logger.info("Fetching GeoConfirmed placemark list...")
    resp = requests.get(GEOCONFIRMED_LIST_URL, headers=_HEADERS, timeout=30)
    resp.raise_for_status()
    factions = resp.json()

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

    all_pms.sort(key=lambda x: x["date"], reverse=True)
    logger.info(f"GeoConfirmed: {len(all_pms)} placemarks available in last {days_back} days.")
    return all_pms


def fetch_placemark_detail(placemark_id: str) -> Optional[dict]:
    url = GEOCONFIRMED_DETAIL_URL.format(id=placemark_id)
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning(f"Failed to fetch detail for {placemark_id}: {exc}")
        return None


def extract_first_url(raw: str) -> Optional[str]:
    if not raw:
        return None
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("http://") or line.startswith("https://"):
            return line
    return None


def extract_video_incidents(max_incidents: int) -> list[dict]:
    recent_pms = fetch_recent_placemark_ids(days_back=60)

    seen_hashes: set[str] = set()
    results: list[dict] = []
    skipped = 0
    checked = 0
    
    BATCH_SIZE = 20
    MAX_WORKERS = 10

    for i in range(0, len(recent_pms), BATCH_SIZE):
        if len(results) >= max_incidents:
            break
            
        batch = recent_pms[i : i + BATCH_SIZE]
        logger.info(f"Processing GeoConfirmed batch {i} to {i+len(batch)}...")

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_pm = {executor.submit(fetch_placemark_detail, pm["id"]): pm for pm in batch}
            
            for future in concurrent.futures.as_completed(future_to_pm):
                pm_stub = future_to_pm[future]
                checked += 1
                
                try:
                    detail = future.result()
                except Exception:
                    skipped += 1
                    continue

                if not detail:
                    skipped += 1
                    continue

                raw_source = detail.get("originalSource") or ""
                source_url = extract_first_url(raw_source)
                if not source_url or not is_downloadable(source_url):
                    skipped += 1
                    continue

                h = url_hash(source_url)
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)

                name = (detail.get("name") or "").strip()
                desc = (detail.get("description") or "").strip()
                
                gear = str(detail.get("gear") or "")
                units = str(detail.get("units") or "")
                
                title = f"{name} — {desc}" if name and desc else name or desc

                # Feed description + gear into the filter
                filter_text = f"{desc} {gear} {units}"
                equip_ok, equip_reason = check_equipment(name, filter_text)
                is_infra, infra_reason = is_infrastructure_strike(name, filter_text)

                # Multi-line console logging restored
                logger.info(
                    f"  GeoConfirmed candidate  equipment={equip_reason!r}  impact={is_infra}\n"
                    f"    name: {name}\n"
                    f"    desc: {desc}\n"
                    f"    gear: {gear}"
                )

                if is_infra:
                    logger.info(f"    → SKIP: {infra_reason}")
                    skipped += 1
                    continue
                if not equip_ok:
                    logger.info(f"    → SKIP: {equip_reason}")
                    skipped += 1
                    continue

                results.append({
                    "url": canonical_url(source_url),
                    "url_hash": h,
                    "title": title[:500],
                    "description": desc[:2000],
                    "published_at": pm_stub["date"],
                    "equipment_match": equip_reason,
                })
                logger.info(f"    → ACCEPT  equipment='{equip_reason}'")
                
                if len(results) >= max_incidents:
                    break

    logger.info(f"GeoConfirmed: {len(results)} accepted, {skipped} skipped (checked {checked} placemarks)")
    return results[:max_incidents]


def _download_video(video_url: str, output_path: Path) -> dict:
    import yt_dlp
    stem = str(output_path.with_suffix(""))
    ydl_opts = {
        "format": settings.YTDLP_FORMAT,
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


@celery_app.task(
    bind=True,
    name="tasks.scrape_geoconfirmed.scrape_geoconfirmed",
    queue="default",
    autoretry_for=(Exception,),
    max_retries=3,
    default_retry_delay=300,
)
def scrape_geoconfirmed(self) -> dict:
    import redis as redis_lib
    r = redis_lib.from_url(settings.REDIS_URL)
    lock_key = "lock:scrape_geoconfirmed"
    if not r.set(lock_key, self.request.id, ex=3600, nx=True):
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
                    download_geoconfirmed_video.delay(clip_id=clip_id, video_url=incident["url"])
                else:
                    skipped_count += 1

        summary = {"source": "geoconfirmed", "incidents_checked": len(incidents), "new": new_count, "skipped": skipped_count}
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
    with get_session() as session:
        clip = session.get(Clip, clip_id)
        if clip is None:
            raise ValueError(f"Clip {clip_id} not found")
        if clip.file_path and Path(clip.file_path).exists():
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
        return {"status": "downloaded", "clip_id": clip_id, "file_path": meta["file_path"]}
    except Exception as exc:
        with get_session() as session:
            clip = session.get(Clip, clip_id)
            if clip:
                clip.status = ClipStatus.ERROR
                clip.error_message = str(exc)[:1000]
        raise