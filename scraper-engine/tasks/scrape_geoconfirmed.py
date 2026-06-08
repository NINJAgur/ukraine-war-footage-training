"""
Celery task: fetch geolocated Ukraine incidents from GeoConfirmed REST API,
create Clip records for new video entries, and dispatch yt-dlp downloads.
"""
import concurrent.futures
import hashlib
import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests
from sqlalchemy.dialects.postgresql import insert as pg_insert

from celery_app import celery_app
from config import settings
from db.models import Clip, ClipSource, ClipStatus
from db.session import get_session
from utils._filter import get_equipment_scores, is_negative_input, is_pov_noise

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


def get_output_path(url: str, title: str, date_str: str = None) -> Path:
    h = url_hash(url)
    slug = slugify(title)
    date_str = date_str or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = settings.GEOCONFIRMED_DIR / date_str / f"{h[:8]}_{slug}.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def is_downloadable(url: str) -> bool:
    if not url:
        return False
    return any(domain in url for domain in _DOWNLOADABLE_DOMAINS)


def fetch_recent_placemark_ids(since_date: datetime) -> list[dict]:
    resp = requests.get(GEOCONFIRMED_LIST_URL, headers=_HEADERS, timeout=30)
    resp.raise_for_status()
    factions = resp.json()

    cutoff = since_date.replace(tzinfo=None) if since_date.tzinfo else since_date
    cutoff = cutoff.replace(hour=0, minute=0, second=0, microsecond=0)

    all_pms: list[dict] = []
    for faction in factions:
        for icon in faction.get("icons", []):
            for pm in icon.get("placemarks", []):
                pm_date = None
                if pm.get("date"):
                    try:
                        pm_date = datetime.fromisoformat(pm["date"])
                        if pm_date.tzinfo:
                            pm_date = pm_date.replace(tzinfo=None)
                    except (ValueError, TypeError):
                        pass
                if pm_date and pm_date >= cutoff:
                    all_pms.append({"id": pm["id"], "date": pm_date})

    all_pms.sort(key=lambda x: x["date"], reverse=True)
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


# ── Internal helpers ──────────────────────────────────────────────────

def _process_placemarks(placemarks: list[dict], max_incidents: int = None) -> list[dict]:
    """Fetch details and filter a list of placemarks. Stops early if max_incidents is set."""
    seen_hashes: set[str] = set()
    results: list[dict] = []
    skipped = checked = 0
    BATCH_SIZE = 20
    MAX_WORKERS = 10
    done = False

    for i in range(0, len(placemarks), BATCH_SIZE):
        if done:
            break
        batch = placemarks[i : i + BATCH_SIZE]
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
                raw_desc = (detail.get("description") or "").strip()
                desc_lines = [line.strip() for line in raw_desc.split('\n') if line.strip()]
                desc = desc_lines[0] if desc_lines else ""
                units = str(detail.get("units") or "")
                title = f"{name} - {desc}" if name and desc else name or desc

                filter_text = f"{desc} {units}"
                scores, equip_ok = get_equipment_scores(name, filter_text)
                is_neg, neg_reason = is_negative_input(name, filter_text)

                logger.info(
                    f"  GeoConfirmed candidate  date={pm_stub['date'].date()}  scores={scores}  negative={is_neg}\n"
                    f"    name: {name}\n"
                    f"    desc: {desc}"
                )

                if is_neg:
                    logger.info(f"    -> SKIP: {neg_reason}")
                    skipped += 1
                    continue
                if not equip_ok:
                    logger.info("    -> SKIP: no equipment matches")
                    skipped += 1
                    continue
                if is_pov_noise(scores):
                    logger.info("    -> SKIP: pure FPV noise (pov + no class score)")
                    skipped += 1
                    continue

                results.append({
                    "url": canonical_url(source_url),
                    "url_hash": h,
                    "title": title[:500],
                    "description": desc[:2000],
                    "published_at": pm_stub["date"],
                    "scores": scores,
                })
                logger.info(f"    -> ACCEPT  scores='{scores}'")

                if max_incidents and len(results) >= max_incidents:
                    logger.info(f"  Reached max_incidents={max_incidents} — stopping")
                    done = True
                    break

    logger.info(f"GeoConfirmed: {len(results)} accepted, {skipped} skipped (checked {checked} placemarks)")
    return results


# ── Public API ────────────────────────────────────────────────────────

# Celery / scrape_daily: fetch everything published in the last N hours
def extract_video_incidents_since(since_date: datetime) -> list[dict]:
    """Date mode — all incidents published on or after since_date. Used by Celery and scrape_daily."""
    placemarks = fetch_recent_placemark_ids(since_date)
    logger.info(f"GeoConfirmed: {len(placemarks)} total placemarks — filtering since {since_date.date()}")
    return _process_placemarks(placemarks)


# Tests: grab the first N passing incidents to verify the pipeline end-to-end
def extract_video_incidents_sample(max_incidents: int) -> list[dict]:
    """Count mode — first N passing incidents regardless of date. Used by tests."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=365)
    placemarks = fetch_recent_placemark_ids(cutoff)
    logger.info(f"GeoConfirmed: {len(placemarks)} total placemarks — sampling max_incidents={max_incidents}")
    return _process_placemarks(placemarks, max_incidents=max_incidents)


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
        since_date = datetime.now(timezone.utc) - timedelta(hours=settings.SCRAPE_LOOKBACK_HOURS)
        incidents = extract_video_incidents_since(since_date)
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
                        **incident["scores"]
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

    with get_session() as session:
        clip = session.get(Clip, clip_id)
        date_str = clip.published_at.strftime("%Y-%m-%d") if clip and clip.published_at else datetime.now(timezone.utc).strftime("%Y-%m-%d")

    output_path = get_output_path(video_url, "", date_str)
    try:
        meta = _download_video(video_url, output_path)
        file_path = meta["file_path"]
        if settings.STORAGE_MODE == "remote":
            from utils.gcs import upload_raw
            file_path = upload_raw(Path(file_path), "geoconfirmed", settings.REMOTE_STORAGE_BUCKET)
        with get_session() as session:
            clip = session.get(Clip, clip_id)
            clip.status = ClipStatus.DOWNLOADED
            clip.file_path = file_path
            clip.duration_seconds = meta["duration_seconds"]
            clip.width = meta["width"]
            clip.height = meta["height"]
            if not clip.title and meta["title"]:
                clip.title = meta["title"]
        return {"status": "downloaded", "clip_id": clip_id, "file_path": file_path}
    except Exception as exc:
        with get_session() as session:
            clip = session.get(Clip, clip_id)
            if clip:
                clip.status = ClipStatus.ERROR
                clip.error_message = str(exc)[:1000]
        raise