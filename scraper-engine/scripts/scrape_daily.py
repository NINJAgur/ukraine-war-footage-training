"""
scrape_daily.py

Standalone daily scrape script — intended to be called by Celery Beat or run manually.
Fetches all videos published in the last SCRAPE_LOOKBACK_HOURS hours from both sources,
inserts new Clip rows into DB, and dispatches yt-dlp download tasks.

Usage:
    cd scraper-engine && python scripts/scrape_daily.py
    cd scraper-engine && python scripts/scrape_daily.py --hours 48
"""
import sys
import logging
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("scrape_daily")

SCRAPER_ENGINE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRAPER_ENGINE_DIR))
sys.path.insert(0, str(SCRAPER_ENGINE_DIR.parent / "shared"))

from sqlalchemy.dialects.postgresql import insert as pg_insert

from config import settings
from db.models import Clip, ClipSource, ClipStatus
from db.session import get_session
from tasks.scrape_funker530 import (
    fetch_ukraine_posts_since,
    download_funker530_video,
)
from tasks.scrape_geoconfirmed import (
    extract_video_incidents_since,
    download_geoconfirmed_video,
)


def _insert_clip(session, values: dict, source: ClipSource) -> int | None:
    stmt = (
        pg_insert(Clip)
        .values(**values)
        .on_conflict_do_nothing(index_elements=["url_hash"])
        .returning(Clip.id)
    )
    row = session.execute(stmt).fetchone()
    return row[0] if row else None


def run(since_date: datetime) -> dict:
    log.info("=" * 60)
    log.info(f"DAILY SCRAPE  since={since_date.strftime('%Y-%m-%d %H:%M UTC')}")
    log.info("=" * 60)

    # ── Funker530 ─────────────────────────────────────────────────────
    log.info("Scraping Funker530...")
    f_posts = fetch_ukraine_posts_since(since_date)
    f_new = f_skip = 0

    with get_session() as session:
        for post in f_posts:
            clip_id = _insert_clip(session, {
                "url":          post["page_url"],
                "url_hash":     post["url_hash"],
                "source":       ClipSource.FUNKER530,
                "title":        post["title"] or None,
                "description":  post["description"] or None,
                "published_at": post["published_at"],
                "status":       ClipStatus.PENDING,
                **post["scores"],
            }, ClipSource.FUNKER530)
            if clip_id:
                f_new += 1
                download_funker530_video.delay(
                    clip_id=clip_id,
                    video_url=post["video_url"],
                    page_url=post["page_url"],
                )
            else:
                f_skip += 1

    log.info(f"Funker530: {f_new} new  {f_skip} already in DB")

    # ── GeoConfirmed ──────────────────────────────────────────────────
    log.info("Scraping GeoConfirmed...")
    g_incidents = extract_video_incidents_since(since_date)
    g_new = g_skip = 0

    with get_session() as session:
        for inc in g_incidents:
            clip_id = _insert_clip(session, {
                "url":          inc["url"],
                "url_hash":     inc["url_hash"],
                "source":       ClipSource.GEOCONFIRMED,
                "title":        inc["title"] or None,
                "description":  inc["description"] or None,
                "published_at": inc["published_at"],
                "status":       ClipStatus.PENDING,
                **inc["scores"],
            }, ClipSource.GEOCONFIRMED)
            if clip_id:
                g_new += 1
                download_geoconfirmed_video.delay(clip_id=clip_id, video_url=inc["url"])
            else:
                g_skip += 1

    log.info(f"GeoConfirmed: {g_new} new  {g_skip} already in DB")

    summary = {
        "funker530":    {"new": f_new, "skipped": f_skip},
        "geoconfirmed": {"new": g_new, "skipped": g_skip},
        "total_new":    f_new + g_new,
    }
    log.info("=" * 60)
    log.info(f"DONE  total_new={summary['total_new']}")
    log.info("=" * 60)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hours", type=int, default=settings.SCRAPE_LOOKBACK_HOURS,
        help=f"Lookback window in hours (default: {settings.SCRAPE_LOOKBACK_HOURS})"
    )
    args = parser.parse_args()
    since = datetime.now(timezone.utc) - timedelta(hours=args.hours)
    run(since)


if __name__ == "__main__":
    main()
