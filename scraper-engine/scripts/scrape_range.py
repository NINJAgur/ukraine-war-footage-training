"""
scrape_range.py

Scrape a bounded date range from both sources and dispatch downloads.
Posts outside [from_date, to_date] are skipped before DB insertion.

Usage:
    cd scraper-engine && python scripts/scrape_range.py --from 2026-05-12 --to 2026-05-14
"""
import sys
import logging
import argparse
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("scrape_range")

SCRAPER_ENGINE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRAPER_ENGINE_DIR))

from sqlalchemy.dialects.postgresql import insert as pg_insert

from db.models import Clip, ClipSource, ClipStatus
from db.session import get_session
from tasks.scrape_funker530 import fetch_ukraine_posts_since, download_funker530_video
from tasks.scrape_geoconfirmed import extract_video_incidents_since, download_geoconfirmed_video


def _insert_clip(session, values: dict):
    stmt = (
        pg_insert(Clip)
        .values(**values)
        .on_conflict_do_nothing(index_elements=["url_hash"])
        .returning(Clip.id)
    )
    row = session.execute(stmt).fetchone()
    return row[0] if row else None


def _in_range(published_at, from_dt: datetime, to_dt: datetime) -> bool:
    if published_at is None:
        return False
    ts = published_at.replace(tzinfo=None) if published_at.tzinfo else published_at
    return from_dt.replace(tzinfo=None) <= ts <= to_dt.replace(tzinfo=None)


def run(from_dt: datetime, to_dt: datetime) -> dict:
    log.info("=" * 60)
    log.info(f"RANGE SCRAPE  {from_dt.strftime('%Y-%m-%d')} → {to_dt.strftime('%Y-%m-%d')}")
    log.info("=" * 60)

    # ── Funker530 ─────────────────────────────────────────────────────
    log.info("Scraping Funker530...")
    f_posts = fetch_ukraine_posts_since(from_dt)
    f_new = f_skip = f_out_of_range = 0

    with get_session() as session:
        for post in f_posts:
            if not _in_range(post.get("published_at"), from_dt, to_dt):
                f_out_of_range += 1
                continue
            clip_id = _insert_clip(session, {
                "url":          post["page_url"],
                "url_hash":     post["url_hash"],
                "source":       ClipSource.FUNKER530,
                "title":        post["title"] or None,
                "description":  post["description"] or None,
                "published_at": post["published_at"],
                "status":       ClipStatus.PENDING,
                **post["scores"],
            })
            if clip_id:
                f_new += 1
                download_funker530_video.delay(
                    clip_id=clip_id,
                    video_url=post["video_url"],
                    page_url=post["page_url"],
                )
            else:
                f_skip += 1

    log.info(f"Funker530: {f_new} new  {f_skip} already in DB  {f_out_of_range} out of range")

    # ── GeoConfirmed ──────────────────────────────────────────────────
    log.info("Scraping GeoConfirmed...")
    g_incidents = extract_video_incidents_since(from_dt)
    g_new = g_skip = g_out_of_range = 0

    with get_session() as session:
        for inc in g_incidents:
            if not _in_range(inc.get("published_at"), from_dt, to_dt):
                g_out_of_range += 1
                continue
            clip_id = _insert_clip(session, {
                "url":          inc["url"],
                "url_hash":     inc["url_hash"],
                "source":       ClipSource.GEOCONFIRMED,
                "title":        inc["title"] or None,
                "description":  inc["description"] or None,
                "published_at": inc["published_at"],
                "status":       ClipStatus.PENDING,
                **inc["scores"],
            })
            if clip_id:
                g_new += 1
                download_geoconfirmed_video.delay(clip_id=clip_id, video_url=inc["url"])
            else:
                g_skip += 1

    log.info(f"GeoConfirmed: {g_new} new  {g_skip} already in DB  {g_out_of_range} out of range")

    total_new = f_new + g_new
    log.info("=" * 60)
    log.info(f"DONE  total_new={total_new}")
    log.info("=" * 60)
    return {"funker530": f_new, "geoconfirmed": g_new, "total_new": total_new}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="from_date", required=True, help="Start date inclusive (YYYY-MM-DD)")
    parser.add_argument("--to",   dest="to_date",   required=True, help="End date inclusive (YYYY-MM-DD)")
    args = parser.parse_args()
    from_dt = datetime.strptime(args.from_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    to_dt   = datetime.strptime(args.to_date,   "%Y-%m-%d").replace(hour=23, minute=59, second=59, tzinfo=timezone.utc)
    run(from_dt, to_dt)


if __name__ == "__main__":
    main()
