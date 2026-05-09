"""
test_scrape_24h.py — Date-window scrape test (last 24 hours).

Validates that both scrapers work correctly in daily/date mode —
the same mode used by scrape_daily.py and Celery Beat.
0 results is a valid outcome if nothing was published in the last 24h.

Run from repo root:
    cd scraper-engine && python tests/test_scrape_24h.py
    cd scraper-engine && python tests/test_scrape_24h.py --hours 48
"""
import sys
import logging
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("test_scrape_24h")

SCRAPER_ENGINE_DIR = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, SCRAPER_ENGINE_DIR)


def run_funker530_date(since: datetime) -> list[dict]:
    logger.info("=" * 60)
    logger.info(f"TEST: Funker530 — date window since {since.strftime('%Y-%m-%d %H:%M UTC')}")
    logger.info("=" * 60)

    from tasks.scrape_funker530 import fetch_ukraine_posts_since
    posts = fetch_ukraine_posts_since(since)

    logger.info(f"Funker530: {len(posts)} posts accepted")
    for p in posts:
        logger.info(
            f"  [{p['url_hash'][:8]}] published={p.get('published_at')}\n"
            f"    title={p['title'][:80]!r}"
        )

    logger.info("PASS: Funker530 24h (0 results is acceptable)\n")
    return posts


def run_geoconfirmed_date(since: datetime) -> list[dict]:
    logger.info("=" * 60)
    logger.info(f"TEST: GeoConfirmed — date window since {since.strftime('%Y-%m-%d %H:%M UTC')}")
    logger.info("=" * 60)

    from tasks.scrape_geoconfirmed import extract_video_incidents_since
    incidents = extract_video_incidents_since(since)

    logger.info(f"GeoConfirmed: {len(incidents)} incidents accepted")
    for inc in incidents:
        logger.info(
            f"  [{inc['url_hash'][:8]}] published={inc.get('published_at')}\n"
            f"    title={inc['title'][:80]!r}"
        )

    logger.info("PASS: GeoConfirmed 24h (0 results is acceptable)\n")
    return incidents


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=int, default=24, help="Lookback window in hours (default: 24)")
    args = parser.parse_args()

    since = datetime.now(timezone.utc) - timedelta(hours=args.hours)

    passed: list[str] = []
    failed: list[str] = []

    for name, fn in [
        ("funker530_date", lambda: run_funker530_date(since)),
        ("geoconfirmed_date", lambda: run_geoconfirmed_date(since)),
    ]:
        try:
            fn()
            passed.append(name)
        except Exception as exc:
            logger.error(f"FAIL: {name} — {exc}", exc_info=True)
            failed.append(name)

    logger.info("=" * 60)
    logger.info(f"Results: {len(passed)} passed, {len(failed)} failed")
    if failed:
        logger.error(f"Failed: {failed}")
        sys.exit(1)
    else:
        logger.info("All tests passed!")
