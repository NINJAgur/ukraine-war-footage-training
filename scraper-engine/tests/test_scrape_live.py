"""
test_scrape_live.py — Phase 1 end-to-end scrape test.

Covers three things:
  1. Funker530 — REST API fetch (Ukraine categoryId=16, video URL resolution)
  2. GeoConfirmed — REST API fetch (returns real video incidents)
  3. DB write — inserts Clip rows into PostgreSQL and verifies they exist

Run from repo root:
    cd scraper-engine && python tests/test_scrape_live.py
"""
import sys
import os
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("test_scrape_live")

# Test lives in scraper-engine/tests/ — parent is scraper-engine/ (the importable package root)
SCRAPER_ENGINE_DIR = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, SCRAPER_ENGINE_DIR)

# Module-level cache — populated by the individual tests, reused by test_db_write
_funker_posts: list[dict] = []
_geo_incidents: list[dict] = []


# ── Funker530 ─────────────────────────────────────────────────────────

def test_funker530() -> None:
    logger.info("=" * 60)
    logger.info("TEST: Funker530 — REST API Ukraine video post fetch")
    logger.info("=" * 60)

    from tasks.scrape_funker530 import fetch_ukraine_posts

    global _funker_posts
    posts = fetch_ukraine_posts(max_count=5)
    _funker_posts = posts
    logger.info(f"Funker530: fetched {len(posts)} Ukraine video posts")
    for p in posts:
        logger.info(
            f"  [{p['url_hash'][:8]}] {p['page_url'][:70]}\n"
            f"    title={p['title'][:80]!r}"
        )

    assert len(posts) > 0, "Funker530: expected ≥1 Ukraine video post — got 0"
    for p in posts:
        assert p["video_url"], f"Post {p['page_url']} has no video URL"
    logger.info("PASS: Funker530\n")


# ── GeoConfirmed ──────────────────────────────────────────────────────

def test_geoconfirmed() -> None:
    logger.info("=" * 60)
    logger.info("TEST: GeoConfirmed — REST API video incident fetch")
    logger.info("=" * 60)

    from tasks.scrape_geoconfirmed import extract_video_incidents

    global _geo_incidents
    incidents = extract_video_incidents(max_incidents=5)
    _geo_incidents = incidents
    logger.info(f"GeoConfirmed: fetched {len(incidents)} video incidents")
    logger.info("  Filter: origin='VID' on GeoConfirmed Ukraine map + equipment keyword preference")
    for inc in incidents:
        eq = inc.get("equipment_match")
        tier = f"equipment_match='{eq}'" if eq else "(no equipment match)"
        logger.info(
            f"  [{inc['url_hash'][:8]}] {tier}\n"
            f"    url={inc['url'][:80]}\n"
            f"    title={inc['title'][:80]!r}"
        )

    assert len(incidents) > 0, "GeoConfirmed: expected ≥1 video incident — got 0"
    logger.info("PASS: GeoConfirmed\n")


# ── DB write ──────────────────────────────────────────────────────────

def test_db_write() -> None:
    """
    Run both scrapers end-to-end and write Clip rows to PostgreSQL.
    Verifies rows exist in DB after insertion.
    """
    logger.info("=" * 60)
    logger.info("TEST: DB write — Funker530 + GeoConfirmed → PostgreSQL")
    logger.info("=" * 60)

    from sqlalchemy.dialects.postgresql import insert as pg_insert
    from db.models import Clip, ClipSource, ClipStatus
    from db.session import get_session

    # Reuse data already fetched by the earlier tests — no second HTTP round-trip
    funker_posts = _funker_posts
    geo_incidents = _geo_incidents
    logger.info(f"Funker530 posts available: {len(funker_posts)}")
    logger.info(f"GeoConfirmed incidents available: {len(geo_incidents)}")

    # ── Insert all into DB ────────────────────────────────────────────
    new_funker = new_geo = skipped = 0

    with get_session() as session:
        for post in funker_posts:
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
            row = session.execute(stmt).fetchone()
            if row:
                new_funker += 1
                logger.info(f"  INSERT funker530 Clip id={row[0]}  {post['page_url'][:70]}")
            else:
                skipped += 1

        for inc in geo_incidents:
            stmt = (
                pg_insert(Clip)
                .values(
                    url=inc["url"],
                    url_hash=inc["url_hash"],
                    source=ClipSource.GEOCONFIRMED,
                    title=inc["title"] or None,
                    description=inc["description"] or None,
                    published_at=inc["published_at"],
                    status=ClipStatus.PENDING,
                )
                .on_conflict_do_nothing(index_elements=["url_hash"])
                .returning(Clip.id)
            )
            row = session.execute(stmt).fetchone()
            if row:
                new_geo += 1
                logger.info(f"  INSERT geoconfirmed Clip id={row[0]}  {inc['url'][:70]}")
            else:
                skipped += 1

    # ── Verify rows in DB ─────────────────────────────────────────────
    with get_session() as session:
        total = session.query(Clip).count()
        f_count = session.query(Clip).filter(Clip.source == ClipSource.FUNKER530).count()
        g_count = session.query(Clip).filter(Clip.source == ClipSource.GEOCONFIRMED).count()

    logger.info(f"DB state: total={total}  funker530={f_count}  geoconfirmed={g_count}")
    logger.info(f"Inserted this run: funker530={new_funker}  geoconfirmed={new_geo}  skipped={skipped}")

    assert total > 0, "Expected ≥1 Clip in DB after scrape — got 0"
    logger.info("PASS: DB write\n")


# ── Download all scraped videos ───────────────────────────────────────

def _download_clip(clip_id: int, video_url: str, source_label: str, download_fn, output_fn) -> bool:
    """Download one clip; returns True on success, False on failure (logs error)."""
    from pathlib import Path
    from db.models import Clip, ClipStatus
    from db.session import get_session

    output_path = output_fn(video_url, "clip")
    try:
        meta = download_fn(video_url, output_path)
    except Exception as exc:
        logger.error(f"[{source_label}] clip_id={clip_id} download failed: {exc}")
        with get_session() as session:
            clip = session.get(Clip, clip_id)
            if clip:
                clip.status = ClipStatus.ERROR
                clip.error_message = str(exc)[:1000]
        return False

    file_path = Path(meta["file_path"])
    if not file_path.exists():
        logger.error(f"[{source_label}] clip_id={clip_id} file not on disk: {file_path}")
        return False

    size_mb = file_path.stat().st_size / 1024 / 1024
    logger.info(
        f"[{source_label}] clip_id={clip_id} saved: {file_path.name}  "
        f"({size_mb:.1f} MB  {meta['duration_seconds']}s  {meta['width']}x{meta['height']})"
    )

    with get_session() as session:
        clip = session.get(Clip, clip_id)
        if clip:
            clip.status = ClipStatus.DOWNLOADED
            clip.file_path = str(file_path)
            clip.duration_seconds = meta["duration_seconds"]
            clip.width = meta["width"]
            clip.height = meta["height"]
    return True


def test_download_video() -> None:
    """
    Download ALL scraped clips (all Funker530 + all GeoConfirmed rows in DB).
    Each video is saved to scraper-engine/media/raw/<source>/<hash>_clip.mp4.
    Clip.status is updated to DOWNLOADED or ERROR per row.
    Asserts ≥1 successful download per source.
    """
    logger.info("=" * 60)
    logger.info("TEST: Download all Funker530 + GeoConfirmed clips → media/raw/")
    logger.info("=" * 60)

    from db.models import Clip, ClipSource, ClipStatus
    from db.session import get_session
    from tasks.scrape_geoconfirmed import _download_video as geo_dl, get_output_path as geo_path
    from tasks.scrape_funker530 import _download_video as f530_dl, get_output_path as f530_path

    if not _funker_posts:
        raise AssertionError("_funker_posts cache is empty — run test_funker530 first")

    # Build page_url → video_url map from in-memory cache
    page_to_video: dict[str, str] = {p["page_url"]: p["video_url"] for p in _funker_posts}

    # ── Funker530 ─────────────────────────────────────────────────────────
    with get_session() as session:
        f_clips = (
            session.query(Clip)
            .filter(Clip.source == ClipSource.FUNKER530)
            .order_by(Clip.id)
            .all()
        )
        f_rows = [(c.id, c.url, c.title) for c in f_clips]
        for c in f_clips:
            c.status = ClipStatus.DOWNLOADING

    f_ok = f_fail = 0
    for clip_id, page_url, title in f_rows:
        video_url = page_to_video.get(page_url)
        if not video_url:
            logger.warning(f"[funker530] clip_id={clip_id} no video URL in cache — skip")
            f_fail += 1
            continue
        logger.info(f"[funker530] clip_id={clip_id}  {title!r}")
        if _download_clip(clip_id, video_url, "funker530", f530_dl, f530_path):
            f_ok += 1
        else:
            f_fail += 1

    # ── GeoConfirmed ──────────────────────────────────────────────────────
    with get_session() as session:
        g_clips = (
            session.query(Clip)
            .filter(Clip.source == ClipSource.GEOCONFIRMED)
            .order_by(Clip.id)
            .all()
        )
        g_rows = [(c.id, c.url, c.title) for c in g_clips]
        for c in g_clips:
            c.status = ClipStatus.DOWNLOADING

    g_ok = g_fail = 0
    for clip_id, video_url, title in g_rows:
        logger.info(f"[geoconfirmed] clip_id={clip_id}  {(title or '')[:80]!r}")
        if _download_clip(clip_id, video_url, "geoconfirmed", geo_dl, geo_path):
            g_ok += 1
        else:
            g_fail += 1

    logger.info(
        f"Download summary: funker530={f_ok} ok / {f_fail} fail  |  "
        f"geoconfirmed={g_ok} ok / {g_fail} fail"
    )
    assert f_ok >= 1, f"Expected ≥1 Funker530 download — got 0"
    assert g_ok >= 1, f"Expected ≥1 GeoConfirmed download — got 0"
    logger.info("PASS: download_video\n")


# ── Pre-test cleanup ──────────────────────────────────────────────────

def _cleanup() -> None:
    """Wipe DB clips table and raw video dirs before each full run."""
    import shutil
    from pathlib import Path

    sys.path.insert(0, SCRAPER_ENGINE_DIR)
    from db.session import get_session
    from db.models import Clip

    with get_session() as session:
        # Delete datasets first to avoid FK violation (datasets.clip_id → clips.id)
        from sqlalchemy import text
        session.execute(text("DELETE FROM datasets WHERE clip_id IN (SELECT id FROM clips)"))
        deleted = session.query(Clip).delete()
        logger.info(f"Cleanup: deleted {deleted} Clip rows from DB")

    # scraper-engine/media/raw/ is the canonical raw video location
    scraper_engine_dir = Path(__file__).resolve().parent.parent
    for subdir in ["funker530", "geoconfirmed"]:
        raw_dir = scraper_engine_dir / "media" / "raw" / subdir
        if raw_dir.exists():
            shutil.rmtree(raw_dir)
            logger.info(f"Cleanup: removed {raw_dir}")


# ── Runner ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _cleanup()

    passed: list[str] = []
    failed: list[str] = []

    for name, fn in [
        ("funker530_fetch", test_funker530),
        ("geoconfirmed_fetch", test_geoconfirmed),
        ("db_write", test_db_write),
        ("download_all", test_download_video),
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
