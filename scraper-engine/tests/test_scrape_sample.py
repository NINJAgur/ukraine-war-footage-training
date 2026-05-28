"""
test_scrape_sample.py — Count-based scrape test (N clips from each source).

Covers three things:
  1. Funker530 — REST API fetch (Ukraine categoryId=16, video URL resolution)
  2. GeoConfirmed — REST API fetch (returns real video incidents)
  3. DB write — inserts Clip rows into PostgreSQL and verifies they exist
  4. Download — downloads clips inserted by this test only, cleans up in finally

Run from repo root:
    cd scraper-engine && python tests/test_scrape_sample.py
"""
import sys
import logging
from pathlib import Path

import pytest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("test_scrape_sample")

# Test lives in scraper-engine/tests/ — parent is scraper-engine/ (the importable package root)
SCRAPER_ENGINE_DIR = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, SCRAPER_ENGINE_DIR)


# ── Funker530 ─────────────────────────────────────────────────────────

@pytest.mark.network
def test_funker530() -> None:
    logger.info("=" * 60)
    logger.info("TEST: Funker530 — REST API Ukraine video post fetch")
    logger.info("=" * 60)

    from tasks.scrape_funker530 import fetch_ukraine_posts_sample

    posts = fetch_ukraine_posts_sample(max_count=5)
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

@pytest.mark.network
def test_geoconfirmed() -> None:
    logger.info("=" * 60)
    logger.info("TEST: GeoConfirmed — REST API video incident fetch")
    logger.info("=" * 60)

    from tasks.scrape_geoconfirmed import extract_video_incidents_sample

    incidents = extract_video_incidents_sample(max_incidents=5)
    logger.info(f"GeoConfirmed: fetched {len(incidents)} video incidents")
    logger.info("  Filter: origin='VID' on GeoConfirmed Ukraine map + equipment keyword preference")
    for inc in incidents:
        scores = inc.get("scores", {})
        logger.info(
            f"  [{inc['url_hash'][:8]}] scores={scores}\n"
            f"    url={inc['url'][:80]}\n"
            f"    title={inc['title'][:80]!r}"
        )

    assert len(incidents) > 0, "GeoConfirmed: expected ≥1 video incident — got 0"
    logger.info("PASS: GeoConfirmed\n")


# ── DB write ──────────────────────────────────────────────────────────

@pytest.mark.network
def test_db_write() -> None:
    """
    Fetch posts fresh inside the test, insert Clip rows, verify only those IDs,
    then delete them in finally. Does not touch any pre-existing DB rows.
    """
    logger.info("=" * 60)
    logger.info("TEST: DB write — Funker530 + GeoConfirmed → PostgreSQL")
    logger.info("=" * 60)

    from tasks.scrape_funker530 import fetch_ukraine_posts_sample
    from tasks.scrape_geoconfirmed import extract_video_incidents_sample
    from sqlalchemy.dialects.postgresql import insert as pg_insert
    from db.models import Clip, ClipSource, ClipStatus
    from db.session import get_session

    funker_posts = fetch_ukraine_posts_sample(max_count=5)
    geo_incidents = extract_video_incidents_sample(max_incidents=5)
    logger.info(f"Funker530 posts fetched: {len(funker_posts)}")
    logger.info(f"GeoConfirmed incidents fetched: {len(geo_incidents)}")

    inserted_ids: list[int] = []

    try:
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
                        **post.get("scores", {}),
                    )
                    .on_conflict_do_nothing(index_elements=["url_hash"])
                    .returning(Clip.id)
                )
                row = session.execute(stmt).fetchone()
                if row:
                    new_funker += 1
                    inserted_ids.append(row[0])
                    logger.info(f"  INSERT funker530 Clip id={row[0]}  {post['page_url'][:70]}")
                else:
                    skipped += 1
                    logger.info(f"  SKIP funker530 (conflict)  {post['page_url'][:70]}")

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
                        **inc.get("scores", {}),
                    )
                    .on_conflict_do_nothing(index_elements=["url_hash"])
                    .returning(Clip.id)
                )
                row = session.execute(stmt).fetchone()
                if row:
                    new_geo += 1
                    inserted_ids.append(row[0])
                    logger.info(f"  INSERT geoconfirmed Clip id={row[0]}  {inc['url'][:70]}")
                else:
                    skipped += 1
                    logger.info(f"  SKIP geoconfirmed (conflict)  {inc['url'][:70]}")

        logger.info(
            f"Inserted this run: funker530={new_funker}  geoconfirmed={new_geo}  skipped={skipped}"
        )

        # Verify only the rows this test inserted
        with get_session() as session:
            found = session.query(Clip).filter(Clip.id.in_(inserted_ids)).count()
        logger.info(f"Verified {found}/{len(inserted_ids)} inserted rows exist in DB")
        assert found == len(inserted_ids), f"Expected {len(inserted_ids)} rows, found {found}"
        assert len(inserted_ids) > 0, "Expected ≥1 insert — got 0 (all conflicts?)"

    finally:
        if inserted_ids:
            with get_session() as session:
                session.query(Clip).filter(Clip.id.in_(inserted_ids)).delete(
                    synchronize_session=False
                )
            logger.info(f"Cleanup: deleted {len(inserted_ids)} test Clip rows {inserted_ids}")

    logger.info("PASS: DB write\n")


# ── Download all scraped videos ───────────────────────────────────────

def _download_clip(clip_id: int, video_url: str, source_label: str, download_fn, output_fn) -> "Path | None":
    """Download one clip; returns the file Path on success, None on failure."""
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
        return None

    file_path = Path(meta["file_path"])
    if not file_path.exists():
        logger.error(f"[{source_label}] clip_id={clip_id} file not on disk: {file_path}")
        return None

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
    return file_path


@pytest.mark.network
def test_download_video() -> None:
    """
    Fetch posts fresh, insert clips, download ONLY those clips (filtered by ID),
    then delete all inserted DB rows and downloaded files in finally.
    """
    logger.info("=" * 60)
    logger.info("TEST: Download Funker530 + GeoConfirmed clips → media/<source>/")
    logger.info("=" * 60)

    from tasks.scrape_funker530 import fetch_ukraine_posts_sample
    from tasks.scrape_geoconfirmed import extract_video_incidents_sample
    from sqlalchemy.dialects.postgresql import insert as pg_insert
    from db.models import Clip, ClipSource, ClipStatus
    from db.session import get_session
    from tasks.scrape_geoconfirmed import _download_video as geo_dl, get_output_path as geo_path
    from tasks.scrape_funker530 import _download_video as f530_dl, get_output_path as f530_path

    funker_posts = fetch_ukraine_posts_sample(max_count=5)
    geo_incidents = extract_video_incidents_sample(max_incidents=5)
    logger.info(f"Funker530 posts fetched: {len(funker_posts)}")
    logger.info(f"GeoConfirmed incidents fetched: {len(geo_incidents)}")

    # page_url → video_url map for funker (video_url is separate from page_url)
    page_to_video: dict[str, str] = {p["page_url"]: p["video_url"] for p in funker_posts}

    inserted_ids: list[int] = []
    downloaded_files: list[Path] = []

    try:
        # ── Insert rows ───────────────────────────────────────────────
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
                        **post.get("scores", {}),
                    )
                    .on_conflict_do_nothing(index_elements=["url_hash"])
                    .returning(Clip.id)
                )
                row = session.execute(stmt).fetchone()
                if row:
                    inserted_ids.append(row[0])
                    logger.info(f"  INSERT funker530 Clip id={row[0]}  {post['page_url'][:70]}")
                else:
                    logger.info(f"  SKIP funker530 (conflict)  {post['page_url'][:70]}")

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
                        **inc.get("scores", {}),
                    )
                    .on_conflict_do_nothing(index_elements=["url_hash"])
                    .returning(Clip.id)
                )
                row = session.execute(stmt).fetchone()
                if row:
                    inserted_ids.append(row[0])
                    logger.info(f"  INSERT geoconfirmed Clip id={row[0]}  {inc['url'][:70]}")
                else:
                    logger.info(f"  SKIP geoconfirmed (conflict)  {inc['url'][:70]}")

        logger.info(f"Inserted {len(inserted_ids)} rows: {inserted_ids}")

        # ── Mark only our rows as DOWNLOADING ────────────────────────
        with get_session() as session:
            session.query(Clip).filter(Clip.id.in_(inserted_ids)).update(
                {Clip.status: ClipStatus.DOWNLOADING}, synchronize_session=False
            )

        # ── Fetch only our rows, split by source ─────────────────────
        with get_session() as session:
            our_clips = (
                session.query(Clip)
                .filter(Clip.id.in_(inserted_ids))
                .order_by(Clip.id)
                .all()
            )
            rows = [(c.id, c.url, c.source, c.title) for c in our_clips]

        # ── Download ──────────────────────────────────────────────────
        f_ok = f_fail = 0
        g_ok = g_fail = 0

        for clip_id, url, source, title in rows:
            if source == ClipSource.FUNKER530:
                video_url = page_to_video.get(url)
                if not video_url:
                    logger.warning(f"[funker530] clip_id={clip_id} no video URL in cache — skip")
                    f_fail += 1
                    continue
                logger.info(f"[funker530] clip_id={clip_id}  {(title or '')[:80]!r}")
                result = _download_clip(clip_id, video_url, "funker530", f530_dl, f530_path)
                if result:
                    downloaded_files.append(result)
                    f_ok += 1
                else:
                    f_fail += 1

            elif source == ClipSource.GEOCONFIRMED:
                # For geoconfirmed, c.url IS the video url
                logger.info(f"[geoconfirmed] clip_id={clip_id}  {(title or '')[:80]!r}")
                result = _download_clip(clip_id, url, "geoconfirmed", geo_dl, geo_path)
                if result:
                    downloaded_files.append(result)
                    g_ok += 1
                else:
                    g_fail += 1

        logger.info(
            f"Download summary: funker530={f_ok} ok / {f_fail} fail  |  "
            f"geoconfirmed={g_ok} ok / {g_fail} fail"
        )
        assert f_ok >= 1, "Expected ≥1 Funker530 download — got 0"
        assert g_ok >= 1, "Expected ≥1 GeoConfirmed download — got 0"

    finally:
        if inserted_ids:
            with get_session() as session:
                session.query(Clip).filter(Clip.id.in_(inserted_ids)).delete(
                    synchronize_session=False
                )
            logger.info(f"Cleanup: deleted {len(inserted_ids)} test Clip rows {inserted_ids}")

        for fp in downloaded_files:
            try:
                fp.unlink(missing_ok=True)
                logger.info(f"Cleanup: deleted file {fp.name}")
            except Exception as exc:
                logger.warning(f"Cleanup: could not delete {fp}: {exc}")

    logger.info("PASS: download_video\n")


# ── Runner ────────────────────────────────────────────────────────────

if __name__ == "__main__":
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
