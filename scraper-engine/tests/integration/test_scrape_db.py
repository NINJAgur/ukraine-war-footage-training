"""
Integration test: scraper task inserts Clip rows that are retrievable from DB.
Requires: live PostgreSQL at DATABASE_SYNC_URL, internet access for scraper.

Run with:
    pytest -m "integration and network" tests/integration/test_scrape_db.py
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from db.models import Clip
from db.session import get_session


@pytest.mark.integration
@pytest.mark.network
def test_db_connection_returns_clip_model():
    """DB session can query the clips table without error."""
    with get_session() as session:
        count = session.query(Clip).count()
    assert isinstance(count, int)
    assert count >= 0


@pytest.mark.integration
@pytest.mark.network
def test_inserted_clip_is_retrievable():
    """A manually inserted Clip row persists and can be queried by url_hash."""
    import hashlib
    from db.models import ClipSource, ClipStatus

    test_url = "https://test.example.com/scraper-integration-test-clip"
    url_hash = hashlib.sha256(test_url.encode()).hexdigest()
    inserted = False

    try:
        with get_session() as session:
            if session.query(Clip).filter_by(url_hash=url_hash).first() is None:
                session.add(Clip(
                    url=test_url, url_hash=url_hash,
                    source=ClipSource.FUNKER530, status=ClipStatus.PENDING,
                    title="Integration test clip",
                ))
                inserted = True

        with get_session() as session:
            result = session.query(Clip).filter_by(url_hash=url_hash).first()
            assert result is not None
            assert result.url == test_url
            assert result.url_hash == url_hash
    finally:
        if inserted:
            with get_session() as session:
                clip = session.query(Clip).filter_by(url_hash=url_hash).first()
                if clip:
                    session.delete(clip)


@pytest.mark.integration
@pytest.mark.network
def test_clip_status_can_be_updated():
    """Clip status update persists across sessions."""
    import hashlib
    from db.models import ClipSource, ClipStatus

    test_url = "https://test.example.com/scraper-status-update-test"
    url_hash = hashlib.sha256(test_url.encode()).hexdigest()
    inserted = False

    try:
        with get_session() as session:
            if session.query(Clip).filter_by(url_hash=url_hash).first() is None:
                session.add(Clip(
                    url=test_url, url_hash=url_hash,
                    source=ClipSource.GEOCONFIRMED, status=ClipStatus.PENDING,
                ))
                inserted = True

        with get_session() as session:
            clip = session.query(Clip).filter_by(url_hash=url_hash).first()
            clip.status = ClipStatus.DOWNLOADED

        with get_session() as session:
            clip = session.query(Clip).filter_by(url_hash=url_hash).first()
            assert clip.status == ClipStatus.DOWNLOADED
    finally:
        if inserted:
            with get_session() as session:
                clip = session.query(Clip).filter_by(url_hash=url_hash).first()
                if clip:
                    session.delete(clip)
